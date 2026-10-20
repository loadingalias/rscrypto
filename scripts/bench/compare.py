"""Score retained benchmark runs against the comparisons that the catalog declares.

Each catalog comparison names one task, the benchmark target that measures it, and a case pattern
with `implementation` and `input` groups (and an optional `variant`). One comparison point is one
(comparison, variant, input) on one platform. It needs exactly one canonical `rscrypto` row from
the comparison's target and at least one external row. External rows may also come from the
comparison's `competitor_benches`, such as a competitor's best supported build. Rows whose
implementation starts with `rscrypto-` are rscrypto variants; they never enter a ratio.

ratio = fastest external median / rscrypto median. A ratio above 1 favors rscrypto.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics
import sys

# Embedded Windows Python omits the script directory from its import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_catalog import ARCHITECTURES, MACHINES, applies, architecture, case_class, load_catalog
from execution import read_json
from measure import verify

# Ratios from 0.95x through 1.05x are ties.
TIE = 0.05
CANONICAL = "rscrypto"


class ComparisonError(ValueError):
  pass


def classify(ratio: float) -> str:
  if ratio > 1 + TIE:
    return "faster"
  if ratio < 1 - TIE:
    return "slower"
  return "tie"


def geomean(values: list[float]) -> float:
  return math.exp(statistics.fmean(math.log(value) for value in values))


def bench_of(catalog: dict, entry: dict) -> str:
  """Return the catalog target that built one plan entry."""
  if "bench" in entry:
    return entry["bench"]
  # Plans written before the runner recorded the target name: match the build configuration.
  command = entry["artifact"]["command"]
  features = set(command[command.index("--features") + 1].split(","))
  manifest = command[command.index("--manifest-path") + 1] if "--manifest-path" in command else None
  matches = [name for name, bench in catalog["benches"].items()
             if bench["kind"] == "criterion" and bench["binary"] == entry["binary"]
             and bench.get("manifest") == manifest
             and bench.get("rustflags", []) == entry.get("rustflags", [])
             and set(bench["features"]) in (features, features - {"diag"})]
  if len(matches) != 1:
    raise ComparisonError(f"cannot identify the catalog target of {entry['home']}: {matches}")
  return matches[0]


def load_platform(catalog: dict, label: str, roots: list[Path]) -> tuple[dict[tuple[str, str], dict], str]:
  """Read the median estimates of every measured case of one platform, keyed by (target, case), and its architecture."""
  records = {}
  hosts = []
  for root in roots:
    if "state=complete" not in (root / "status.txt").read_text().splitlines():
      raise ComparisonError(f"{root} is not a completed run")
    measured = verify(root)
    for entry in read_json(root / "plan.json"):
      bench = bench_of(catalog, entry)
      hosts.append(entry["compatibility"]["host"])
      for case, path in measured[entry["home"]].items():
        median = read_json(path / "estimates.json")["median"]
        interval = median["confidence_interval"]
        key = (bench, case)
        if key in records:
          raise ComparisonError(f"{label}: {bench} {case} was measured by more than one run")
        records[key] = {"median": median["point_estimate"], "lower": interval["lower_bound"],
                        "upper": interval["upper_bound"], "binary": entry["binary"], "run": str(root),
                        "quick": entry["settings"].get("quick", False)}
  if any(host != hosts[0] for host in hosts):
    raise ComparisonError(f"{label}: runs come from different hosts")
  machine = hosts[0]["machine"].lower()
  return records, MACHINES.get(machine, machine)


def points(catalog: dict, label: str, records: dict, arch: str | None = None,
           standalone: list[dict] | None = None) -> tuple[list[dict], list[str]]:
  """Pair each comparison point's canonical rscrypto row with its fastest external row.

  With an architecture, a competitor build that applies there but has no rows makes the
  comparison incomplete: without it, the fastest external would come from a weaker build.
  """
  rows = []
  problems = []
  for name, comparison in catalog["comparisons"].items():
    pattern = re.compile(comparison["cases"])
    sources = [comparison["bench"], *comparison.get("competitor_benches", [])]
    measured = {bench for bench, case in records if bench in sources and pattern.fullmatch(case)}
    if arch and comparison["bench"] in measured:
      absent = [bench for bench in sources[1:] if applies(catalog["benches"][bench], arch) and bench not in measured]
      if absent:
        problems.append(f"{label}: {name} needs the competitor build {', '.join(absent)}")
        continue
    groups = defaultdict(lambda: {"rscrypto": [], "external": []})
    for (bench, case), record in records.items():
      if bench not in sources or not (match := pattern.fullmatch(case)):
        continue
      implementation = match["implementation"]
      key = (match.groupdict().get("variant") or "", match["input"])
      if implementation == CANONICAL:
        if bench == comparison["bench"]:
          groups[key]["rscrypto"].append(record | {"case": case})
      elif not implementation.startswith(CANONICAL + "-"):
        source = implementation if bench == comparison["bench"] else f"{implementation}@{bench}"
        groups[key]["external"].append(record | {"case": case, "implementation": source})
    for (variant, item), group in sorted(groups.items()):
      where = f"{label}: {name} {variant + ' ' if variant else ''}{item}"
      if arch and not applies(comparison, arch):
        if group["external"] or len(group["rscrypto"]) != 1:
          problems.append(f"{where} is declared standalone on {arch}; expected one rscrypto row and no external rows")
        elif standalone is not None:
          standalone.append({"platform": label, "comparison": name, "case": group["rscrypto"][0]["case"],
                             "reason": comparison["availability"]})
        continue
      if len(group["rscrypto"]) != 1 or not group["external"]:
        problems.append(f"{where} needs one rscrypto row and an external row; found "
                        f"{len(group['rscrypto'])} and {len(group['external'])}")
        continue
      ours = group["rscrypto"][0]
      best = min(group["external"], key=lambda record: record["median"])
      ratio = best["median"] / ours["median"]
      rows.append({
        "platform": label, "comparison": name, "family": comparison["family"],
        "headline": comparison["headline"], "variant": variant, "input": item,
        "rscrypto_case": ours["case"], "rscrypto_ns": ours["median"],
        "external": best["implementation"], "external_case": best["case"], "external_ns": best["median"],
        "ratio": ratio,
        # Bounds from the two medians' confidence intervals; wider than either alone.
        "ratio_lower": best["lower"] / ours["upper"], "ratio_upper": best["upper"] / ours["lower"],
        "class": classify(ratio),
        "externals": sorted(record["implementation"] for record in group["external"]),
      })
  return rows, problems


def undeclared(catalog: dict, cases: dict[str, set[str]]) -> list[str]:
  """Find comparison-shaped groups, keyed by binary, that no catalog comparison claims."""
  binaries = {name: bench["binary"] for name, bench in catalog["benches"].items()}
  claims = defaultdict(list)
  for comparison in catalog["comparisons"].values():
    claims[binaries[comparison["bench"]]].append(re.compile(comparison["cases"]))
  missing = []
  for binary, names in sorted(cases.items()):
    ordinary = {case for case in names if case_class(catalog, binary, case) != "diagnostic"}
    for case in sorted(ordinary):
      parts = case.split("/")
      for index, part in enumerate(parts):
        if part != CANONICAL:
          continue
        # A sibling row that differs only in the implementation segment makes this a comparison.
        siblings = [other for other in ordinary if other != case
                    and (other_parts := other.split("/"))[:index] == parts[:index]
                    and other_parts[index + 1:] == parts[index + 1:] and len(other_parts) == len(parts)
                    and not other_parts[index].startswith(CANONICAL)]
        if siblings and not any(pattern.fullmatch(case) for pattern in claims[binary]):
          missing.append(f"{binary}: {case} has external rows but no catalog comparison")
  return missing


def listed_cases(path: Path) -> dict[str, set[str]]:
  """Read `just bench --list` output: `[class] binary case` per line."""
  cases = defaultdict(set)
  for line in path.read_text().splitlines():
    if line.startswith("["):
      _, binary, case = line.split(" ", 2)
      cases[binary].add(case)
  return cases


def validate_cases(catalog: dict, cases: dict[str, set[str]], arch: str | None = None) -> list[str]:
  """Check declarations against discovered cases: every comparison matches, and every point is complete."""
  problems = undeclared(catalog, cases)
  binaries = {name: bench["binary"] for name, bench in catalog["benches"].items()}
  for name, comparison in catalog["comparisons"].items():
    binary = binaries[comparison["bench"]]
    if binary not in cases:
      continue
    # A listing names binaries, not builds, so each source sees every case of its binary.
    records = {(source, case): {"median": 1.0, "lower": 1.0, "upper": 1.0}
               for source in [comparison["bench"], *comparison.get("competitor_benches", [])]
               for case in cases.get(binaries[source], ()) if re.fullmatch(comparison["cases"], case)}
    if not records:
      problems.append(f"{name} matches no {binary} case")
      continue
    single = {"comparisons": {name: comparison}, "benches": catalog["benches"]}
    problems += points(single, "listing", records, arch or architecture())[1]
  return problems


def table(header: list[str], body: list[list[str]]) -> str:
  lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" if index == 0 else "---:" for index in range(len(header))) + " |"]
  lines += ["| " + " | ".join(row) + " |" for row in body]
  return "\n".join(lines)


def families(rows: list[dict], platforms: list[str]) -> list[list[str]]:
  body = []
  for family in sorted({row["family"] for row in rows}):
    cells = [family]
    for platform in platforms:
      ratios = [row["ratio"] for row in rows if row["family"] == family and row["platform"] == platform]
      cells.append(f"{geomean(ratios):.2f}x ({len(ratios)})" if ratios else "-")
    body.append(cells)
  return body


def report(rows: list[dict], platforms: list[str]) -> str:
  headline = [row for row in rows if row["headline"]]
  summary = []
  for platform in platforms:
    selected = [row for row in headline if row["platform"] == platform]
    if not selected:
      continue
    ratios = [row["ratio"] for row in selected]
    counts = {name: sum(row["class"] == name for row in selected) for name in ("faster", "tie", "slower")}
    summary.append([platform, str(len(selected)), str(counts["faster"]), str(counts["tie"]), str(counts["slower"]),
                    f"{statistics.median(ratios):.2f}x", f"{geomean(ratios):.2f}x"])
  other = [row for row in rows if not row["headline"]]
  sections = [
    "## Headline comparisons",
    f"Ratio = fastest external median / rscrypto median. Ties are {1 - TIE:.2f}x through {1 + TIE:.2f}x.",
    table(["Platform", "Points", "Faster", "Tie", "Slower", "Median", "Geomean"], summary),
    "## Headline geometric mean by family (points)",
    table(["Family", *platforms], families(headline, platforms)),
  ]
  if other:
    sections += ["## Other comparisons: geometric mean by family (points)",
                 table(["Family", *platforms], families(other, platforms))]
  return "\n\n".join(sections) + "\n"


def parse_runs(values: list[str]) -> dict[str, list[Path]]:
  runs = defaultdict(list)
  for value in values:
    label, separator, path = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label) or not path:
      raise ComparisonError(f"expected PLATFORM=RUN_DIRECTORY, got {value!r}")
    runs[label].append(Path(path).resolve())
  return runs


def main(arguments: list[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("runs", nargs="*", metavar="PLATFORM=RUN_DIRECTORY")
  parser.add_argument("--cases", type=Path, help="validate declarations against `just bench --list` output")
  parser.add_argument("--architecture", choices=sorted(ARCHITECTURES),
                      help="architecture of --cases (default: this host); retained runs carry their own")
  parser.add_argument("--json", type=Path, help="write every comparison point as JSON")
  parser.add_argument("--partial", action="store_true",
                      help="report incomplete points as notes, for runs whose case filter cut comparison groups")
  args = parser.parse_args(arguments)
  catalog = load_catalog()
  if args.cases:
    if args.runs or args.json:
      raise ComparisonError("--cases takes no runs and writes no JSON")
    cases = listed_cases(args.cases)
    # An empty or misformatted listing would otherwise skip every comparison and pass.
    if not cases:
      raise ComparisonError(f"{args.cases} lists no `[class] binary case` lines from `just bench --list`")
    problems = validate_cases(catalog, cases, args.architecture)
    for problem in problems:
      print(f"error: {problem}", file=sys.stderr)
    binaries = {name: bench["binary"] for name, bench in catalog["benches"].items()}
    checked = sum(binaries[comparison["bench"]] in cases for comparison in catalog["comparisons"].values())
    print(f"{checked} of {len(catalog['comparisons'])} comparisons checked against {args.cases}; "
          "the others measure binaries absent from the listing")
    return 1 if problems else 0
  if not args.runs:
    raise ComparisonError("give at least one PLATFORM=RUN_DIRECTORY")
  if args.architecture:
    raise ComparisonError("--architecture applies only to --cases")
  rows = []
  problems = []
  measured = defaultdict(set)
  runs = parse_runs(args.runs)
  platforms = list(runs)
  notes = []
  modes = set()
  coverage = []
  standalone = []
  for label, roots in runs.items():
    records, arch = load_platform(catalog, label, roots)
    modes.update(record["quick"] for record in records.values())
    for (bench, case), record in records.items():
      measured[record["binary"]].add(case)
    found, missing = points(catalog, label, records, arch, standalone)
    coverage.append({"platform": label, "cases": len(records), "comparison_points": len(found)})
    rows += found
    for problem in missing:
      # A missing competitor build is never a filter artifact.
      (notes if args.partial and "needs one rscrypto row" in problem else problems).append(problem)
  problems += undeclared(catalog, measured)
  if len(modes) != 1:
    raise ComparisonError("cannot combine quick and regular benchmark runs")
  quick = modes == {True}
  result = ({"mode": "quick", "partial": args.partial, "coverage": coverage} if quick else
            {"tie": TIE, "partial": args.partial, "points": rows})
  if standalone:
    result["standalone"] = standalone
  if args.json:
    args.json.write_text(json.dumps(result, indent=2) + "\n")
  if quick:
    print("## Quick benchmark validation\n\nEvery selected row ran its output check. "
          "Quick estimates do not establish performance rankings.\n")
    print(table(["Platform", "Cases", "Comparison points"],
                [[row["platform"], str(row["cases"]), str(row["comparison_points"])] for row in coverage]))
  else:
    print(report(rows, platforms), end="")
  for row in standalone:
    print(f"\n{row['platform']}: `{row['case']}` was measured without a comparison. {row['reason']}")
  if notes:
    print(f"\n{len(notes)} comparison points are incomplete because the run's case filter omitted rows.")
  for problem in problems:
    print(f"error: {problem}", file=sys.stderr)
  return 1 if problems else 0


if __name__ == "__main__":
  try:
    raise SystemExit(main(sys.argv[1:]))
  except (ComparisonError, OSError, KeyError) as error:
    print(f"error: {error}", file=sys.stderr)
    raise SystemExit(2)
