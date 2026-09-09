"""Benchmark, profile, inspect, and export resolved production workloads."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile

# Embedded Windows Python omits the script directory from its import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_catalog import case_class, load_catalog, resolve_selector
from execution import build, build_command, build_environment, build_identity, digest, discover, match_cases, exit_code, identity, write_json, source_evidence
from measure import measure, verify
import settings

ROOT = Path(__file__).resolve().parents[2]


def boolean(value):
  if isinstance(value, bool):
    return value
  if value.lower() not in {"true", "false"}:
    raise argparse.ArgumentTypeError("expected true or false")
  return value.lower() == "true"


def parse(arguments: list[str]):
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="mode", required=True)
  bench = commands.add_parser("bench")
  bench.add_argument("selectors", nargs="*")
  bench.add_argument("--bench", action="append", default=[])
  bench.add_argument("--filter", action="append", default=[])
  bench.add_argument("--list", action="store_true")
  bench.add_argument("--diag", nargs="?", const=True, type=boolean, default=False)
  bench.add_argument("--output-dir", default="benchmark_results")
  bench.add_argument("--baseline", type=Path)
  for name in ("warmup-ms", "measure-ms", "sample-size"):
    bench.add_argument("--" + name, type=int)
  profile = commands.add_parser("profile")
  profile.add_argument("target")
  profile.add_argument("case", nargs="?")
  profile.add_argument("seconds", nargs="?", type=float, default=10)
  profile.add_argument("--list", action="store_true")
  profile.add_argument("--diag", action="store_true")
  for name in ("codegen", "llvm-lines"):
    command = commands.add_parser(name)
    command.add_argument("target")
    command.add_argument("--diag", action="store_true")
  export = commands.add_parser("export")
  export.add_argument("run", type=Path)
  normalized = []
  value_follows = False
  for index, token in enumerate(arguments):
    if not value_follows and arguments[0] == "bench" and token == "--diag":
      value = arguments[index + 1].lower() if index + 1 < len(arguments) else ""
      normalized.append(token if value in {"true", "false"} else "--diag=true")
    elif not value_follows and arguments[0] == "bench" and "=" in token and re.fullmatch(r"[a-z_]+", token.split("=", 1)[0]):
      key, value = token.split("=", 1)
      normalized.append("--" + key.replace("_", "-") + "=" + value)
    else:
      normalized.append(token)
    value_follows = token in {"--bench", "--filter", "--output-dir", "--baseline", "--warmup-ms", "--measure-ms", "--sample-size"}
  if arguments[:1] in (["codegen"], ["llvm-lines"]):
    boundary = normalized.index("--") if "--" in normalized else len(normalized)
    args = parser.parse_args(normalized[:boundary])
    args.args = normalized[boundary + 1:]
  else:
    args = parser.parse_args(normalized)
  if args.mode == "bench":
    # Every accepted environment option has the same meaning as its CLI form.
    known = {"BENCH_BENCH", "BENCH_FILTER", "BENCH_DIAG", "BENCH_LIST", "BENCH_OUTPUT_DIR", "BENCH_BASELINE",
             "BENCH_WARMUP_MS", "BENCH_MEASURE_MS", "BENCH_SAMPLE_SIZE"}
    unknown = set(key for key in os.environ if key.startswith("BENCH_")) - known
    if unknown:
      parser.error(f"unknown benchmark environment options: {', '.join(sorted(unknown))}")
    for key in known:
      if key not in os.environ:
        continue
      name = key.removeprefix("BENCH_").lower()
      value = os.environ[key]
      explicit = any(token.split("=", 1)[0] == "--" + name.replace("_", "-") for token in normalized)
      if name in {"bench", "filter"}:
        getattr(args, name).insert(0, value)
      elif not explicit:
        setattr(args, name, boolean(value) if name in {"diag", "list"} else
                int(value) if name in {"warmup_ms", "measure_ms", "sample_size"} else
                Path(value) if name == "baseline" else value)
    if any(not pattern for pattern in args.filter):
      parser.error("filter must not be empty")
  return args


def target(catalog, name: str, diag: bool) -> dict:
  entry = catalog["benches"].get(name)
  if entry is None or entry["kind"] != "criterion":
    raise ValueError(f"unknown Criterion target: {name}")
  features = sorted(set(entry["features"]) | ({"diag"} if diag else set()))
  return {"binary": entry["binary"], "features": features}


def requests(args, catalog) -> list[dict]:
  names = list(dict.fromkeys(name.strip() for value in args.bench for name in value.split(",")))
  algorithms = []
  for selector in args.selectors:
    selected = resolve_selector(catalog, selector)
    if selected is None:
      raise ValueError(f"unknown selector: {selector}")
    algorithms.extend(selected)
  rows = []
  if algorithms:
    for algorithm in dict.fromkeys(algorithms):
      entry = catalog["algorithms"][algorithm]
      if not names or entry["bench"] in names:
        rows.extend((entry["bench"], entry["filter"], pattern) for pattern in (args.filter or [entry["filter"]]))
  else:
    names = names or [name for name, entry in catalog["benches"].items() if entry["kind"] == "criterion" and entry["required"]]
    rows = [(name, "", pattern) for name in names for pattern in (args.filter or [""])]
  for name in names:
    target(catalog, name, args.diag)
  if not rows:
    raise ValueError("selection produced no benchmark configurations")
  return [target(catalog, name, args.diag) | {"scope": scope, "pattern": pattern}
          for name, scope, pattern in dict.fromkeys(rows)]


def resolve(rows, effective, log, env) -> list[dict]:
  execution = build_identity()
  builds = {}
  matches = {}
  matched = dict.fromkeys((row["pattern"] for row in rows), False)
  for row in rows:
    command = build_command(row["binary"], row["features"])
    key = tuple(command)
    if key not in builds:
      artifact = build(command, log, env)
      cases = discover(artifact, "", log, env)
      patterns = [pattern for request in rows if build_command(request["binary"], request["features"]) == command
                  for pattern in (request["scope"], request["pattern"])]
      matches[key] = match_cases(artifact, cases, patterns, log, env)
      config = identity([command, execution, effective])
      builds[key] = {"id": config, "binary": row["binary"], "artifact": artifact,
                     "compatibility": execution, "settings": effective, "cases": [], "baselines": [],
                     "home": "criterion/" + row["binary"] + "-" + config}
    entry = builds[key]

    scope = set(matches[key][row["scope"]])
    cases = [case for case in matches[key][row["pattern"]] if case in scope]
    matched[row["pattern"]] |= bool(cases)
    entry["cases"] = list(dict.fromkeys([*entry["cases"], *cases]))
  if not all(matched.values()):
    raise ValueError(f"no cases matched: {[pattern for pattern, found in matched.items() if not found]}")
  return [entry for entry in builds.values() if entry["cases"]]


def export_run(root: Path) -> None:
  root = root.resolve()
  if not {"state=complete", "state=failed"} & set((root / "status.txt").read_text().splitlines()):
    raise ValueError("export requires a completed or failed run")
  transfer = root.parent.parent / ".transfers"
  transfer.mkdir(exist_ok=True)
  archive = transfer / (root.name + ".tar")
  if archive.exists() or archive.with_suffix(".tar.sha256").exists():
    raise ValueError(f"export already exists: {archive}")
  with tempfile.TemporaryDirectory(prefix=".export-", dir=transfer) as directory:
    partial = Path(directory) / "run.tar"
    with tarfile.open(partial, mode="w") as bundle:
      bundle.add(root, arcname="criterion/" + root.name)
    checksum = Path(directory) / "run.sha256"
    checksum.write_text(f"{digest(partial)}  {archive.name}\n")
    # Publish complete files without replacing a concurrent export.
    os.link(partial, archive)
    os.link(checksum, archive.with_suffix(".tar.sha256"))
  print(f"Archive: {archive}")


def bench(args, catalog) -> None:
  rows = requests(args, catalog)
  effective = settings.load({name: value for name in ("warmup_ms", "measure_ms", "sample_size")
                             if (value := getattr(args, name)) is not None})
  effective.pop("max_run_seconds")
  if args.list:
    with tempfile.TemporaryDirectory(prefix="rscrypto-list-") as directory:
      root = Path(directory)
      plan = resolve(rows, effective, root / "output.txt", dict(os.environ) | {"CRITERION_HOME": directory})
      for entry in plan:
        for case in entry["cases"]:
          print(f"[{case_class(catalog, entry['binary'], case)}] {entry['binary']} {case}")
    return
  baseline = args.baseline.resolve() if args.baseline else None
  if baseline and "state=complete" not in (baseline / "status.txt").read_text().splitlines():
    raise ValueError("baseline must be a completed run")
  parent = Path(args.output_dir).resolve() / "criterion"
  run_id = os.environ.get("RSCRYPTO_BENCH_RUN_ID")
  if run_id and (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) or ".." in run_id):
    raise ValueError("invalid run ID")
  parent.mkdir(parents=True, exist_ok=True)
  if run_id:
    if any((parent.parent / ".transfers").glob(run_id + ".tar*")):
      raise ValueError("run ID already exported")
    root = parent / run_id
    root.mkdir()
  else:
    root = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-"), dir=parent))
  print(f"Run directory: {root}", flush=True)
  (root / "status.txt").write_text("state=running\n")
  status = 1
  try:
    write_json(root / "requests.json", {"configurations": rows, "budget_seconds": settings.load()["max_run_seconds"]})
    source_evidence(root)
    env = dict(os.environ) | {"CRITERION_HOME": str(root / "discovery")}
    plan = resolve(rows, effective, root / "output.txt", env)
    shutil.rmtree(root / "discovery", ignore_errors=True)
    write_json(root / "plan.json", plan)
    measure(root, plan, baseline)
    verify(root)
    status = 0
  except BaseException as error:
    status = exit_code(error)
    raise
  finally:
    (root / "status.txt").write_text(f"state={'complete' if status == 0 else 'failed'}\nexit_code={status}\n")
    print(f"Results: {root}", flush=True)


def main() -> int:
  os.chdir(ROOT)
  args = parse(sys.argv[1:])
  if args.mode == "export":
    export_run(args.run)
    return 0
  os.environ.pop("RSCRYPTO_BENCH_CASES", None)
  build_environment()
  catalog = load_catalog()
  if args.mode == "bench":
    bench(args, catalog)
  elif args.mode == "profile":
    from profile import profile
    profile(args, target(catalog, args.target, args.diag))
  else:
    entry = target(catalog, args.target, args.diag)
    command = ["cargo", "asm" if args.mode == "codegen" else "llvm-lines", "--locked", "--lib", "--profile", "bench",
               "--no-default-features", "--features", ",".join(entry["features"])]
    extra = args.args[1:] if args.args[:1] == ["--"] else args.args
    subprocess.run([*command, *extra], check=True)
  return 0


if __name__ == "__main__":
  def interrupted(signum, _frame):
    raise SystemExit(128 + signum)
  signal.signal(signal.SIGTERM, interrupted)
  try:
    raise SystemExit(main())
  except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
    print(f"error: {error}", file=sys.stderr)
    raise SystemExit(exit_code(error))
