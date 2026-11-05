#!/usr/bin/env -S uv run python
"""Measure native stack use on painted, guarded private stacks.

`edges` measures the declared libc edges of the whole-call stack review on the
running Linux GNU host and derives their allowances. `worker` measures the
provider TLS worker's high-water mark for every scenario of
`tools/stack-paint`, one fresh process each. Both build the tool with the
shipping release profile unless `--binary-dir` names prebuilt binaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "tools" / "stack-paint"
EDGES_BINARY = "rscrypto-stack-edges"
WORKER_BINARY = "rscrypto-stack-worker"
# The libc imports at which `frames.py --whole-calls` stops its owned bounds.
DECLARED_EDGES = ("calloc", "malloc", "free", "memcpy", "memmove", "memset", "memcmp", "bcmp", "getauxval")
STACK_CAP = 8 * 1024 * 1024
DEFAULT_ALLOWANCES = ROOT / ".config" / "stack-allowances.json"
DEFAULT_WORKER_BUDGETS = ROOT / ".config" / "stack-worker-budgets.json"


def host_target() -> str:
  return subprocess.run(["rustc", "--print", "host-tuple"], check=True, capture_output=True, text=True).stdout.strip()


def executable(name: str) -> str:
  return f"{name}.exe" if platform.system() == "Windows" else name


def build(target: str, portable: bool) -> Path:
  directory = ROOT / "target" / "stack-paint" / ("portable" if portable else "native")
  command = ["cargo", "build", "--locked", "--release", "--manifest-path", str(CRATE / "Cargo.toml"),
             "--target", target, "--target-dir", str(directory)]
  if portable:
    command += ["--features", "portable-only"]
  environment = dict(os.environ)
  # The shipping configuration: no target CPU or other code generation flags.
  environment.pop("RUSTFLAGS", None)
  environment.pop("CARGO_ENCODED_RUSTFLAGS", None)
  subprocess.run(command, check=True, cwd=ROOT, env=environment)
  return directory / target / "release"


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: str) -> str:
  try:
    return Path(path).read_text(errors="replace")
  except OSError:
    return ""


def command_output(command: list[str], environment: dict[str, str] | None = None) -> str:
  try:
    return subprocess.run(command, capture_output=True, text=True, check=False, env=environment).stdout.strip()
  except OSError:
    return ""


def cpu_identity() -> str:
  """The processor as /proc/cpuinfo names it; IFUNC selection depends on it."""
  fields = {}
  for line in read("/proc/cpuinfo").splitlines():
    key, _, value = line.partition(":")
    key = key.strip()
    if key and key not in fields and value.strip():
      fields[key] = value.strip()
  for keys in (("model name",), ("cpu", "revision"), ("CPU implementer", "CPU part", "CPU variant"),
               ("uarch", "isa"), ("machine",), ("vendor_id",)):
    if all(key in fields for key in keys):
      return ", ".join(f"{key} {fields[key]}" for key in keys)
  if platform.system() == "Darwin":
    return command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.machine()
  return platform.processor() or platform.machine()


def host_identity() -> dict[str, object]:
  release = {}
  for line in read("/etc/os-release").splitlines():
    key, _, value = line.partition("=")
    release[key] = value.strip().strip('"')
  auxv = command_output(["/bin/true"], {**os.environ, "LD_SHOW_AUXV": "1"}) if Path("/bin/true").exists() else ""
  hwcaps = [line for line in auxv.splitlines() if line.startswith(("AT_HWCAP", "AT_PLATFORM", "AT_BASE_PLATFORM"))]
  return {
    "os": release.get("PRETTY_NAME", platform.platform()),
    "kernel": platform.release(),
    "machine": platform.machine(),
    "cpu": cpu_identity(),
    "hwcaps": hwcaps,
    "libc6_package": command_output(["dpkg-query", "-W", "-f", "${Version}", "libc6"]) or None,
  }


ALLOWANCE_RULE = ("Measured depth below the entry stack pointer, rounded up to 16 bytes, plus the ABI red zone below "
                  "the deepest frame, then 12.5% more, rounded up to 64 bytes; paths the scenarios did not reach "
                  "are not covered.")


def allowance(measured: int, red_zone: int) -> int:
  """The committed allowance for a measured depth.

  The measured depth, rounded up to 16 bytes, plus the ABI red zone below the
  deepest frame, then 12.5% more, rounded up to 64 bytes. A path the scenarios
  did not reach is not covered.
  """
  rounded = -(-measured // 16) * 16 + red_zone
  return -(-rounded * 9 // (8 * 64)) * 64


def worker_budget(measured: int) -> int:
  """The committed worker budget: 12.5% above the deepest scenario, rounded up to 1 KiB."""
  return max(1024, -(-measured * 9 // (8 * 1024)) * 1024)


def libc_identity(report: dict[str, object]) -> str:
  host = report["host"]
  libc = report["libc"]
  package = f" ({host['libc6_package']})" if host.get("libc6_package") else ""
  return (f"glibc {report['measurement']['glibc_version']}{package}, {Path(str(libc['path'])).name} "
          f"sha256 {str(libc['sha256'])[:16]}; {host['os']}; Linux {host['kernel']}; {host['cpu']}")


def measure_edges(binaries: Path, target: str) -> dict[str, object]:
  binary = binaries / executable(EDGES_BINARY)
  result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
  measurement = json.loads(result.stdout)
  objects = {edge["implementation"]["object"] for edge in measurement["edges"].values() if edge["implementation"]}
  libc_path = next((path for path in sorted(objects) if "libc.so" in path), None)
  if libc_path is None:
    raise SystemExit(f"no edge resolved into libc: {sorted(objects)}")
  resolved = Path(libc_path).resolve()
  red_zone = int(measurement["red_zone_bytes"])
  edges = {name: {**entry, "allowance_bytes": allowance(int(entry["max_below_entry_bytes"]), red_zone)}
           for name, entry in measurement["edges"].items()}
  if set(edges) != set(DECLARED_EDGES):
    raise SystemExit(f"measured edges {sorted(edges)} differ from the declared edges")
  report = {
    "kind": "rscrypto.stack.libc-edges",
    "target": target,
    "binary": str(binary),
    "binary_sha256": sha256(binary),
    "libc": {"path": str(resolved), "sha256": sha256(resolved)},
    "host": host_identity(),
    "measurement": measurement,
    "edges": edges,
    "rule": ALLOWANCE_RULE,
    "generated_at": datetime.now(timezone.utc).isoformat(),
  }
  report["libc_identity"] = libc_identity(report)
  return report


def check_edges(report: dict[str, object], path: Path) -> list[str]:
  committed = json.loads(path.read_text()).get(str(report["target"]))
  if committed is None:
    return [f"{path} has no allowances for {report['target']}"]
  problems = []
  for name, entry in report["edges"].items():
    limit = committed["edges"].get(name)
    if limit is None:
      problems.append(f"{name}: no committed allowance")
    elif entry["allowance_bytes"] > limit:
      problems.append(f"{name}: measured allowance {entry['allowance_bytes']} B exceeds the committed {limit} B")
  if committed["libc"] != report["libc_identity"]:
    print(f"note: committed allowances name {committed['libc']!r}; this host is {report['libc_identity']!r}",
          flush=True)
  return problems


def write_allowances(reports: list[dict[str, object]], path: Path, evidence: str) -> None:
  document = json.loads(path.read_text()) if path.exists() else {}
  for report in reports:
    document[str(report["target"])] = {
      "libc": report["libc_identity"],
      "edges": {name: report["edges"][name]["allowance_bytes"] for name in DECLARED_EDGES},
      "evidence": evidence,
    }
  path.write_text(json.dumps(dict(sorted(document.items())), indent=2) + "\n")


def run_scenario(binary: Path, scenario: str) -> dict[str, object]:
  result = subprocess.run([str(binary), scenario], capture_output=True, text=True, check=False)
  try:
    row = json.loads(result.stdout)
  except json.JSONDecodeError:
    row = {"scenario": scenario, "passed": False}
  row["exit_code"] = result.returncode
  if result.returncode:
    row["passed"] = False
    row["stderr"] = result.stderr[-4096:]
  return row


def measure_worker(binaries: Path, target: str, portable: bool, jobs: int) -> dict[str, object]:
  binary = binaries / executable(WORKER_BINARY)
  scenarios = subprocess.run([str(binary), "--list"], check=True, capture_output=True, text=True).stdout.split()
  with ThreadPoolExecutor(max_workers=jobs) as pool:
    rows = list(pool.map(lambda scenario: run_scenario(binary, scenario), scenarios))
  measured = [row for row in rows if isinstance(row.get("cold_bytes"), int)]
  deepest = max(measured, key=lambda row: (row["cold_bytes"], row["scenario"]), default=None)
  warm = [row for row in measured if isinstance(row.get("warm_bytes"), int)]
  deepest_warm = max(warm, key=lambda row: (row["warm_bytes"], row["scenario"]), default=None)
  return {
    "kind": "rscrypto.stack.tls-worker",
    "target": target,
    "configuration": "portable" if portable else "native",
    "binary": str(binary),
    "binary_sha256": sha256(binary),
    "host": host_identity(),
    "scenarios": len(rows),
    "failed": [row["scenario"] for row in rows if not row.get("passed")],
    "max_cold_bytes": deepest["cold_bytes"] if deepest else None,
    "max_cold_scenario": deepest["scenario"] if deepest else None,
    "max_warm_bytes": deepest_warm["warm_bytes"] if deepest_warm else None,
    "max_warm_scenario": deepest_warm["scenario"] if deepest_warm else None,
    "rows": rows,
    "generated_at": datetime.now(timezone.utc).isoformat(),
  }


def check_worker(report: dict[str, object], path: Path) -> list[str]:
  problems = [f"scenario failed: {name}" for name in report["failed"]]
  budgets = json.loads(path.read_text())["targets"]
  budget = budgets.get(str(report["target"]), {}).get(str(report["configuration"]))
  deepest = report["max_cold_bytes"]
  if budget is None:
    problems.append(f"{path} has no {report['configuration']} budget for {report['target']}")
  elif deepest is None or deepest > budget:
    problems.append(f"deepest scenario {report['max_cold_scenario']} used {deepest} B; budget {budget} B")
  elif budget > STACK_CAP:
    problems.append(f"budget {budget} B exceeds the 8 MiB worker cap")
  return problems


def write_worker_budgets(reports: list[dict[str, object]], path: Path) -> None:
  document = json.loads(path.read_text()) if path.exists() else {"targets": {}}
  document["rule"] = ("Painted-stack high-water of the deepest `tools/stack-paint` TLS worker scenario, cold, "
                      "12.5% above the measurement and rounded up to 1 KiB. Regenerate with "
                      "`just stack-worker --write-budgets` on each target after a reviewed change.")
  for report in reports:
    if report["failed"] or report["max_cold_bytes"] is None:
      raise SystemExit(f"{report['target']} {report['configuration']}: failed scenarios; no budget written")
    targets = document.setdefault("targets", {})
    targets.setdefault(str(report["target"]), {})[str(report["configuration"])] = worker_budget(
      int(report["max_cold_bytes"]))
  document["targets"] = {target: dict(sorted(value.items())) for target, value in sorted(document["targets"].items())}
  path.write_text(json.dumps({"rule": document["rule"], "targets": document["targets"]}, indent=2) + "\n")


def save(output: Path | None, name: str, report: dict[str, object], echo: bool) -> None:
  if echo:
    print(f"--- report {name}\n{json.dumps(report)}\n--- end {name}", flush=True)
  if output is None:
    return
  output.mkdir(parents=True, exist_ok=True)
  (output / name).write_text(json.dumps(report, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("measurement", choices=("edges", "worker", "record"),
                      help="record: write committed allowances or worker budgets from saved reports")
  parser.add_argument("reports", nargs="*", type=Path, help="record: saved JSON reports")
  parser.add_argument("--target", help="target tuple of the binaries (default: the host)")
  parser.add_argument("--binary-dir", type=Path, help="directory of prebuilt binaries; skips the build")
  parser.add_argument("--portable", action="store_true", help="worker: build or label the portable-only configuration")
  parser.add_argument("--all-configurations", action="store_true", help="worker: measure native and portable builds")
  parser.add_argument("--output", type=Path, help="directory for the JSON reports")
  parser.add_argument("--check", nargs="?", const=None, default=argparse.SUPPRESS, type=Path,
                      help="compare against the committed allowances or worker budgets (default file when omitted)")
  parser.add_argument("--write", type=Path, help="edges: merge this target's allowances into FILE")
  parser.add_argument("--evidence", default="", help="edges: evidence note recorded with --write")
  parser.add_argument("--write-budgets", type=Path, nargs="?", const=DEFAULT_WORKER_BUDGETS, default=None,
                      help="worker: merge this target's budgets into FILE")
  parser.add_argument("--print-reports", action="store_true", help="also print each full JSON report")
  parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="worker: concurrent scenario processes")
  args = parser.parse_args(argv)
  target = args.target or host_target()
  check = getattr(args, "check", False)
  if args.binary_dir is not None and args.all_configurations:
    parser.error("--binary-dir names one configuration; omit --all-configurations")

  problems: list[str] = []
  if args.measurement == "record":
    reports = [json.loads(path.read_text()) for path in args.reports]
    edges = [report for report in reports if report.get("kind") == "rscrypto.stack.libc-edges"]
    workers = [report for report in reports if report.get("kind") == "rscrypto.stack.tls-worker"]
    if len(edges) + len(workers) != len(reports) or not reports:
      parser.error("record takes saved libc-edge or TLS-worker reports")
    if edges:
      write_allowances(edges, args.write or DEFAULT_ALLOWANCES, args.evidence)
    if workers:
      write_worker_budgets(workers, args.write_budgets or DEFAULT_WORKER_BUDGETS)
    return 0
  if args.measurement == "edges":
    if "linux-gnu" not in target:
      parser.error("declared libc edges are measured on Linux GNU targets only")
    binaries = args.binary_dir or build(target, portable=False)
    report = measure_edges(binaries, target)
    save(args.output, f"stack-allowances-{target}.json", report, args.print_reports)
    for name in DECLARED_EDGES:
      entry = report["edges"][name]
      print(f"{target} {name}: {entry['max_below_entry_bytes']} B below entry ({entry['deepest_case']}, "
            f"{entry['calls']} calls); allowance {entry['allowance_bytes']} B", flush=True)
    print(f"libc: {report['libc_identity']}", flush=True)
    if check is not False:
      problems = check_edges(report, check or DEFAULT_ALLOWANCES)
    if args.write is not None:
      write_allowances([report], args.write, args.evidence)
  else:
    configurations = [False, True] if args.all_configurations else [args.portable]
    reports = []
    for portable in configurations:
      binaries = args.binary_dir or build(target, portable)
      report = measure_worker(binaries, target, portable, max(1, args.jobs))
      save(args.output, f"stack-worker-{target}-{report['configuration']}.json", report, args.print_reports)
      print(f"{target} {report['configuration']}: {report['scenarios']} scenarios, {len(report['failed'])} failed; "
            f"deepest cold {report['max_cold_bytes']} B ({report['max_cold_scenario']}), "
            f"deepest warm {report['max_warm_bytes']} B ({report['max_warm_scenario']})", flush=True)
      for name in report["failed"]:
        print(f"failed: {name}", flush=True)
      if check is not False:
        problems += check_worker(report, check or DEFAULT_WORKER_BUDGETS)
      reports.append(report)
    if args.write_budgets is not None:
      write_worker_budgets(reports, args.write_budgets)
  for problem in problems:
    print(f"stack check failure: {problem}", flush=True)
  if args.measurement == "worker" and any(report["failed"] for report in reports):
    return 1
  return 1 if problems else 0


if __name__ == "__main__":
  sys.exit(main())
