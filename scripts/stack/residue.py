#!/usr/bin/env python3
"""Measure moved-copy residue of secret owners under QEMU.

Boots `tools/residue-harness` on the RV32 `virt` and Cortex-M3 `mps2-an385`
boards, with native and portable backends. For each scenario the harness
prints the dead stack and allocator arenas left after the operation returns,
followed by the scenario's secret byte strings ("needles"). This script finds
every 16-byte window of each needle in each region.

Expectations come from the harness: `none` fails on any needle byte found,
`stack` and `arena` are controls whose needle must be found in full in that
region, and `report` records the residue without judging it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import selectors
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "tools" / "residue-harness"
BINARY = "rscrypto-residue-harness"
WINDOW = 16


@dataclass(frozen=True)
class Board:
  target: str
  script: str
  command: tuple[str, ...]


BOARDS = {
  "riscv32-virt": Board(
    target="riscv32imac-unknown-none-elf",
    script="riscv32-virt.x",
    command=("qemu-system-riscv32", "-machine", "virt", "-bios", "none", "-m", "64M"),
  ),
  "thumb-mps2-an385": Board(
    target="thumbv6m-none-eabi",
    script="thumb-mps2-an385.x",
    command=("qemu-system-arm", "-machine", "mps2-an385", "-cpu", "cortex-m3"),
  ),
}
BACKENDS = ("native", "portable")
EXPECTATIONS = {"none", "report", "stack", "arena"}


@dataclass
class Scenario:
  name: str
  expect: str
  regions: dict[str, tuple[int, bytes]] = field(default_factory=dict)
  needles: dict[str, bytes] = field(default_factory=dict)


@dataclass
class Run:
  board: str
  backend: str
  declared: int
  scenarios: list[Scenario]
  complete: bool
  panic: str | None


HEADER = re.compile(r"^residue 1 board=(\S+) backend=(\S+) scenarios=(\d+)$")


def parse(text: str) -> Run:
  lines = iter(text.splitlines())
  run: Run | None = None
  current: Scenario | None = None
  for line in lines:
    line = line.strip()
    if match := HEADER.match(line):
      run = Run(match.group(1), match.group(2), int(match.group(3)), [], False, None)
    elif run is None:
      continue  # QEMU or firmware noise before the harness starts
    elif line.startswith("panic "):
      run.panic = line.removeprefix("panic ")
    elif line == "done":
      run.complete = True
    elif line.startswith("scenario "):
      name, expect = line.removeprefix("scenario ").split(" expect=")
      if expect not in EXPECTATIONS:
        raise ValueError(f"unknown expectation {expect!r} for {name}")
      current = Scenario(name, expect)
      run.scenarios.append(current)
    elif line.startswith("region ") and current is not None:
      _, name, base, length = line.split()
      data = bytearray()
      for row in lines:
        row = row.strip()
        if row == "endregion":
          break
        data += bytes.fromhex(row)
      if len(data) != int(length, 16):
        raise ValueError(f"{current.name} {name}: expected {int(length, 16)} bytes, read {len(data)}")
      current.regions[name] = (int(base, 16), bytes(data))
    elif line.startswith("needle ") and current is not None:
      _, name, length = line.split()
      data = bytearray()
      while len(data) < int(length):
        data += bytes.fromhex(next(lines).strip())
      if len(data) != int(length):
        raise ValueError(f"{current.name} needle {name}: expected {length} bytes, read {len(data)}")
      current.needles[name] = bytes(data)
    elif line == "end":
      current = None
  if run is None:
    raise ValueError("no harness header in the output")
  return run


def windows(needle: bytes) -> list[int]:
  """Window offsets that cover every byte of `needle`."""
  if len(needle) < WINDOW:
    raise ValueError(f"needles must be at least {WINDOW} bytes")
  offsets = list(range(0, len(needle) - WINDOW + 1, WINDOW))
  if offsets[-1] != len(needle) - WINDOW:
    offsets.append(len(needle) - WINDOW)
  return offsets


def found_bytes(needle: bytes, region: bytes) -> int:
  """Count needle bytes covered by a window that occurs anywhere in `region`."""
  covered = bytearray(len(needle))
  for offset in windows(needle):
    if needle[offset : offset + WINDOW] in region:
      covered[offset : offset + WINDOW] = b"\x01" * WINDOW
  return sum(covered)


def judge(run: Run) -> dict[str, object]:
  problems = []
  if run.panic is not None:
    problems.append(f"harness panicked: {run.panic}")
  if not run.complete:
    problems.append("harness did not finish")
  if len(run.scenarios) != run.declared:
    problems.append(f"expected {run.declared} scenarios, read {len(run.scenarios)}")
  rows = []
  for scenario in run.scenarios:
    if not scenario.needles:
      problems.append(f"{scenario.name}: no needles")
    if set(scenario.regions) != {"stack", "heap", "arena"}:
      problems.append(f"{scenario.name}: missing regions")
    results = {}
    for needle_name, needle in scenario.needles.items():
      results[needle_name] = {
        "bytes": len(needle),
        "found": {region: found_bytes(needle, data) for region, (_, data) in scenario.regions.items()},
        "copies": sum(data.count(needle) for _, data in scenario.regions.values()),
      }
    for needle_name, result in results.items():
      total = sum(result["found"].values())
      if scenario.expect == "none" and total:
        problems.append(f"{scenario.name}: {total} bytes of {needle_name} remain")
      if scenario.expect in {"stack", "arena"} and result["found"].get(scenario.expect) != result["bytes"]:
        problems.append(f"{scenario.name}: control {needle_name} not found in full in {scenario.expect}")
    stack = scenario.regions.get("stack", (0, b""))[1]
    rows.append({"scenario": scenario.name, "expect": scenario.expect, "stack_used": len(stack), "needles": results})
  return {"board": run.board, "backend": run.backend, "scenarios": rows, "problems": problems, "passed": not problems}


def build(board: Board, backend: str) -> Path:
  target_dir = ROOT / "target" / "residue-harness" / backend
  environment = dict(os.environ)
  environment.pop("RUSTFLAGS", None)
  environment["CARGO_ENCODED_RUSTFLAGS"] = f"-Clink-arg=-T{CRATE / 'link' / board.script}"
  command = [
    "cargo",
    "build",
    "--locked",
    "--release",
    "--manifest-path",
    str(CRATE / "Cargo.toml"),
    "--target",
    board.target,
    "--target-dir",
    str(target_dir),
  ]
  if backend == "portable":
    command += ["--features", "portable-only"]
  subprocess.run(command, check=True, env=environment, cwd=ROOT)
  return target_dir / board.target / "release" / BINARY


def boot(board: Board, binary: Path, timeout: float) -> str:
  """Run the harness until it reports `done` or a panic, then stop QEMU."""
  command = [*board.command, "-display", "none", "-monitor", "none", "-serial", "stdio", "-kernel", str(binary)]
  process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  output = bytearray()
  selector = selectors.DefaultSelector()
  assert process.stdout is not None
  selector.register(process.stdout, selectors.EVENT_READ)
  deadline = time.monotonic() + timeout
  try:
    while time.monotonic() < deadline:
      if not selector.select(timeout=1.0):
        if process.poll() is not None:
          break
        continue
      chunk = os.read(process.stdout.fileno(), 1 << 16)
      if not chunk:
        break
      output += chunk
      tail = output[-4096:]
      finished = b"\ndone\n" in tail or b"\ndone\r\n" in tail
      if finished or (b"\npanic " in tail and tail.endswith(b"\n")):
        break
  finally:
    process.kill()
    process.wait()
  return output.decode(errors="replace")


def markdown(report: dict[str, object]) -> str:
  lines = [f"## {report['board']} {report['backend']}: {'pass' if report['passed'] else 'FAIL'}", ""]
  lines.extend(f"- problem: {problem}" for problem in report["problems"])
  lines += [
    "",
    "| Scenario | Expect | Stack used | Needle | Bytes | Stack | Heap | Arena | Whole copies |",
    "|---|---|---:|---|---:|---:|---:|---:|---:|",
  ]
  for row in report["scenarios"]:
    for name, result in row["needles"].items():
      found = result["found"]
      lines.append(
        f"| {row['scenario']} | {row['expect']} | {row['stack_used']} | {name} | {result['bytes']} "
        f"| {found.get('stack', 0)} | {found.get('heap', 0)} | {found.get('arena', 0)} | {result['copies']} |"
      )
  return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--board", action="append", choices=sorted(BOARDS), help="repeatable (default: all)")
  parser.add_argument("--backend", action="append", choices=BACKENDS, help="repeatable (default: both)")
  parser.add_argument("--timeout", type=float, default=900.0, help="seconds per boot")
  parser.add_argument("--output", type=Path, help="directory for raw UART logs and report.json")
  args = parser.parse_args(argv)

  reports = []
  for board_name in args.board or sorted(BOARDS):
    for backend in args.backend or BACKENDS:
      board = BOARDS[board_name]
      binary = build(board, backend)
      text = boot(board, binary, args.timeout)
      if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"{board_name}-{backend}.log").write_text(text)
      run = parse(text)
      if (run.board, run.backend) != (board_name, backend):
        raise SystemExit(f"booted {run.board}/{run.backend}, expected {board_name}/{backend}")
      report = judge(run)
      reports.append(report)
      print(markdown(report))
  if args.output is not None:
    (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
  return 0 if all(report["passed"] for report in reports) else 1


if __name__ == "__main__":
  sys.exit(main())
