#!/usr/bin/env -S uv run python
"""Regression tests for painted-stack allowances, budgets, and checks on synthetic reports."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import painted
from painted import DECLARED_EDGES, EDGES_KIND, WORKER_KIND, allowance, check_worker, judge, worker_budget

TARGET = "x86_64-unknown-linux-gnu"


def edge_report(depths: dict[str, int], identity: str = "glibc fixture") -> dict:
  edges = {name: {"max_below_entry_bytes": depth, "allowance_bytes": allowance(depth, 128)}
           for name, depth in depths.items()}
  return {"kind": EDGES_KIND, "target": TARGET, "edges": edges, "libc_identity": identity}


def worker_report(target: str, configuration: str) -> dict:
  return {"kind": WORKER_KIND, "target": target, "configuration": configuration, "failed": [],
          "max_cold_bytes": 54_224, "max_cold_scenario": "deepest"}


def test_allowance_rounds_then_adds_margin() -> None:
  # (round16(depth) + red zone) * 9 / 8, rounded up to 64 bytes.
  assert allowance(0, 0) == 0
  assert allowance(1, 128) == 192  # (16 + 128) * 1.125 = 162
  assert allowance(16, 0) == 64  # 18
  assert allowance(1000, 288) == 1472  # (1008 + 288) * 1.125 = 1458
  assert [worker_budget(depth) for depth in (0, 54_224, 9216)] == [1024, 61_440, 11_264]  # 10,368


def test_worker_fails_on_any_scenario_or_budget() -> None:
  with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / "budgets.json"
    path.write_text(json.dumps({"targets": {TARGET: {"native": 61_440}}}))
    report = {"target": TARGET, "configuration": "native", "failed": [], "max_cold_bytes": 54_224,
              "max_cold_scenario": "deepest"}
    assert check_worker(report, path) == []
    assert check_worker({**report, "failed": ["a/b"]}, path) == ["scenario failed: a/b"]
    assert check_worker({**report, "max_cold_bytes": 61_441}, path) == [
      "deepest scenario deepest used 61441 B; budget 61440 B"]
    assert check_worker({**report, "configuration": "portable"}, path) == [
      f"{path} has no portable budget for {TARGET}"]


def test_judge_fails_on_a_budget_or_an_absent_report_and_writes_the_measured_allowances() -> None:
  # The CI `stack` job judges the uploads of every leg with this, after all legs measured.
  depths = {name: 100 for name in DECLARED_EDGES}
  windows = "x86_64-pc-windows-msvc"
  expected = [TARGET, windows]
  complete = [edge_report(depths), *(worker_report(target, configuration)
                                     for target in expected for configuration in ("native", "portable"))]
  with tempfile.TemporaryDirectory() as temporary:
    budgets = Path(temporary) / "budgets.json"
    budgets.write_text(json.dumps({"targets": {target: {"native": 61_440, "portable": 61_440} for target in expected}}))
    assert judge(complete, expected, budgets) == []
    # A leg that failed before it uploaded, or a skipped RISC-V run that was required.
    riscv = "riscv64gc-unknown-linux-gnu"
    assert judge(complete[:-1], [*expected, riscv], budgets) == [
      f"{windows} worker portable: no report", f"{riscv} edges: no report", f"{riscv} worker native: no report",
      f"{riscv} worker portable: no report"]
    # The same run's edge measurements become the allowances that frames.py composes.
    paths = []
    for index, report in enumerate(complete):
      path = Path(temporary) / f"report-{index}.json"
      path.write_text(json.dumps(report))
      paths.append(str(path))
    allowances = Path(temporary) / "allowances.json"
    with patch.object(painted, "DEFAULT_WORKER_BUDGETS", budgets), contextlib.redirect_stdout(io.StringIO()):
      assert painted.main(["judge", *paths, "--expect", TARGET, "--expect", windows,
                           "--allowances", str(allowances)]) == 0
    written = json.loads(allowances.read_text())
    assert written == {TARGET: {"libc": "glibc fixture",
                                "edges": {name: allowance(100, 128) for name in DECLARED_EDGES}}}, written
    # The command fails the job on any problem.
    with patch.object(painted, "DEFAULT_WORKER_BUDGETS", budgets), contextlib.redirect_stdout(io.StringIO()) as output:
      assert painted.main(["judge", paths[0], "--expect", TARGET]) == 1
    assert f"stack check failure: {TARGET} worker native: no report" in output.getvalue(), output.getvalue()


def main() -> None:
  test_allowance_rounds_then_adds_margin()
  test_worker_fails_on_any_scenario_or_budget()
  test_judge_fails_on_a_budget_or_an_absent_report_and_writes_the_measured_allowances()
  print("painted-stack tests passed")


if __name__ == "__main__":
  main()
