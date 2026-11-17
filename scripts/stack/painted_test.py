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
from painted import (DECLARED_EDGES, EDGES_KIND, WORKER_KIND, allowance, check_edges, check_worker, judge,
                     worker_budget, write_allowances)

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


def test_edges_fail_only_above_their_allowance() -> None:
  depths = {name: 100 for name in DECLARED_EDGES}
  with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / "allowances.json"
    write_allowances([edge_report(depths)], path, "fixture evidence")
    written = json.loads(path.read_text())[TARGET]
    assert written["libc"] == "glibc fixture" and written["evidence"] == "fixture evidence", written
    assert set(written["edges"]) == set(DECLARED_EDGES), written
    assert check_edges(edge_report(depths), path) == []
    deeper = edge_report({**depths, "free": 200})
    assert check_edges(deeper, path) == [f"free: measured allowance {allowance(200, 128)} B exceeds the committed "
                                          f"{allowance(100, 128)} B"], check_edges(deeper, path)
    # Another libc is reported, not failed, while its use stays within the allowance.
    with contextlib.redirect_stdout(io.StringIO()) as output:
      assert check_edges(edge_report(depths, "glibc other"), path) == []
    assert "glibc other" in output.getvalue()
    missing = edge_report(depths)
    missing["target"] = "aarch64-unknown-linux-gnu"
    assert check_edges(missing, path) == [f"{path} has no allowances for aarch64-unknown-linux-gnu"]


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


def test_judge_fails_on_a_missing_or_exceeded_allowance_or_an_absent_report() -> None:
  # The CI `stack` job judges the uploads of every leg with this, after all legs measured.
  depths = {name: 100 for name in DECLARED_EDGES}
  windows = "x86_64-pc-windows-msvc"
  expected = [TARGET, windows]
  complete = [edge_report(depths), *(worker_report(target, configuration)
                                     for target in expected for configuration in ("native", "portable"))]
  with tempfile.TemporaryDirectory() as temporary:
    allowances, budgets = Path(temporary) / "allowances.json", Path(temporary) / "budgets.json"
    write_allowances([edge_report(depths)], allowances, "fixture evidence")
    budgets.write_text(json.dumps({"targets": {target: {"native": 61_440, "portable": 61_440} for target in expected}}))
    assert judge(complete, expected, allowances, budgets) == []
    # The first run after a new edge is declared: the leg measured it, and no allowance is committed.
    committed = json.loads(allowances.read_text())
    del committed[TARGET]["edges"]["madvise"]
    allowances.write_text(json.dumps(committed))
    assert judge(complete, expected, allowances, budgets) == [f"{TARGET} edges: madvise: no committed allowance"]
    write_allowances([edge_report(depths)], allowances, "fixture evidence")
    deeper = edge_report({**depths, "free": 200})
    assert judge([deeper, *complete[1:]], expected, allowances, budgets) == [
      f"{TARGET} edges: free: measured allowance {allowance(200, 128)} B exceeds the committed "
      f"{allowance(100, 128)} B"]
    # A leg that failed before it uploaded, or a skipped RISC-V run that was required.
    riscv = "riscv64gc-unknown-linux-gnu"
    assert judge(complete[:-1], [*expected, riscv], allowances, budgets) == [
      f"{windows} worker portable: no report", f"{riscv} edges: no report", f"{riscv} worker native: no report",
      f"{riscv} worker portable: no report"]
    # The command fails the job on any problem.
    report = Path(temporary) / "report.json"
    report.write_text(json.dumps(complete[0]))
    with patch.object(painted, "DEFAULT_ALLOWANCES", allowances), \
         patch.object(painted, "DEFAULT_WORKER_BUDGETS", budgets), contextlib.redirect_stdout(io.StringIO()) as output:
      assert painted.main(["judge", str(report), "--expect", TARGET]) == 1
    assert f"stack check failure: {TARGET} worker native: no report" in output.getvalue(), output.getvalue()


def main() -> None:
  test_allowance_rounds_then_adds_margin()
  test_edges_fail_only_above_their_allowance()
  test_worker_fails_on_any_scenario_or_budget()
  test_judge_fails_on_a_missing_or_exceeded_allowance_or_an_absent_report()
  print("painted-stack tests passed")


if __name__ == "__main__":
  main()
