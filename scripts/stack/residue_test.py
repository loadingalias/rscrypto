#!/usr/bin/env -S uv run python
"""Regression tests for residue log parsing and judgement, on synthetic logs."""

from __future__ import annotations

from residue import found_bytes, judge, parse, windows
from mldsa_residue import judge as native_judge, needles as native_needles, windows as native_windows

NEEDLE = bytes(range(0x40, 0x40 + 40))
OTHER = bytes(range(0x90, 0x90 + 32))


def region(name: str, data: bytes) -> list[str]:
  rows = [data[offset : offset + 32].hex() for offset in range(0, len(data), 32)]
  return [f"region {name} 0x80000000 {len(data):#x}", *rows, "endregion"]


def scenario(name: str, expect: str, stack: bytes, arena: bytes = b"", needle: bytes = NEEDLE) -> list[str]:
  return [
    f"scenario {name} expect={expect}",
    *region("stack", stack),
    *region("heap", b""),
    *region("arena", arena),
    f"needle secret {len(needle)}",
    needle[:32].hex(),
    needle[32:].hex(),
    "end",
  ]


def log(*scenarios: list[str], declared: int | None = None, done: bool = True) -> str:
  count = len(scenarios) if declared is None else declared
  lines = ["QEMU noise", f"residue 1 board=riscv32-virt backend=native scenarios={count}"]
  for item in scenarios:
    lines += item
  if done:
    lines.append("done")
  return "\n".join(lines) + "\n"


def test_windows_cover_every_byte() -> None:
  assert windows(bytes(32)) == [0, 16]
  assert windows(bytes(40)) == [0, 16, 24]
  try:
    windows(bytes(15))
  except ValueError:
    pass
  else:
    raise AssertionError("short needles must be rejected")


def test_partial_and_scattered_residue() -> None:
  # The first and last windows remain, in separate places; the middle is gone.
  region_bytes = NEEDLE[24:] + b"\xa5" * 8 + NEEDLE[:16]
  assert found_bytes(NEEDLE, region_bytes) == 32
  assert found_bytes(NEEDLE, NEEDLE[:15] + NEEDLE[16:]) == 24
  assert found_bytes(NEEDLE, b"\xa5" * 64) == 0


def test_expectations() -> None:
  clean = judge(parse(log(scenario("in", "none", b"\xa5" * 64), scenario("ctl", "stack", NEEDLE + bytes(24)))))
  assert clean["passed"], clean
  assert clean["scenarios"][1]["needles"]["secret"]["copies"] == 1

  leaked = judge(parse(log(scenario("in", "none", b"\xa5" * 8 + NEEDLE[:16] + bytes(8)))))
  assert leaked["problems"] == ["in: 16 bytes of secret remain"], leaked

  missed = judge(parse(log(scenario("ctl", "arena", NEEDLE + bytes(24), arena=NEEDLE[:32]))))
  assert missed["problems"] == ["ctl: control secret not found in full in arena"], missed

  measured = judge(parse(log(scenario("by-value", "report", NEEDLE + NEEDLE + bytes(16)))))
  assert measured["passed"] and measured["scenarios"][0]["needles"]["secret"]["copies"] == 2, measured


def test_incomplete_runs_fail() -> None:
  truncated = judge(parse(log(scenario("a", "none", b"\xa5" * 32), declared=2, done=False)))
  assert truncated["problems"] == ["harness did not finish", "expected 2 scenarios, read 1"], truncated
  panicked = parse(log(scenario("a", "none", b"\xa5" * 32)).replace("done", "panic out of memory"))
  assert judge(panicked)["problems"][0] == "harness panicked: out of memory"
  try:
    parse(log(scenario("a", "none", b"\xa5" * 32)).replace("region stack 0x80000000 0x20", "region stack 0x80000000 0x40"))
  except ValueError:
    pass
  else:
    raise AssertionError("a short region must be rejected")
  try:
    parse(log(scenario("a", "sometimes", b"\xa5" * 32)))
  except ValueError:
    pass
  else:
    raise AssertionError("an unknown expectation must be rejected")


def test_native_sparse_hint_windows() -> None:
  # Cleared slots must not be mistaken for zero-valued hint coefficients.
  assert native_windows(bytes(1024), True) == []
  sparse = b"\x01\0\0\0" + bytes(1016) + b"\x01\0\0\0"
  assert native_windows(sparse, True) == [(0, sparse)]
  dense = (b"\x01\0\0\0" + bytes(12)) * 4 + bytes(960)
  assert (0, dense[:256]) in native_windows(dense, True)


def test_native_capture_completeness_and_injected_residue() -> None:
  import tempfile
  from pathlib import Path

  # A synthetic full ML-DSA-44 trace with one norm-rejected and one accepted
  # candidate. These are scanner fixtures, not a test implementation of signing.
  poly = bytes(range(256)) * 4
  hint = b"\x01\0\0\0" + bytes(1020)
  records = []
  for attempt in range(2):
    for stage in (1, 2, 4):
      records.extend((attempt, stage, index, poly) for index in range(4))
    records += [(attempt, 3, 0, OTHER), (attempt, 5, 0, bytes([1 - attempt]))]
  records += [(1, 6, index, hint) for index in range(4)] + [(1, 7, 0, b"\0")]
  capture = {"capture": records, "signature": bytes(2420)}
  selected, counters = native_needles(capture)
  assert counters == {"attempts": 2, "norm_rejections": 1, "hint_rejections": 0}
  assert any(value["kind"] == "rejected_commitment" for value in selected)
  truncated = {**capture, "capture": records[1:]}
  try:
    native_needles(truncated)
  except ValueError:
    pass
  else:
    raise AssertionError("a missing captured intermediate must fail")
  with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    stack = directory / "case.stack"
    measured = {"case": {"capture": [], "signature": bytes(2420)}}
    stack.write_bytes(b"\xa5" * 4096 + bytes(1024 * 1024 - 4096))
    assert native_judge({"case": capture}, measured, directory)[0]["passed"]
    # A single 16-byte commitment fragment must be enough to fail qualification.
    stack.write_bytes(b"\xa5" * 4096 + OTHER[:16] + bytes(1024 * 1024 - 4096 - 16))
    report = native_judge({"case": capture}, measured, directory)[0]
    assert not report["passed"]
    assert any(row["kind"] == "rejected_commitment" for row in report["cases"][0]["found"])
    # The first byte of a retained fragment may be indistinguishable from paint.
    # Trimming the painted prefix must preserve overlap for that full window.
    leading_paint = b"\xa5" + OTHER[:31]
    painted_capture = {**capture, "capture": [
      (attempt, stage, index, leading_paint if stage == 3 else data)
      for attempt, stage, index, data in records
    ]}
    stack.write_bytes(b"\xa5" * 4096 + leading_paint[:16] + bytes(1024 * 1024 - 4096 - 16))
    report = native_judge({"case": painted_capture}, measured, directory)[0]
    assert not report["passed"]
    assert any(row["kind"] == "rejected_commitment" for row in report["cases"][0]["found"])
    all_paint_capture = {**capture, "capture": [
      (attempt, stage, index, b"\xa5" * 32 if stage == 3 else data)
      for attempt, stage, index, data in records
    ]}
    try:
      native_judge({"case": all_paint_capture}, measured, directory)
    except ValueError:
      pass
    else:
      raise AssertionError("a needle equal to stack paint must fail closed")
    for corrupt in (b"", b"\xa5" * 4096, bytes(1024 * 1024)):
      stack.write_bytes(corrupt)
      try:
        native_judge({"case": capture}, measured, directory)
      except ValueError:
        pass
      else:
        raise AssertionError("truncated or exhausted stack must fail")


def main() -> None:
  test_windows_cover_every_byte()
  test_partial_and_scattered_residue()
  test_expectations()
  test_incomplete_runs_fail()
  test_native_sparse_hint_windows()
  test_native_capture_completeness_and_injected_residue()
  print("residue tests passed")


if __name__ == "__main__":
  main()
