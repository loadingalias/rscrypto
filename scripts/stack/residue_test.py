#!/usr/bin/env python3
"""Regression tests for residue log parsing and judgement, on synthetic logs."""

from __future__ import annotations

from residue import found_bytes, judge, parse, windows

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


def main() -> None:
  test_windows_cover_every_byte()
  test_partial_and_scattered_residue()
  test_expectations()
  test_incomplete_runs_fail()
  print("residue tests passed")


if __name__ == "__main__":
  main()
