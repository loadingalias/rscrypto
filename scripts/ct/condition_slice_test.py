#!/usr/bin/env -S uv run python
"""Regression tests for branch-condition slices used by waiver renewal."""

from pathlib import Path

from asm_heuristics import FunctionBody
from condition_slice import condition_slice


def body(lines: list[str], address: int = 0x1000) -> FunctionBody:
  return FunctionBody("f", Path("fixture.binary.disasm.txt"), address, list(enumerate(lines, start=1)))


def slice_of(target: str, lines: list[str], branch_line: int, address: int = 0x1000):
  function = body(lines, address)
  return condition_slice(target, function, None, branch_line, {"f", "g"})


def x86_loop(limit: str = "0x34", start: str = "1000: 31 c0\txorl\t%eax, %eax", inserted: list[str] | None = None):
  prefix = inserted or []
  shift = 3 * len(prefix)
  return [*prefix, start.replace("1000", f"{0x1000 + shift:x}"),
          f"{0x1002 + shift:x}: 48 83 c0 01\taddq\t$0x1, %rax",
          f"{0x1006 + shift:x}: 48 83 f8 {limit[2:]}\tcmpq\t${limit}, %rax",
          f"{0x100a + shift:x}: 75 f6\tjne\t0x{0x1002 + shift:x} <f+0x{0x2 + shift:x}>",
          f"{0x100c + shift:x}: c3\tretq"]


def main() -> None:
  target = "x86_64-unknown-linux-gnu"
  # The loop bound slice reaches the counter's initial value and its increment.
  original = slice_of(target, x86_loop(), 4)
  assert original.reasons == [], original.reasons
  assert len(original.nodes) == 4, original.canonical
  # Code that moves the loop but stays outside the slice keeps the canonical slice.
  moved = slice_of(target, x86_loop(inserted=["1000: 48 89 fe\tmovq\t%rdi, %rsi"]), 5)
  assert moved.canonical == original.canonical
  # A changed comparison is a changed slice.
  assert slice_of(target, x86_loop(limit="0x35"), 4).canonical != original.canonical
  # A counter loaded from memory, or returned by a call, needs manual review.
  loaded = slice_of(target, x86_loop(start="1000: 48 8b 47 08\tmovq\t0x8(%rdi), %rax"), 4)
  assert any(reason.startswith("non-constant load") for reason in loaded.reasons), loaded.reasons
  called = slice_of(target, [
    "1000: e8 fb 0f 00 00\tcallq\t0x2000 <g>",
    "1005: 48 85 c0\ttestq\t%rax, %rax",
    "1008: 74 01\tje\t0x100b <f+0xb>",
    "100a: 90\tnop",
    "100b: c3\tretq",
  ], 3)
  assert any(reason.startswith("call result") for reason in called.reasons), called.reasons
  # An 8-bit write keeps the upper bits, so the earlier definition stays in the slice.
  partial = slice_of(target, [
    "1000: 48 8b 07\tmovq\t(%rdi), %rax",
    "1003: 0f 94 c0\tsete\t%al",
    "1006: 48 85 c0\ttestq\t%rax, %rax",
    "1009: 74 01\tje\t0x100c <f+0xc>",
    "100b: 90\tnop",
    "100c: c3\tretq",
  ], 4)
  assert any(reason.startswith("non-constant load") for reason in partial.reasons), partial.canonical

  # AArch64 flags come from the last flag-setting instruction on every path.
  target = "aarch64-unknown-linux-gnu"
  merged = slice_of(target, [
    "1000: f100041f\tcmp\tx0, #0x1",
    "1004: 54000040\tb.eq\t0x100c <f+0xc>",
    "1008: f100081f\tcmp\tx0, #0x2",
    "100c: d2800001\tmov\tx1, #0x0",
    "1010: 54000041\tb.ne\t0x1018 <f+0x18>",
    "1014: d2800020\tmov\tx0, #0x1",
    "1018: d65f03c0\tret",
  ], 5)
  assert merged.reasons == [], merged.reasons
  assert sum("cmp" in node["identity"] for node in merged.nodes) == 2, merged.canonical
  assert merged.nodes[0]["inputs"] == {"nzcv": ["+0x0", "+0x8"]}, merged.nodes[0]

  # RISC-V compares two registers directly; function inputs are slice leaves.
  target = "riscv64gc-unknown-linux-gnu"
  riscv = slice_of(target, [
    "1000: 03200713\tli\ta4, 0x32",
    "1004: 00e5e463\tbltu\ta1, a4, 0x100c <f+0xc>",
    "1008: 00000513\tli\ta0, 0x0",
    "100c: 00008067\tret",
  ], 2)
  assert riscv.reasons == [] and riscv.nodes[0]["inputs"] == {"a1": ["entry"], "a4": ["+0x0"]}, riscv.nodes

  # POWER branches read the named CR field; a compare into another field is not in the slice.
  target = "powerpc64le-unknown-linux-gnu"
  power = slice_of(target, [
    "1000: 40 20 23 7c\tcmpld 3, 4",
    "1004: 40 20 a3 7c\tcmpld 1, 3, 5",
    "1008: 08 00 86 40\tbne 1, 0x1010 <f+0x10>",
    "100c: 00 00 60 38\tli 3, 0",
    "1010: 20 00 80 4e\tblr",
  ], 3)
  assert [node["offset"] for node in power.nodes] == [0x8, 0x4], power.canonical
  assert power.nodes[1]["inputs"] == {"r3": ["entry"], "r5": ["entry"]}, power.nodes

  # IBM Z conditions read CC; a load without CC effects does not end the search.
  target = "s390x-unknown-linux-gnu"
  s390x = slice_of(target, [
    "1000: b9 20 00 23\tcgr\t%r2, %r3",
    "1004: b9 04 00 42\tlgr\t%r4, %r2",
    "1008: a7 24 00 04\tjh\t0x1010",
    "100c: a7 49 00 00\tlghi\t%r4, 0",
    "1010: 07 fe\tbr\t%r14",
  ], 3)
  assert [node["offset"] for node in s390x.nodes] == [0x8, 0x0], s390x.canonical


if __name__ == "__main__":
  main()
