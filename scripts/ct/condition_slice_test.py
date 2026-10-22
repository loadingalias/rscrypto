#!/usr/bin/env -S uv run python
"""Regression tests for branch-condition slices used by waiver renewal."""

import re
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
  # A counter loaded through an input pointer that nothing stores to reads entry memory: a closed leaf.
  loaded = slice_of(target, x86_loop(start="1000: 48 8b 47 08\tmovq\t0x8(%rdi), %rax"), 4)
  assert loaded.reasons == [], loaded.reasons
  assert loaded.nodes[-1]["inputs"] == {"rdi": ["entry"], "mem": ["entry"]}, loaded.canonical
  # A counter returned by a call needs manual review.
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
  assert partial.reasons == [] and len(partial.nodes) == 4, partial.canonical

  # A definition before a return cannot reach code that only a branch enters.
  returned = slice_of(target, [
    "1000: 48 83 ff 01\tcmpq\t$0x1, %rdi",
    "1004: 74 06\tje\t0x100c <f+0xc>",
    "1006: b8 05 00 00 00\tmovl\t$0x5, %eax",
    "100b: c3\tretq",
    "100c: 48 85 c0\ttestq\t%rax, %rax",
    "100f: 74 01\tje\t0x1012 <f+0x12>",
    "1011: 90\tnop",
    "1012: c3\tretq",
  ], 6)
  assert returned.nodes[1]["inputs"] == {"rax": ["entry"]}, returned.canonical
  # Stack addresses are frame leaves; above the entry stack pointer, entry memory holds stack arguments.
  spilled = slice_of(target, [
    "1000: 48 8b 44 24 08\tmovq\t0x8(%rsp), %rax",
    "1005: 48 85 c0\ttestq\t%rax, %rax",
    "1008: 74 01\tje\t0x100b <f+0xb>",
    "100a: 90\tnop",
    "100b: c3\tretq",
  ], 3)
  assert spilled.nodes[2]["inputs"] == {"rsp": ["frame"], "mem": ["entry"]}, spilled.canonical
  assert spilled.reasons == [], spilled.reasons
  memory_regressions()

  # Win64 callees preserve rsi, so its value crosses the call; System V callees may change it.
  kept = [
    "1000: be 05 00 00 00\tmovl\t$0x5, %esi",
    "1005: e8 f6 0f 00 00\tcallq\t0x2000 <g>",
    "100a: 48 85 f6\ttestq\t%rsi, %rsi",
    "100d: 74 01\tje\t0x1010 <f+0x10>",
    "100f: 90\tnop",
    "1010: c3\tretq",
  ]
  windows = slice_of("x86_64-pc-windows-msvc", kept, 4)
  assert windows.reasons == [] and windows.nodes[1]["inputs"] == {"rsi": ["+0x0"]}, windows.canonical
  assert any(reason.startswith("call result") for reason in slice_of(target, kept, 4).reasons)
  # Win64 callees may change the upper bits of xmm6, so a vector value never crosses a call.
  vector = slice_of("x86_64-pc-windows-msvc", [
    "1000: 66 48 0f 6e f7\tmovq\t%rdi, %xmm6",
    "1005: e8 f6 0f 00 00\tcallq\t0x2000 <g>",
    "100a: 66 48 0f 7e f0\tmovq\t%xmm6, %rax",
    "100f: 48 85 c0\ttestq\t%rax, %rax",
    "1012: 74 01\tje\t0x1015 <f+0x15>",
    "1014: 90\tnop",
    "1015: c3\tretq",
  ], 5)
  assert any(reason.startswith("call result") for reason in vector.reasons), vector.reasons

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


def memory_regressions() -> None:
  target = "x86_64-unknown-linux-gnu"
  # A spilled value reloaded after a call: the frame slot is private, so the reload reads the spill.
  spill = [
    "1000: 48 83 ec 18\tsubq\t$0x18, %rsp",
    "1004: 48 89 7c 24 08\tmovq\t%rdi, 0x8(%rsp)",
    "1009: e8 f2 0f 00 00\tcallq\t0x2000 <g>",
    "100e: 48 8b 44 24 08\tmovq\t0x8(%rsp), %rax",
    "1013: 48 85 c0\ttestq\t%rax, %rax",
    "1016: 74 01\tje\t0x1019 <f+0x19>",
    "1018: 90\tnop",
    "1019: c3\tretq",
  ]
  reloaded = slice_of(target, spill, 6)
  assert reloaded.reasons == [], reloaded.reasons
  load = next(node for node in reloaded.nodes if node["offset"] == 0xe)
  assert load["inputs"]["mem"] == ["+0x4"], reloaded.canonical
  store = next(node for node in reloaded.nodes if node["offset"] == 0x4)
  assert store["inputs"]["rdi"] == ["entry"], reloaded.canonical
  # Code outside the slice, even a store to another slot, keeps the canonical slice.
  moved = [spill[0], "1004: 48 89 74 24 10\tmovq\t%rsi, 0x10(%rsp)",
           *[shift(line, 5) for line in spill[1:]]]
  assert slice_of(target, moved, 7).canonical == reloaded.canonical
  # A store that may overwrite the slot joins the slice.
  overlapped = [spill[0], spill[1], "1009: 89 74 24 0c\tmovl\t%esi, 0xc(%rsp)",
                *[shift(line, 4) for line in spill[2:]]]
  overlap = slice_of(target, overlapped, 7)
  assert next(node for node in overlap.nodes if node["offset"] == 0x12)["inputs"]["mem"] == ["+0x4", "+0x9"]
  # A frame address passed to the callee lets the call write the slot.
  escaped = [spill[0], spill[1], "1009: 48 8d 7c 24 08\tleaq\t0x8(%rsp), %rdi",
             *[shift(line, 5) for line in spill[2:]]]
  assert any(reason.startswith("memory may be written by the call") for reason in slice_of(target, escaped, 7).reasons)

  # A field of an input pointer: a store through another pointer may alias it, a disjoint field cannot.
  def field_load(store: str) -> list[str]:
    return [store, "1004: 48 8b 47 08\tmovq\t0x8(%rdi), %rax", "1008: 48 85 c0\ttestq\t%rax, %rax",
            "100b: 74 01\tje\t0x100e <f+0xe>", "100d: 90\tnop", "100e: c3\tretq"]
  other = slice_of(target, field_load("1000: 48 89 16 90\tmovq\t%rdx, (%rsi)"), 4)
  assert other.reasons == [] and other.nodes[-2]["inputs"]["mem"] == ["entry", "+0x0"], other.canonical
  disjoint = slice_of(target, field_load("1000: 48 89 57 10\tmovq\t%rdx, 0x10(%rdi)"), 4)
  assert disjoint.nodes[-1]["inputs"]["mem"] == ["entry"] and len(disjoint.nodes) == 3, disjoint.canonical
  called = slice_of(target, field_load("1000: e8 fb 0f 00\tcallq\t0x2000 <g>"), 4)
  assert any(reason.startswith("memory may be written by the call") for reason in called.reasons), called.reasons

  # Each ISA: spill an argument, call, reload, and branch on it.
  cases = {
    "aarch64-unknown-linux-gnu": ([
      "1000: a9be7bfd\tstp\tx29, x30, [sp, #-0x20]!",
      "1004: 910003fd\tmov\tx29, sp",
      "1008: f9000be0\tstr\tx0, [sp, #0x10]",
      "100c: 94000000\tbl\t0x2000 <g>",
      "1010: f9400be8\tldr\tx8, [sp, #0x10]",
      "1014: b4000048\tcbz\tx8, 0x101c <f+0x1c>",
      "1018: d503201f\tnop",
      "101c: a8c27bfd\tldp\tx29, x30, [sp], #0x20",
      "1020: d65f03c0\tret",
    ], 6, 0x10, "+0x8"),
    "riscv64gc-unknown-linux-gnu": ([
      "1000: 1141\taddi\tsp, sp, -0x10",
      "1002: e42a\tsd\ta0, 0x8(sp)",
      "1004: 00000097\tjal\t0x2000 <g>",
      "1008: 67a2\tld\ta5, 0x8(sp)",
      "100a: c391\tbeqz\ta5, 0x1010 <f+0x10>",
      "100c: 0001\tnop",
      "100e: 0141\taddi\tsp, sp, 0x10",
      "1010: 8082\tret",
    ], 5, 0x8, "+0x2"),
    "powerpc64le-unknown-linux-gnu": ([
      "1000: d1 ff 21 f8\tstdu 1, -48(1)",
      "1004: 20 00 61 f8\tstd 3, 32(1)",
      "1008: 01 00 00 48\tbl 0x2000 <g>",
      "100c: 20 00 21 e9\tld 9, 32(1)",
      "1010: 00 00 29 2c\tcmpdi 9, 0",
      "1014: 08 00 82 41\tbeq 0x101c <f+0x1c>",
      "1018: 00 00 00 60\tnop",
      "101c: 20 00 80 4e\tblr",
    ], 6, 0xc, "+0x4"),
    "s390x-unknown-linux-gnu": ([
      "1000: eb 6f f0 30 00 24\tstmg\t%r6, %r15, 48(%r15)",
      "1006: a7 fb ff 60\taghi\t%r15, -160",
      "100a: e3 20 f0 a8 00 24\tstg\t%r2, 168(%r15)",
      "1010: c0 e5 00 00 00 00\tbrasl\t%r14, 0x2000 <g>",
      "1016: e3 10 f0 a8 00 04\tlg\t%r1, 168(%r15)",
      "101c: b9 02 00 11\tltgr\t%r1, %r1",
      "1020: a7 84 00 04\tje\t0x1028",
      "1024: 07 07\tnopr\t%r7",
      "1028: 07 fe\tbr\t%r14",
    ], 7, 0x16, "+0xa"),
  }
  for isa, (lines, branch, load_offset, store) in cases.items():
    result = slice_of(isa, lines, branch)
    assert result.reasons == [], (isa, result.reasons, result.canonical)
    load = next(node for node in result.nodes if node["offset"] == load_offset)
    assert load["inputs"]["mem"] == [store], (isa, result.canonical)


def shift(line: str, by: int) -> str:
  """Move one fixture line, and a local branch target in it, `by` bytes later."""
  address, rest = line.split(":", 1)
  moved = f"{int(address, 16) + by:x}:{rest}"
  return re.sub(r"0x([0-9a-f]+) <f\+0x([0-9a-f]+)>",
                lambda match: f"0x{int(match.group(1), 16) + by:x} <f+0x{int(match.group(2), 16) + by:x}>", moved)


if __name__ == "__main__":
  main()
