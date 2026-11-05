#!/usr/bin/env -S uv run python
"""Regression tests for the linked-frame review, using synthetic disassembly."""

from __future__ import annotations

import dataclasses
import json
import re
import struct
import tempfile
from pathlib import Path

from frames import (
  ElfImage,
  apply_budgets,
  load_budgets,
  stack_budget,
  updated_budgets,
  PREPARATION_ROOTS,
  WHOLE_CALL_GROUPS,
  whole_call_roots,
  Boundary,
  load_allowances,
  parse_disassembly,
  parse_dynamic_symbols,
  parse_stack_sizes,
  parse_symbols,
  review,
  resolve_jump_tables,
  review_preparation,
)

WORKER = "rscrypto::fixture::worker"
SCRUB = "rscrypto::fixture::scrub_dead_stack"
CALLER = "rscrypto::fixture::hash"


def boundary(directory: Path, words: int = 256) -> Boundary:
  source = directory / "fixture.rs"
  source.write_text(f"const STACK_SCRUB_WORDS: usize = {words};\n")
  return Boundary(
    name="fixture",
    workers=re.compile(rf"^{re.escape(WORKER)}$"),
    scrub=re.compile(rf"^{re.escape(SCRUB)}$"),
    source=source,
    constant="STACK_SCRUB_WORDS",
  )


def program(target: str, functions: list[tuple[int, str, int | None, list[str]]], dynamic: str = ""):
  """Build a program from (address, name, frame, instruction lines) tuples.

  Names double as mangled and demangled symbols, which the review accepts.
  """
  symbols = {address: (name, name) for address, name, _, _ in functions if not name.endswith("@plt")}
  sizes = {name: frame for _, name, frame, _ in functions if frame is not None}
  lines = []
  for address, name, _, body in functions:
    lines.append(f"{address:016x} <{name}>:")
    lines.extend(body)
    lines.append("")
  return parse_disassembly(target, "\n".join(lines), symbols, sizes, parse_dynamic_symbols(dynamic))


def worker_report(report: dict) -> dict:
  return report["boundaries"][0]["workers"][0]


def test_s390x_detection_is_reported_not_dropped(directory: Path) -> None:
  functions = [
    (0x1000, CALLER, 160, ["    1000:      \tbrasl\t%r14, 0x2000", "    1006:      \tbrasl\t%r14, 0x3000", "    100c:      \tbr\t%r14"]),
    (0x2000, WORKER, 816, ["    2000:      \tbrasl\t%r14, 0x4000", "    2006:      \tbrasl\t%r14, 0x5000", "    200c:      \tbr\t%r14"]),
    (0x3000, SCRUB, 2208, ["    3000:      \tbr\t%r14"]),
    (0x4000, "rscrypto::keccak::keccakf_portable", 400, ["    4000:      \tbr\t%r14"]),
    (
      0x5000,
      "<rscrypto::platform::detect::atomic_cache::Cache>::initialize::<rscrypto::platform::detect::detect_with_override>",
      192,
      ["    5000:      \tbrasl\t%r14, 0x6000", "    5006:      \tbr\t%r14"],
    ),
    (0x6000, "<core::fmt::Formatter>::write_fmt", None, ["    6000:      \tbasr\t%r14, %r1"]),
  ]
  report = review(program("s390x-unknown-linux-gnu", functions), [boundary(directory)])
  worker = worker_report(report)
  assert not report["passed"]
  assert worker["depth"] == 816 + 400, worker
  assert len(worker["detection"]) == 1 and "platform::detect" in worker["detection"][0][-1]
  assert [item["reason"] for item in worker["unbounded"]] == ["no frame record", "indirect"], worker["unbounded"]
  assert worker["callers"] == [CALLER]


def test_detection_free_worker_passes(directory: Path) -> None:
  functions = [
    (0x1000, CALLER, 160, ["    1000:      \tbrasl\t%r14, 0x2000", "    1006:      \tbrasl\t%r14, 0x3000"]),
    (0x2000, WORKER, 816, ["    2000:      \tbrasl\t%r14, 0x4000", "    2006:      \tjg\t0x2000"]),
    (0x3000, SCRUB, 2208, ["    3000:      \tbr\t%r14"]),
    (0x4000, "rscrypto::keccak::keccakf_portable", 400, ["    4000:      \tbr\t%r14"]),
  ]
  report = review(program("s390x-unknown-linux-gnu", functions), [boundary(directory)])
  assert report["passed"], report


def test_depth_beyond_scrub_fails(directory: Path) -> None:
  functions = [
    (0x1000, CALLER, 160, ["    1000:      \tbrasl\t%r14, 0x2000", "    1006:      \tbrasl\t%r14, 0x3000"]),
    (0x2000, WORKER, 1500, ["    2000:      \tbrasl\t%r14, 0x4000"]),
    (0x3000, SCRUB, 2208, ["    3000:      \tbr\t%r14"]),
    (0x4000, "rscrypto::keccak::keccakf_portable", 600, ["    4000:      \tbr\t%r14"]),
  ]
  report = review(program("s390x-unknown-linux-gnu", functions), [boundary(directory)])
  assert worker_report(report)["depth"] == 2100
  assert any("exceeds" in problem for problem in worker_report(report)["problems"])


def test_caller_without_scrub_and_recursion_fail(directory: Path) -> None:
  functions = [
    (0x1000, CALLER, 160, ["    1000:      \tbrasl\t%r14, 0x2000", "    1006:      \tbrasl\t%r14, 0x3000"]),
    (0x1800, "rscrypto::fixture::unscrubbed", 160, ["    1800:      \tbrasl\t%r14, 0x2000"]),
    (0x2000, WORKER, 100, ["    2000:      \tbrasl\t%r14, 0x2000"]),
    (0x3000, SCRUB, 2208, ["    3000:      \tbr\t%r14"]),
  ]
  worker = worker_report(review(program("s390x-unknown-linux-gnu", functions), [boundary(directory)]))
  assert any("rscrypto::fixture::unscrubbed" in problem for problem in worker["problems"]), worker
  assert [item["reason"] for item in worker["unbounded"]] == ["recursion"], worker


def test_power_local_entries_returns_and_red_zone(directory: Path) -> None:
  functions = [
    (0x1000, CALLER, 112, ["    1000:      \tbl 0x2008 <worker+0x8>", "    1004:      \tbl 0x3008 <scrub+0x8>", "    1008:      \tblr"]),
    (0x2000, WORKER, 496, ["    2008:      \tbl 0x4008 <leaf+0x8>", "    200c:      \tbl 0x9000 <__plt_memcpy>", "    2010:      \tbeqlr", "    2014:      \tblr"]),
    (0x3000, SCRUB, 2080, ["    3008:      \tblr"]),
    (0x4000, "rscrypto::keccak::keccakf_portable", 0, ["    4008:      \tblr"]),
    (0x9000, "__plt_memcpy", None, ["    9000:      \tb 0x9100"]),
  ]
  worker = worker_report(review(program("powerpc64le-unknown-linux-gnu", functions), [boundary(directory)]))
  # A frameless leaf may still use the 288-byte protected zone.
  assert worker["depth"] == 496 + 288, worker
  assert worker["assumptions"] == ["memcpy: external leaf, frame 0 + red zone"], worker
  assert not worker["problems"], worker

  indirect = [
    (0x1000, CALLER, 112, ["    1000:      \tbl 0x2008 <worker+0x8>", "    1004:      \tbl 0x3008 <scrub+0x8>"]),
    (0x2000, WORKER, 96, ["    2008:      \tbctrl"]),
    (0x3000, SCRUB, 2080, ["    3008:      \tblr"]),
  ]
  worker = worker_report(review(program("powerpc64le-unknown-linux-gnu", indirect), [boundary(directory)]))
  assert [item["reason"] for item in worker["unbounded"]] == ["indirect"], worker


def test_riscv_auipc_pairs_labels_and_returns(directory: Path) -> None:
  functions = [
    (
      0x10000,
      CALLER,
      32,
      [
        "   10000:      \tauipc\tra, 0x1",
        "   10004:      \tjalr\t0x0(ra) <worker>",
        "   10008:      \tjal\t0x30000 <scrub>",
        "   1000c:      \tret",
      ],
    ),
    (
      0x11000,
      WORKER,
      464,
      [
        "   11000:      \tauipc\tra, 0xfffff",
        "0000000000011004 <.Lpcrel_hi1>:",
        "   11004:      \tjalr\t-0x4(ra) <memset@plt>",
        "   11008:      \tauipc\tt1, 0x1f",
        "   1100c:      \tjr\t-0xc(t1) <leaf>",
      ],
    ),
    (0x30000, SCRUB, 2064, ["   30000:      \tret"]),
    (0x0FFFC, "memset@plt", None, ["    fffc:      \tjr\tt3"]),
    (0x2FFFC, "rscrypto::keccak::keccakf_portable", 320, ["   2fffc:      \tret"]),
  ]
  report = review(program("riscv64gc-unknown-linux-gnu", functions), [boundary(directory)])
  worker = worker_report(report)
  # The negative auipc immediate reaches the PLT stub below the worker, and the
  # tail call through t1 replaces the worker's frame with the leaf's.
  assert worker["assumptions"] == ["memset: external leaf, frame 0 + red zone"], worker
  assert worker["deepest_path"] == [WORKER], worker
  assert worker["depth"] == 464, worker
  assert worker["callers"] == [CALLER], worker
  assert report["passed"], report

  stale = [
    (0x10000, CALLER, 32, ["   10000:      \tauipc\tra, 0x1", "   10004:      \tjalr\t0x0(ra)", "   10008:      \tjal\t0x30000"]),
    (0x11000, WORKER, 64, ["   11000:      \tauipc\ta5, 0x10", "   11004:      \tld\ta5, 0x8(a5)", "   11008:      \tjalr\ta5"]),
    (0x30000, SCRUB, 2064, ["   30000:      \tret"]),
  ]
  worker = worker_report(review(program("riscv64gc-unknown-linux-gnu", stale), [boundary(directory)]))
  assert [item["reason"] for item in worker["unbounded"]] == ["indirect"], worker


def test_x86_got_imports_call_slots_and_red_zone(directory: Path) -> None:
  dynamic = "\n".join(
    [
      "OFFSET           TYPE                     VALUE",
      "0000000000009000 R_X86_64_GLOB_DAT        memcpy@GLIBC_2.14",
      "0000000000009008 R_X86_64_GLOB_DAT        getauxval",
      "0000000000009010 R_X86_64_RELATIVE        *ABS*+0x51d50",
    ]
  )
  functions = [
    (0x1000, CALLER, 24, ["    1000:      \tcallq\t0x2000 <worker>", "    1005:      \tcallq\t0x3000 <scrub>", "    100a:      \tretq"]),
    (
      0x2000,
      WORKER,
      472,
      [
        "    2000:      \tcallq\t*0x7000(%rip)           # 0x9000 <write+0x9000>",
        "    2006:      \tjne\t0x2000 <worker>",
        "    2008:      \tcallq\t0x4000 <leaf>",
      ],
    ),
    (0x3000, SCRUB, 1920, ["    3000:      \tretq"]),
    (0x4000, "rscrypto::keccak::keccakf_portable", 200, ["    4000:      \tretq"]),
  ]
  report = review(program("x86_64-unknown-linux-gnu", functions, dynamic), [boundary(directory)])
  worker = worker_report(report)
  # Caller's return slot, worker frame, leaf return slot, leaf frame, red zone.
  assert worker["depth"] == 8 + 472 + 8 + 200 + 128, worker
  assert worker["assumptions"] == ["memcpy: external leaf, frame 0 + red zone"], worker
  # The 1,920-byte scrub frame keeps 128 bytes of its buffer in the red zone.
  assert report["passed"], report

  functions[1] = (0x2000, WORKER, 64, ["    2000:      \tcallq\t*0x7008(%rip)           # 0x9008 <write+0x9008>"])
  worker = worker_report(review(program("x86_64-unknown-linux-gnu", functions, dynamic), [boundary(directory)]))
  assert worker["unbounded"] == [{"reason": "external", "path": [WORKER, "getauxval"], "address": "0x2000"}], worker
  assert worker["detection"] == [[WORKER, "getauxval"]], worker


def test_x86_register_held_imports(directory: Path) -> None:
  dynamic = "0000000000009000 R_X86_64_GLOB_DAT        memcpy"
  caller = (0x1000, CALLER, 24, ["    1000:      \tcallq\t0x2000 <worker>", "    1005:      \tcallq\t0x3000 <scrub>"])
  scrub = (0x3000, SCRUB, 1920, ["    3000:      \tretq"])
  leaf = (0x4000, "rscrypto::keccak::keccakf_portable", 64, ["    4000:      \tretq"])

  def worker(*body: str) -> dict:
    functions = [caller, (0x2000, WORKER, 128, list(body)), scrub, leaf]
    return worker_report(review(program("x86_64-unknown-linux-gnu", functions, dynamic), [boundary(directory)]))

  load = "    2000:      \tmovq\t0x7000(%rip), {}     # 0x9000 <write+0x9000>"
  # A callee-saved register keeps the import across calls.
  held = worker(load.format("%r12"), "    2007:      \tcallq\t*%r12", "    200a:      \tcallq\t0x4000 <leaf>",
                "    200f:      \tcallq\t*%r12")
  assert not held["unbounded"] and held["assumptions"] == ["memcpy: external leaf, frame 0 + red zone"], held
  # Overwriting the register, or a call clobbering a caller-saved one, ends it.
  overwritten = worker(load.format("%r12"), "    2007:      \tmovq\t%rax, %r12", "    200a:      \tcallq\t*%r12")
  assert [item["reason"] for item in overwritten["unbounded"]] == ["indirect"], overwritten
  clobbered = worker(load.format("%rax"), "    2007:      \tcallq\t0x4000 <leaf>", "    200c:      \tcallq\t*%rax")
  assert [item["reason"] for item in clobbered["unbounded"]] == ["indirect"], clobbered
  partial = worker(load.format("%r12"), "2007: movl %eax, %r12d", "200a: callq *%r12")
  assert [item["reason"] for item in partial["unbounded"]] == ["indirect"], partial
  # A load dominating the entire loop remains proved at both calls. Clearing
  # every join would hide this real import behind an unnecessary unknown edge.
  loop = worker(load.format("%r12"), "2007: callq *%r12", "200a: jne 0x2007", "200c: callq *%r12")
  assert not loop["unbounded"] and loop["assumptions"] == ["memcpy: external leaf, frame 0 + red zone"], loop
  copied = worker(load.format("%r12"), "2007: movq %r12, %r13", "200a: movl %eax, %r12d", "200d: callq *%r13")
  assert not copied["unbounded"] and copied["assumptions"] == ["memcpy: external leaf, frame 0 + red zone"], copied
  # LLVM may copy the held import into a caller-saved register for the last call
  # (x86-64 ML-KEM key generation: `movq %rbx, %rax` ... `callq *%rax`), or load
  # the GOT slot into one directly. Explicit-destination instructions keep it.
  for body in (
    (load.format("%rbx"), "2007: callq *%rbx", "2009: movq %rbx, %rax", "200c: leaq 0x7c0(%rsp), %rbx",
     "2014: movl $0x300, %edx", "2019: xorl %esi, %esi", "201b: callq *%rax"),
    (load.format("%r8"), "2007: leaq 0x10(%rsp), %rdi", "200c: callq *%r8"),
  ):
    held = worker(*body)
    assert not held["unbounded"] and held["assumptions"] == ["memcpy: external leaf, frame 0 + red zone"], held
  # An implicit writer, a call, or a partial write between the copy and the call ends the proof.
  for between in ("200c: mulq %rcx", "200c: cqto", "200c: callq 0x4000 <leaf>", "200c: movl %ecx, %eax",
                  "200c: movb $0x1, %ah"):
    lost = worker(load.format("%rbx"), "2007: callq *%rbx", "2009: movq %rbx, %rax", between, "2011: callq *%rax")
    assert [item["reason"] for item in lost["unbounded"]] == ["indirect"], (between, lost)


def test_aarch64_leaf_assembly_frames(directory: Path) -> None:
  def quad(*body: str) -> dict:
    functions = [
      (0x1000, CALLER, 32, ["    1000:      \tbl\t0x2000 <worker>", "    1004:      \tbl\t0x3000 <scrub>"]),
      (0x2000, WORKER, 1888, ["    2000:      \tbl\t0x5000 <kernel>", "    2004:      \tret"]),
      (0x3000, SCRUB, 2064, ["    3000:      \tret"]),
      (0x5000, "rscrypto_keccakf1600_aarch64_sve2_sha3_x4", None, list(body)),
    ]
    return worker_report(review(program("aarch64-unknown-linux-gnu", functions), [boundary(directory, words=512)]))

  derived = quad(
    "    5000:      \tsub\tsp, sp, #0x60",
    "    5004:      \tstp\td8, d9, [sp]",
    "    5008:      \tstp\tx29, x30, [sp, #-16]!",
    "    500c:      \tadd\tsp, sp, #0x70",
    "    5010:      \tret",
  )
  assert derived["depth"] == 1888 + 96 + 16 and not derived["problems"], derived
  assert derived["assumptions"] == ["rscrypto_keccakf1600_aarch64_sve2_sha3_x4: leaf assembly frame 112 B"], derived
  for body in (["    5000:      \tmov\tsp, x9"], ["    5000:      \tsub\tsp, sp, x9"], ["    5000:      \tbl\t0x2000"]):
    unknown = quad(*body, "    5004:      \tret")
    assert any(item["reason"] in {"no frame record", "recursion"} for item in unknown["unbounded"]), unknown


def test_aarch64_tail_scrub_counts_the_caller_frame(directory: Path) -> None:
  def tail(caller_frame: int) -> dict:
    functions = [
      (
        0x1000,
        CALLER,
        caller_frame,
        ["    1000:      \tbl\t0x2000 <worker>", "    1004:      \tadd\tsp, sp, #0xc0", "    1008:      \tb\t0x3000 <scrub>"],
      ),
      (0x2000, WORKER, 1648, ["    2000:      \tret"]),
      (0x3000, SCRUB, 2064, ["    3000:      \tret"]),
    ]
    return worker_report(review(program("aarch64-unknown-linux-gnu", functions), [boundary(directory)]))

  # The scrub runs from the caller's entry, so the caller frame plus the worker must fit.
  covered = tail(192)
  assert not covered["problems"] and covered["tail_scrubbed_callers"] == [CALLER], covered
  oversized = tail(512)
  assert any("frame 512 + depth 1648 exceeds" in problem for problem in oversized["problems"]), oversized
  assert oversized["tail_scrubbed_callers"] == [], oversized


def test_panic_exits_are_listed_and_not_counted(directory: Path) -> None:
  functions = [
    (0x1000, CALLER, 32, ["    1000:      \tbl\t0x2000 <worker>", "    1004:      \tbl\t0x3000 <scrub>"]),
    (0x2000, WORKER, 128, ["    2000:      \tbl\t0x4000 <panic>", "    2004:      \tb.ne\t0x2000 <worker>", "    2008:      \tret"]),
    (0x3000, SCRUB, 2064, ["    3000:      \tret"]),
    (0x4000, "core::panicking::panic_bounds_check", None, ["    4000:      \tbrk\t#0x1"]),
  ]
  worker = worker_report(review(program("aarch64-unknown-linux-gnu", functions), [boundary(directory)]))
  assert worker["panic_exits"] == ["core::panicking::panic_bounds_check"], worker
  assert worker["depth"] == 128 and not worker["problems"], worker


def test_target_and_portable_scoped_boundaries_are_skipped(directory: Path) -> None:
  # No worker is linked; a boundary that applies would fail with "no linked worker".
  functions = [(0x1000, CALLER, 160, ["    1000:      \tbr\t%r14"])]
  x86_only = dataclasses.replace(boundary(directory), targets=frozenset({"x86_64-unknown-linux-gnu"}))
  report = review(program("s390x-unknown-linux-gnu", functions), [x86_only])
  assert report["passed"] and not report["boundaries"], report
  native_only = dataclasses.replace(boundary(directory), portable=False)
  assert review(program("s390x-unknown-linux-gnu", functions), [native_only], portable=True)["passed"]
  assert not review(program("s390x-unknown-linux-gnu", functions), [native_only])["passed"]


def test_record_parsers() -> None:
  sizes = parse_stack_sizes(
    "StackSizes [\n  Entry {\n    Functions: [a, b]\n    Size: 0x330\n  }\n  Entry {\n    Functions: [c]\n    Size: 0x10\n  }\n]"
  )
  assert sizes == {"a": 0x330, "b": 0x330, "c": 0x10}
  symbols = parse_symbols(
    "0000000000001000 t _RNvC1a1f\n0000000000001010 t .Lpcrel_hi4\n0000000000002000 d _RNvC1a4data\n",
    "0000000000001000 t a::f\n0000000000001010 t .Lpcrel_hi4\n0000000000002000 d a::data\n",
  )
  assert symbols == {0x1000: ("_RNvC1a1f", "a::f")}
  try:
    parse_symbols("0000000000001000 t x\n", "0000000000001004 t x\n")
  except ValueError:
    pass
  else:
    raise AssertionError("misaligned symbol tables must be rejected")


def test_preparation_never_turns_unknown_frames_into_a_bound() -> None:
  functions = [
    (0x1000, "prepare", 32, ["1000: callq 0x2000", "1005: callq 0x3000", "100a: retq"]),
    # A dynamic stack allocation: no frame record, and no frame derivable from the instructions.
    (0x2000, "allocator_without_frame", None, [
      "2000: subq %rdi, %rsp", "2003: callq *0x6000(%rip) # 0x8000", "2009: retq",
    ]),
    (0x3000, "cleanup", 64, ["3000: callq *0x5008(%rip) # 0x8008", "3005: retq"]),
  ]
  dynamic = "8000 R_X86_64_GLOB_DAT calloc\n8008 R_X86_64_GLOB_DAT free"
  report = review_preparation(program("x86_64-unknown-linux-gnu", functions, dynamic), ["prepare"])
  call = report["whole_calls"][0]
  assert not report["passed"] and call["whole_call_bound_bytes"] is None, call
  assert call["frame_bytes"] == 32 and call["recorded_frame_path_bytes"] > 32
  assert {(item["reason"], item["path"][-1]) for item in call["unresolved"]} == {
    ("no frame record", "allocator_without_frame"), ("external", "calloc"), ("external", "free")
  }, call
  assert all("address" in item for item in call["unresolved"])
  assert not review_preparation(program("x86_64-unknown-linux-gnu", functions), ["absent"])["passed"]


def test_preparation_resolves_relative_calls_but_not_ifuncs_or_memory_imports() -> None:
  functions = [
    (0x1000, "prepare", 32, ["1000: callq *0x7000(%rip) # 0x8000", "1005: retq"]),
    (0x2000, "leaf", 96, ["2000: retq"]),
  ]
  dynamic = "8000 R_X86_64_RELATIVE *ABS*+0x2000"
  call = review_preparation(program("x86_64-unknown-linux-gnu", functions, dynamic), ["prepare"])["whole_calls"][0]
  assert call["whole_call_bound_bytes"] == 8 + 32 + 8 + 96 + 128, call
  assert call["whole_call_bound_basis"].startswith("static") and call["owned_bound_bytes"] == 272, call
  for relocation, owned in (("8000 R_X86_64_IRELATIVE *ABS*+0x2000", None), ("8000 R_X86_64_GLOB_DAT memcpy", 8 + 32 + 8)):
    call = review_preparation(program("x86_64-unknown-linux-gnu", functions, relocation), ["prepare"])["whole_calls"][0]
    assert call["whole_call_bound_bytes"] is None and not call["assumptions"], call
    assert call["unresolved"][0]["reason"] == "external", call
    assert call["owned_bound_bytes"] == owned, call


def test_got_value_cannot_cross_an_unproved_control_flow_join() -> None:
  functions = [(0x1000, "prepare", 32, [
    "1000: jne 0x1010", "1002: movq 0x7000(%rip), %r12 # 0x8000", "1010: callq *%r12", "1013: retq",
  ])]
  call = review_preparation(program("x86_64-unknown-linux-gnu", functions, "8000 R_X86_64_GLOB_DAT memcpy"), ["prepare"])["whole_calls"][0]
  assert call["unresolved"][0]["reason"] == "indirect", call


def test_branch_beyond_last_function_is_not_a_recursive_call() -> None:
  functions = [(0x1000, "prepare", 32, ["1000: callq 0x9000", "1005: retq"])]
  call = review_preparation(program("x86_64-unknown-linux-gnu", functions), ["prepare"])["whole_calls"][0]
  assert call["unresolved"][0]["reason"] == "external", call
  assert call["unresolved"][0]["path"][-1] == "0x9000", call


def test_assembly_loop_cannot_hide_repeated_stack_growth() -> None:
  functions = [(0x1000, "prepare", None, [
    "1000: sub sp, sp, #16", "1004: cbnz x0, 0x1000", "1008: ret",
  ])]
  call = review_preparation(program("aarch64-unknown-linux-gnu", functions), ["prepare"])["whole_calls"][0]
  assert call["whole_call_bound_bytes"] is None, call
  assert call["unresolved"][0]["reason"] == "no frame record", call


def whole_call(target: str, functions: list[tuple[int, str, int | None, list[str]]]) -> dict:
  return review_preparation(program(target, functions), ["prepare"])["whole_calls"][0]


# A VAES-style leaf: two pushes, then `rsp` saved, lowered, aligned to 64, stored at +0x200, and reloaded.
SAVE_SLOT_KERNEL = [
  "2000: pushq %r12", "2002: pushq %r13", "2004: cmpq $0x100, %rcx", "200b: jae 0x2019",
  "200d: popq %r13", "200f: popq %r12", "2011: movq $0x0, -0x10(%rsp)", "2018: retq",
  "2019: movq %rsp, %rax", "201c: subq $0x240, %rsp # imm = 0x240", "2023: andq $-0x40, %rsp",
  "2027: movq %rax, 0x200(%rsp)", "202f: vmovdqu64 %zmm20, 0x1c0(%rsp)", "2037: cmpq $0x1, %rdx",
  "203b: jne 0x202f", "203d: movq 0x200(%rsp), %rsp", "2045: popq %r13", "2047: popq %r12", "2049: retq",
]


def test_x86_save_slot_assembly_frame() -> None:
  def call(body: list[str]) -> dict:
    functions = [
      (0x1000, "prepare", 32, ["1000: callq 0x2000", "1005: retq"]),
      (0x2000, "rscrypto_fixture_gcm_kernel", None, body),
      (0x3000, "rscrypto::fixture::leaf", 0, ["3000: retq"]),
    ]
    return whole_call("x86_64-unknown-linux-gnu", functions)

  # The root's return address and frame, the call's return address, then 16 B of pushes, 0x240, at most 56 B of
  # alignment, and the 128-byte red zone of a leaf.
  bounded = call(SAVE_SLOT_KERNEL)
  assert bounded["owned_bound_bytes"] == 8 + 32 + 8 + 16 + 0x240 + 56 + 128, bounded
  assert bounded["assumptions"] == [
    "rscrypto_fixture_gcm_kernel: assembly frame 648 B "
    "(1 symbol(s), rsp 8-byte aligned at entry, rsp reloaded from a verified save slot)"
  ], bounded
  refused = {
    "a store overlaps the save slot": ("202f: vmovdqu64 %zmm20, 0x1c0(%rsp)", "202f: vmovdqu64 %zmm20, 0x1c8(%rsp)"),
    "a branch enters the span past the store": ("200b: jae 0x2019", "200b: jae 0x2037"),
    "the stack address escapes": ("2037: cmpq $0x1, %rdx", "2037: leaq 0x10(%rsp), %rdi"),
    "a call runs inside the span": ("2037: cmpq $0x1, %rdx", "2037: callq 0x3000"),
    "the reload reads another slot": ("203d: movq 0x200(%rsp), %rsp", "203d: movq 0x1f8(%rsp), %rsp"),
    "rsp comes from a register": ("203d: movq 0x200(%rsp), %rsp", "203d: movq %rax, %rsp"),
    "an access falls below the red zone": ("2011: movq $0x0, -0x10(%rsp)", "2011: movq $0x0, -0x100(%rsp)"),
    "the alignment repeats in a loop": ("203b: jne 0x202f", "203b: jne 0x2023"),
    "the slot lies outside the allocation": ("201c: subq $0x240, %rsp # imm = 0x240", "201c: subq $0x200, %rsp"),
    "the stack address escapes before the save": ("2004: cmpq $0x100, %rcx", "2004: leaq 0x8(%rsp), %rdi"),
  }
  for reason, (old, new) in refused.items():
    assert old in SAVE_SLOT_KERNEL, reason
    unknown = call([new if line == old else line for line in SAVE_SLOT_KERNEL])
    assert unknown["owned_bound_bytes"] is None, (reason, unknown)
    assert [(item["reason"], item["path"][-1]) for item in unknown["owned_unresolved"]] == [
      ("no frame record", "rscrypto_fixture_gcm_kernel")
    ], (reason, unknown)


def test_assembly_continuations_share_one_frame() -> None:
  def call(seal: list[str], avx2: list[str], poly: list[str]) -> dict:
    functions = [
      (0x1000, "prepare", 32, ["1000: callq 0x2000", "1005: retq"]),
      (0x2000, "rscrypto_fixture_seal", None, seal),
      (0x3000, "rscrypto_fixture_seal_avx2", None, avx2),
      (0x4000, "rscrypto_fixture_poly", None, poly),
      (0x5000, "rscrypto::fixture::compiled", 4096, ["5000: retq"]),
    ]
    return whole_call("x86_64-unknown-linux-gnu", functions)

  seal = [
    "2000: pushq %rbx", "2001: pushq %rbp", "2002: subq $0x20, %rsp", "2006: callq 0x4000",
    "200b: testq %rdi, %rdi", "200e: je 0x3000", "2014: addq $0x20, %rsp", "2018: popq %rbp",
    "2019: popq %rbx", "201a: retq", "201b: nopl (%rax,%rax)",
  ]
  avx2 = ["3000: callq 0x4000", "3005: jmp 0x2014"]
  poly = ["4000: rep retq", "4002: nopl (%rax,%rax)"]
  # The AVX2 continuation jumps back into the SSE routine inside the same 48-byte frame: no recursion, one frame.
  # Each call adds the 8-byte return address and the leaf's 0 B frame plus its red zone.
  shared = call(seal, avx2, poly)
  assert shared["owned_bound_bytes"] == 8 + 32 + 8 + 48 + 8 + 128, shared
  assert shared["assumptions"] == [
    "rscrypto_fixture_poly: leaf assembly frame 0 B", "rscrypto_fixture_seal: assembly frame 48 B (2 symbol(s))",
  ], shared
  # A jump back to the decrement can repeat it.
  looping = call(seal, ["3000: callq 0x4000", "3005: jmp 0x2002"], poly)
  assert looping["owned_bound_bytes"] is None, looping
  assert ("no frame record", "rscrypto_fixture_seal") in {
    (item["reason"], item["path"][-1]) for item in looping["owned_unresolved"]
  }, looping
  # Without a final terminator, the helper runs into compiled code whose frame it cannot own.
  falling = call(seal, avx2, ["4000: addq %rax, %rbx"])
  assert falling["owned_bound_bytes"] is None, falling
  assert any(item["path"][-1] == "rscrypto_fixture_poly" for item in falling["owned_unresolved"]), falling
  # An assembly tail jump may keep its frame: the compiled leaf it jumps to, red zone included, is charged
  # below that frame (charging it at the region's entry would give 4,272 B).
  tail = call(seal[:-2] + ["201a: jmp 0x5000"], avx2, poly)
  assert tail["owned_bound_bytes"] == 8 + 32 + 8 + 48 + 4096 + 128, tail


def test_aarch64_labels_split_one_routine() -> None:
  def call(entry: str, ready: list[str]) -> dict:
    functions = [
      (0x1000, "prepare", 32, [f"1000: bl {entry}", "1004: ret"]),
      (0x2000, "rscrypto_fixture_seal_aarch64", None, ["2000: mov x6, #0x0", "2004: b 0x2010", "2008: nop"]),
      (0x200c, "rscrypto_fixture_seal_gather_aarch64", None, ["200c: mov x6, x1"]),
      (0x2010, "Lseal_entry", None, [
        "2010: stp x29, x30, [sp, #-0x70]!", "2014: cbz x19, 0x2020", "2018: ldp x1, x20, [x19], #0x10",
      ]),
      (0x2020, "Lseal_input_ready", None, ready),
      # A constant table after the final `ret` is data, not a path into the compiled code that follows.
      (0x3000, "Lpoly_hash_ad_internal", None, [
        "3000: ret", "3004: 01 00 00 80 \t.word\t0x80000001", "3008: 00 00 00 00 \t.word\t0x00000000",
      ]),
      (0x3010, "rscrypto::fixture::compiled", 16, ["3010: ret"]),
    ]
    return whole_call("aarch64-unknown-linux-gnu", functions)

  ready = ["2020: bl 0x3000", "2024: ldp x29, x30, [sp], #0x70", "2028: ret"]
  # Both entries reach one 112-byte frame: one through `b`, one by running off its end.
  for entry in ("0x2000", "0x200c"):
    shared = call(entry, ready)
    assert shared["owned_bound_bytes"] == 32 + 0x70, (entry, shared)
  looping = call("0x2000", ["2020: bl 0x3000", "2024: cbnz x0, 0x2010", "2028: ret"])
  assert looping["owned_bound_bytes"] is None, looping
  assert ("no frame record", "rscrypto_fixture_seal_aarch64") in {
    (item["reason"], item["path"][-1]) for item in looping["owned_unresolved"]
  }, looping


def elf_image(base: int, data: bytes, big_endian: bool = False) -> ElfImage:
  """A minimal ELF64 image with one loadable segment holding `data` at `base`."""
  order = ">" if big_endian else "<"
  header = bytearray(64)
  header[:6] = b"\x7fELF" + bytes([2, 2 if big_endian else 1])
  struct.pack_into(order + "Q", header, 0x20, 64)
  struct.pack_into(order + "HH", header, 0x36, 56, 1)
  segment = struct.pack(order + "IIQQQQQQ", 1, 4, 120, base, base, len(data), len(data), 0x1000)
  return ElfImage(bytes(header) + segment + data)


def table(base: int, targets: list[int], big_endian: bool = False) -> ElfImage:
  """A relative jump table at `base`: each entry is its destination minus `base`."""
  order = ">" if big_endian else "<"
  return elf_image(base, b"".join(struct.pack(order + "i", target - base) for target in targets), big_endian)


def jump_table_call(target: str, functions: list[tuple[int, str, int | None, list[str]]], image: ElfImage) -> dict:
  linked = program(target, functions)
  resolve_jump_tables(linked, image)
  return review_preparation(linked, ["prepare"])["whole_calls"][0]


X86_JUMP_TABLE = [
  "2000: cmpq $0x2, %rax", "2004: ja 0x2030", "200a: leaq 0x6ff0(%rip), %rcx # 0x9000",
  "2011: movslq (%rcx,%rax,4), %rax", "2015: addq %rcx, %rax", "2018: movq %r8, 0x28(%rsp)",
  "201d: jmpq *%rax", "2020: callq 0x3000", "2025: retq", "2028: retq", "2030: retq",
]


def test_x86_checked_jump_tables_stay_inside_their_function() -> None:
  def call(body: list[str], targets: list[int] = [0x2020, 0x2028, 0x2030]) -> dict:
    functions = [
      (0x1000, "prepare", 32, ["1000: callq 0x2000", "1005: retq"]),
      (0x2000, "rscrypto::fixture::pack", 64, body),
      (0x3000, "rscrypto::fixture::leaf", 100, ["3000: retq"]),
    ]
    return jump_table_call("x86_64-unknown-linux-gnu", functions, table(0x9000, targets))

  # Every entry of indexes 0 through 2 lands in the function, whose frame and call edge already count.
  resolved = call(X86_JUMP_TABLE)
  assert resolved["owned_bound_bytes"] == 8 + 32 + 8 + 64 + 8 + 100 + 128, resolved
  assert "rscrypto::fixture::pack: jump table at 0x201d, 3 entries inside the function" in resolved["assumptions"]
  refused = {
    "an entry leaves the function": (X86_JUMP_TABLE, [0x2020, 0x3000, 0x2030]),
    "the table is shorter than the bound": (X86_JUMP_TABLE, [0x2020, 0x2028]),
    "no bound check precedes the table": ([line.replace("cmpq $0x2", "testq %rax,") for line in X86_JUMP_TABLE], None),
    "a branch enters past the check": ([line.replace("2028: retq", "2028: jmp 0x2011") for line in X86_JUMP_TABLE], None),
    "a register is rewritten": ([line.replace("movq %r8, 0x28(%rsp)", "movq %rdx, %rax") for line in X86_JUMP_TABLE], None),
    "the table is outside the image": ([line.replace("# 0x9000", "# 0xa000") for line in X86_JUMP_TABLE], None),
  }
  for reason, (body, targets) in refused.items():
    unknown = call(body, targets) if targets is not None else call(body)
    assert unknown["owned_bound_bytes"] is None, (reason, unknown)
    assert [(item["reason"], item["address"]) for item in unknown["owned_unresolved"]] == [("indirect", "0x201d")], (
      reason, unknown)


def test_s390x_and_riscv_checked_jump_tables() -> None:
  s390x = [
    (0x1000, "prepare", 160, ["1000: brasl %r14, 0x2000", "1006: br %r14"]),
    (0x2000, "rscrypto::fixture::pack", 160, [
      "2000: clgijh %r1, 1, 0x2040", "2006: sllg %r1, %r1, 2", "200c: larl %r4, 0x9000",
      "2012: agf %r4, 0(%r1,%r4)", "2018: stg %r6, 184(%r15)", "201e: br %r4",
      "2020: br %r14", "2030: br %r14", "2040: br %r14",
    ]),
  ]
  call = jump_table_call("s390x-unknown-linux-gnu", s390x, table(0x9000, [0x2020, 0x2030], big_endian=True))
  assert call["owned_bound_bytes"] == 160 + 160, call
  # The `li` that sets the limit may come before an unrelated write; `auipc` + `addi` form the table base.
  riscv = [
    "2000: li a5, 0x1", "2002: addi a4, a1, -0x3", "2006: bltu a5, a4, 0x2040", "200a: auipc a5, 0x7",
    "200e: slli a4, a4, 0x2", "2010: addi a5, a5, -0xa", "2014: add a4, a4, a5", "2016: lw a4, 0x0(a4)",
    "2018: add a4, a4, a5", "201a: jr a4", "2020: ret", "2030: ret", "2040: ret",
  ]

  def riscv_call(body: list[str]) -> dict:
    functions = [(0x1000, "prepare", 16, ["1000: jal 0x2000 <pack>", "1004: ret"]), (0x2000, "rscrypto::fixture::pack", 32, body)]
    return jump_table_call("riscv64gc-unknown-linux-gnu", functions, table(0x9000, [0x2020, 0x2030]))

  assert riscv_call(riscv)["owned_bound_bytes"] == 16 + 32, riscv_call(riscv)
  # A later write to the limit register leaves the index unchecked.
  clobbered = riscv_call([line.replace("2002: addi a4, a1, -0x3", "2002: mv a5, a1") for line in riscv])
  assert [item["reason"] for item in clobbered["owned_unresolved"]] == ["indirect"], clobbered


def budget_report(target: str, portable: bool, owned: dict[str, int | None]) -> dict:
  calls = [{"root": f"rscrypto_frame_review::{name}", "owned_bound_bytes": value} for name, value in owned.items()]
  return {"target": target, "portable": portable, "whole_calls": calls}


def test_budgets_bound_each_root(directory: Path) -> None:
  # 12.5% above the bound, rounded up to a whole KiB, never below 1 KiB.
  assert [stack_budget(owned) for owned in (0, 8192, 8193, 59_112)] == [1024, 9216, 10_240, 66_560]
  budgets = {"x86_64-unknown-linux-gnu": {"native": {"seal": 9216, "open": 9216, "removed": 1024}}}
  report = budget_report("x86_64-unknown-linux-gnu", False, {"seal": 9216, "open": 9217})
  assert apply_budgets(report, budgets, "native", complete=False) == ["open: owned bound 9,217 B exceeds its 9,216 B budget"]
  assert [call["within_owned_budget"] for call in report["whole_calls"]] == [True, False], report
  assert apply_budgets(report, budgets, "native", complete=True)[-1] == "removed: budget for a root the review no longer has"
  missing = apply_budgets(budget_report("x86_64-unknown-linux-gnu", True, {"seal": 100, "sign": None}), budgets, "portable", False)
  assert missing == ["seal: no budget", "sign: owned bound unresolved"], missing
  # With allowances, a root also needs a whole-call bound within the worker cap.
  whole = budget_report("x86_64-unknown-linux-gnu", False, {"seal": 9216, "open": 9216})
  whole["whole_calls"][0].update(whole_call_bound_bytes=None, missing_allowances=["memcpy"])
  whole["whole_calls"][1].update(whole_call_bound_bytes=8 * 1024 * 1024 + 1)
  assert apply_budgets(whole, budgets, "native", False) == [], whole
  assert apply_budgets(whole, budgets, "native", False, whole_calls=True) == [
    "seal: no whole-call bound; declared edges without an allowance: memcpy",
    "open: whole-call bound 8,388,609 B exceeds the 8 MiB worker cap",
  ], whole

  # Writing replaces a reviewed configuration under `complete` and keeps every other one.
  existing = {"x86_64-unknown-linux-gnu": {"native": {"stale": 1024}, "portable": {"kept": 2048}}}
  written = updated_budgets(existing, [budget_report("x86_64-unknown-linux-gnu", False, {"seal": 8193})], complete=True)
  assert written == {"x86_64-unknown-linux-gnu": {"native": {"seal": 10_240}, "portable": {"kept": 2048}}}, written
  try:
    updated_budgets({}, [budget_report("x86_64-unknown-linux-gnu", False, {"sign": None})], complete=True)
  except SystemExit:
    pass
  else:
    raise AssertionError("an unresolved bound must not receive a budget")

  path = directory / "budgets.json"
  path.write_text(json.dumps({"targets": written}))
  assert load_budgets(path) == written
  for invalid in ({"targets": {"x86_64-apple-darwin": {}}}, {"targets": {"x86_64-unknown-linux-gnu": {"fast": {}}}},
                  {"targets": {"x86_64-unknown-linux-gnu": {"native": {"seal": 0}}}}, {"budgets": {}}):
    path.write_text(json.dumps(invalid))
    try:
      load_budgets(path)
    except SystemExit:
      pass
    else:
      raise AssertionError(f"invalid budgets accepted: {invalid}")


def test_s390x_unlabelled_plt_keeps_allocator_edges_separate() -> None:
  functions = [
    (0x1000, "prepare", 160, ["1000: brasl %r14, 0x2020", "1006: br %r14"]),
    (0x2000, ".plt", None, [
      "2000: br %r1", "2020: larl %r1, 0x8000", "2026: lg %r1, 0(%r1)", "202c: br %r1",
      "2040: larl %r1, 0x8008", "2046: lg %r1, 0(%r1)", "204c: br %r1",
    ]),
  ]
  dynamic = "8000 R_390_JMP_SLOT calloc\n8008 R_390_JMP_SLOT free"
  call = review_preparation(program("s390x-unknown-linux-gnu", functions, dynamic), ["prepare"])["whole_calls"][0]
  assert call["whole_call_bound_bytes"] is None, call
  assert [item["path"][-1] for item in call["unresolved"]] == ["calloc"], call


def test_riscv_call_clobbers_an_auipc_temporary() -> None:
  functions = [
    (0x1000, "prepare", 32, ["1000: auipc t0, 0x1", "1004: jal 0x3000", "1008: jalr t0", "100c: ret"]),
    (0x2000, "stale_target", 16, ["2000: ret"]),
    (0x3000, "clobbers_t0", 16, ["3000: li t0, 1", "3004: ret"]),
  ]
  call = review_preparation(program("riscv64gc-unknown-linux-gnu", functions), ["prepare"])["whole_calls"][0]
  assert call["whole_call_bound_bytes"] is None and call["unresolved"][0]["reason"] == "indirect", call


ALLOCATOR_IMPORTS = "8000 R_X86_64_GLOB_DAT calloc\n8008 R_X86_64_GLOB_DAT memcpy"


def allocating_preparation() -> list[tuple[int, str, int | None, list[str]]]:
  # prepare (32 B) calls calloc and memcpy itself, and memcpy again below helper (64 B).
  return [
    (0x1000, "prepare", 32, [
      "1000: callq *0x7000(%rip) # 0x8000", "1006: callq 0x2000", "100b: callq *0x7000(%rip) # 0x8008", "1011: retq",
    ]),
    (0x2000, "helper", 64, ["2000: callq *0x6000(%rip) # 0x8008", "2006: retq"]),
  ]


def edges_by_name(call: dict) -> dict[str, dict]:
  return {edge["name"]: edge for edge in call["declared_edges"]}


def test_declared_libc_edges_charge_only_the_call_site() -> None:
  call = review_preparation(program("x86_64-unknown-linux-gnu", allocating_preparation(), ALLOCATOR_IMPORTS), ["prepare"])["whole_calls"][0]
  # Root return slot 8 + prepare 32 + call slot 8 + helper 64 + call slot 8; no callee frame or red zone.
  assert call["owned_bound_bytes"] == 120, call
  edges = edges_by_name(call)
  assert set(edges) == {"calloc", "memcpy"} and {edge["kind"] for edge in edges.values()} == {"libc"}, call
  assert edges["calloc"]["max_entry_depth_bytes"] == 8 + 32 + 8, edges
  # memcpy is entered at 48 B from prepare and 120 B below helper; the deeper entry is its depth.
  assert edges["memcpy"]["max_entry_depth_bytes"] == 120, edges
  assert [(site["caller"], site["address"]) for site in edges["memcpy"]["call_sites"]] == [
    ("helper", "0x2000"), ("prepare", "0x100b")
  ], edges
  # The whole-call bound stays null until both allowances are measured.
  assert call["whole_call_bound_bytes"] is None and call["missing_allowances"] == ["calloc", "memcpy"], call
  assert not call["passed"] and {item["path"][-1] for item in call["unresolved"]} == {"calloc", "memcpy"}, call


def test_tail_calls_enter_declared_edges_at_the_callers_entry() -> None:
  aarch64 = [
    (0x1000, "prepare", 48, ["1000: bl 0x2000", "1004: add sp, sp, #0x30", "1008: b 0x9000 <free@plt>"]),
    (0x2000, "helper", 16, ["2000: b 0x9010 <calloc@plt>"]),
    (0x9000, "free@plt", None, ["9000: br x17"]),
    (0x9010, "calloc@plt", None, ["9010: br x17"]),
  ]
  call = review_preparation(program("aarch64-unknown-linux-gnu", aarch64), ["prepare"])["whole_calls"][0]
  edges = edges_by_name(call)
  # prepare released its frame before branching to free; helper tail-branches to calloc.
  assert edges["free"]["max_entry_depth_bytes"] == 0, edges
  assert edges["calloc"]["max_entry_depth_bytes"] == 48, edges
  assert call["owned_bound_bytes"] == 48 + 16 and not call["owned_unresolved"], call
  x86 = [(0x1000, "prepare", 24, ["1000: jmpq *0x7000(%rip) # 0x8008"])]
  call = review_preparation(program("x86_64-unknown-linux-gnu", x86, "8008 R_X86_64_GLOB_DAT free"), ["prepare"])["whole_calls"][0]
  assert edges_by_name(call)["free"]["max_entry_depth_bytes"] == 8, call
  # Without a call, prepare is a leaf and keeps its red zone, as before.
  assert call["owned_bound_bytes"] == 8 + 24 + 128, call


def test_undeclared_imports_keep_the_owned_bound_unresolved() -> None:
  functions = [(0x1000, "prepare", 32, ["1000: callq *0x7000(%rip) # 0x8000", "1006: callq *0x7000(%rip) # 0x8008", "100c: retq"])]
  for dynamic, unresolved in (
    ("8000 R_X86_64_GLOB_DAT realloc\n8008 R_X86_64_GLOB_DAT calloc", "realloc"),
    ("8000 R_X86_64_GLOB_DAT dlsym\n8008 R_X86_64_GLOB_DAT calloc", "dlsym"),
    ("8000 R_X86_64_IRELATIVE *ABS*+0x2000\n8008 R_X86_64_GLOB_DAT calloc", "IFUNC resolver *ABS*+0x2000 (selected implementation unresolved)"),
  ):
    call = review_preparation(program("x86_64-unknown-linux-gnu", functions, dynamic), ["prepare"])["whole_calls"][0]
    assert call["owned_bound_bytes"] is None and call["whole_call_bound_bytes"] is None, call
    assert [(item["reason"], item["path"][-1]) for item in call["owned_unresolved"]] == [("external", unresolved)], call
    # The declared edge is still reported, but its depth is not a bound while the owned graph is unresolved.
    calloc = edges_by_name(call)["calloc"]
    assert calloc["max_entry_depth_bytes"] is None and calloc["recorded_entry_depth_bytes"] == 48, calloc
    assert any("unresolved owned call paths" in problem for problem in call["problems"]), call


def test_detection_has_no_declared_edge() -> None:
  # Detection is owned code. A std environment read below it, such as the removed
  # `RSCRYPTO_FORCE_AVX512` read, is neither cut nor declared: its subgraph stays
  # owned, so its blockers keep the root unbounded.
  functions = [
    (0x1000, "prepare", 32, ["1000: callq 0x2000", "1005: retq"]),
    (0x2000, "rscrypto::auth::mldsa::poly::Poly::ntt", 96, ["2000: callq 0x3000", "2005: retq"]),
    (0x3000, "rscrypto::platform::detect::hybrid_avx512_override", 48, ["3000: callq 0x4000", "3005: retq"]),
    (0x4000, "<std::sys::sync::rwlock::futex::RwLock>::read_contended", None, ["4000: callq *%r14", "4003: callq *0x4000(%rip) # 0x8000"]),
  ]
  dynamic = "8000 R_X86_64_GLOB_DAT syscall"
  call = review_preparation(program("x86_64-unknown-linux-gnu", functions, dynamic), ["prepare"])["whole_calls"][0]
  assert call["owned_bound_bytes"] is None and not call["declared_edges"], call
  assert {item["reason"] for item in call["owned_unresolved"]} == {"no frame record", "indirect", "external"}, call
  assert call["whole_call_bound_bytes"] is None, call


def test_getauxval_is_a_declared_libc_edge() -> None:
  functions = [
    (0x1000, "prepare", 64, ["1000: bl 0x3000", "1004: ret"]),
    (0x3000, "rscrypto::platform::detect::linux_hwcaps", 32, ["3000: bl 0x9000 <getauxval@plt>", "3004: ret"]),
    (0x9000, "getauxval@plt", None, ["9000: br x17"]),
  ]
  call = review_preparation(program("aarch64-unknown-linux-gnu", functions), ["prepare"])["whole_calls"][0]
  # Owned detection stays in the owned bound; only the libc read is an allowance.
  assert call["owned_bound_bytes"] == 64 + 32 and not call["owned_unresolved"], call
  getauxval = edges_by_name(call)["getauxval"]
  assert getauxval["kind"] == "libc" and getauxval["max_entry_depth_bytes"] == 64 + 32, getauxval
  assert call["missing_allowances"] == ["getauxval"], call


def test_owned_indirection_and_recursion_keep_the_owned_bound_null() -> None:
  for body in (["2000: callq *%rax", "2002: retq"], ["2000: callq 0x1000", "2005: retq"]):
    functions = [
      (0x1000, "prepare", 32, ["1000: callq *0x7000(%rip) # 0x8000", "1006: callq 0x2000", "100b: retq"]),
      (0x2000, "helper", 16, body),
    ]
    call = review_preparation(program("x86_64-unknown-linux-gnu", functions, ALLOCATOR_IMPORTS), ["prepare"])["whole_calls"][0]
    assert call["owned_bound_bytes"] is None and call["whole_call_bound_bytes"] is None, call
    assert [item["reason"] for item in call["owned_unresolved"]] in (["indirect"], ["recursion"]), call
    assert edges_by_name(call)["calloc"]["max_entry_depth_bytes"] is None, call
  # Allowances cannot compose an unresolved owned graph.
  allowances = {"libc": "fixture libc", "edges": {"calloc": 64}}
  call = review_preparation(program("x86_64-unknown-linux-gnu", functions, ALLOCATOR_IMPORTS), ["prepare"], allowances)["whole_calls"][0]
  assert call["whole_call_bound_bytes"] is None and not call["passed"], call
  missing_frame = [
    (0x1000, "prepare", 32, ["1000: callq 0x2000", "1005: retq"]),
    (0x2000, "helper", None, ["2000: subq %rdi, %rsp", "2003: retq"]),
  ]
  call = review_preparation(program("x86_64-unknown-linux-gnu", missing_frame), ["prepare"])["whole_calls"][0]
  assert call["owned_bound_bytes"] is None and call["owned_unresolved"][0]["reason"] == "no frame record", call


def test_allowances_compose_the_whole_call_bound(directory: Path) -> None:
  functions = allocating_preparation()

  def compose(edges: dict[str, int]) -> dict:
    allowances = {"libc": "fixture libc 1.0", "edges": edges}
    return review_preparation(program("x86_64-unknown-linux-gnu", functions, ALLOCATOR_IMPORTS), ["prepare"], allowances)

  # max(owned 120, calloc 48 + 1000, memcpy 120 + 10)
  report = compose({"calloc": 1000, "memcpy": 10})
  call = report["whole_calls"][0]
  assert call["whole_call_bound_bytes"] == 1048 and call["passed"] and report["passed"], call
  assert call["whole_call_bound_basis"] == "owned bound composed with measured declared-edge allowances", call
  # A small allowance never lowers the bound below the owned frames.
  assert compose({"calloc": 0, "memcpy": 0})["whole_calls"][0]["whole_call_bound_bytes"] == 120
  assert compose({"calloc": 0, "memcpy": 900})["whole_calls"][0]["whole_call_bound_bytes"] == 1020
  partial = compose({"calloc": 1000})["whole_calls"][0]
  assert partial["whole_call_bound_bytes"] is None and partial["missing_allowances"] == ["memcpy"], partial
  oversized = compose({"calloc": 8 * 1024 * 1024, "memcpy": 0})["whole_calls"][0]
  assert oversized["whole_call_bound_bytes"] == 8 * 1024 * 1024 + 48 and not oversized["passed"], oversized

  path = directory / "allowances.json"
  for invalid in (
    {"x86_64-unknown-linux-gnu": {"edges": {"calloc": 1}}},
    {"x86_64-unknown-linux-gnu": {"libc": " ", "edges": {"calloc": 1}}},
    {"x86_64-unknown-linux-gnu": {"libc": "glibc", "edges": {"realloc": 1}}},
    {"x86_64-unknown-linux-gnu": {"libc": "glibc", "edges": {"calloc": -1}}},
    {"x86_64-unknown-linux-gnu": {"libc": "glibc", "edges": {"calloc": True}}},
    {"x86_64-unknown-linux-gnu": {"libc": "glibc", "edges": {}, "measured": "yes"}},
    {"x86_64-apple-darwin": {"libc": "libSystem", "edges": {}}},
    # The environment-read edge was removed with `RSCRYPTO_FORCE_AVX512`.
    {"x86_64-unknown-linux-gnu": {"libc": "glibc", "edges": {"rscrypto-force-avx512": 1}}},
  ):
    path.write_text(json.dumps(invalid))
    try:
      load_allowances(path)
    except SystemExit:
      pass
    else:
      raise AssertionError(f"invalid allowances accepted: {invalid}")
  valid = {"x86_64-unknown-linux-gnu": {"libc": "glibc 2.39", "edges": {"calloc": 1, "getauxval": 2}}}
  path.write_text(json.dumps(valid))
  assert load_allowances(path) == valid


def test_whole_call_groups_select_their_roots() -> None:
  assert whole_call_roots("mldsa-preparation") == PREPARATION_ROOTS
  every = whole_call_roots("all")
  assert len(every) == len(set(every)) == sum(map(len, WHOLE_CALL_GROUPS.values())), every
  assert whole_call_roots("mlkem,mlkem") == WHOLE_CALL_GROUPS["mlkem"]
  for invalid in ("", "mldsa", "mlkem,aead"):
    try:
      whole_call_roots(invalid)
    except ValueError:
      pass
    else:
      raise AssertionError(f"accepted whole-call groups {invalid!r}")


def main() -> None:
  with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    test_s390x_detection_is_reported_not_dropped(directory)
    test_detection_free_worker_passes(directory)
    test_depth_beyond_scrub_fails(directory)
    test_caller_without_scrub_and_recursion_fail(directory)
    test_power_local_entries_returns_and_red_zone(directory)
    test_riscv_auipc_pairs_labels_and_returns(directory)
    test_x86_got_imports_call_slots_and_red_zone(directory)
    test_x86_register_held_imports(directory)
    test_aarch64_leaf_assembly_frames(directory)
    test_aarch64_tail_scrub_counts_the_caller_frame(directory)
    test_panic_exits_are_listed_and_not_counted(directory)
    test_target_and_portable_scoped_boundaries_are_skipped(directory)
  test_record_parsers()
  test_preparation_never_turns_unknown_frames_into_a_bound()
  test_preparation_resolves_relative_calls_but_not_ifuncs_or_memory_imports()
  test_got_value_cannot_cross_an_unproved_control_flow_join()
  test_branch_beyond_last_function_is_not_a_recursive_call()
  test_assembly_loop_cannot_hide_repeated_stack_growth()
  test_x86_save_slot_assembly_frame()
  test_assembly_continuations_share_one_frame()
  test_aarch64_labels_split_one_routine()
  test_x86_checked_jump_tables_stay_inside_their_function()
  test_s390x_and_riscv_checked_jump_tables()
  test_s390x_unlabelled_plt_keeps_allocator_edges_separate()
  test_riscv_call_clobbers_an_auipc_temporary()
  test_declared_libc_edges_charge_only_the_call_site()
  test_tail_calls_enter_declared_edges_at_the_callers_entry()
  test_undeclared_imports_keep_the_owned_bound_unresolved()
  test_detection_has_no_declared_edge()
  test_getauxval_is_a_declared_libc_edge()
  test_whole_call_groups_select_their_roots()
  test_owned_indirection_and_recursion_keep_the_owned_bound_null()
  with tempfile.TemporaryDirectory() as temporary:
    test_allowances_compose_the_whole_call_bound(Path(temporary))
  with tempfile.TemporaryDirectory() as temporary:
    test_budgets_bound_each_root(Path(temporary))
  print("frames tests passed")


if __name__ == "__main__":
  main()
