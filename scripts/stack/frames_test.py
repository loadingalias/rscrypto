#!/usr/bin/env python3
"""Regression tests for the linked-frame review, using synthetic disassembly."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from frames import Boundary, parse_disassembly, parse_dynamic_symbols, parse_stack_sizes, parse_symbols, review

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
      "<std::sync::once_lock::OnceLock<rscrypto::platform::detect::Detected>>::initialize",
      192,
      ["    5000:      \tbrasl\t%r14, 0x6000", "    5006:      \tbr\t%r14"],
    ),
    (0x6000, "<std::sys::sync::once::futex::Once>::call", None, ["    6000:      \tbasr\t%r14, %r1"]),
  ]
  report = review(program("s390x-unknown-linux-gnu", functions), [boundary(directory)])
  worker = worker_report(report)
  assert not report["passed"]
  assert worker["depth"] == 816 + 400, worker
  assert len(worker["detection"]) == 1 and "platform::detect" in worker["detection"][0][-1]
  assert [item["reason"] for item in worker["unbounded"]] == ["no frame record"], worker["unbounded"]
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
  assert worker["unbounded"] == [{"reason": "external", "path": [WORKER, "getauxval"]}], worker
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
    test_panic_exits_are_listed_and_not_counted(directory)
  test_record_parsers()
  print("frames tests passed")


if __name__ == "__main__":
  main()
