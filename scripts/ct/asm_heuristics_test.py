#!/usr/bin/env -S uv run python
"""Regression tests for target-specific CT disassembly parsing."""

from pathlib import Path
from tempfile import TemporaryDirectory

from asm_heuristics import (
  prepare_linked_image,
  decode_lines,
  row_successors,
  FunctionBody,
  function_identity,
  instruction_identities,
  all_direct_callees,
  apply_constant_conditions,
  apply_public_inputs,
  apply_select_branch_enforcement,
  apply_waivers,
  attach_condition_identities,
  build_call_graph,
  control_errors,
  direct_callees,
  effect_errors,
  linked_closure,
  is_call_relocation,
  is_conditional_branch,
  is_riscv_conditional_branch,
  is_rsa_inverse_scope,
  instruction_identity,
  parse_disassembly,
  reachable_closure,
  scan_symbol,
  unreachable_symbol_errors,
  validate_public_inputs,
)
from linked_image import MERGED, LinkedImage, Section, Slot, assembly_merge_units, merge_unit_members
from symbolize_linked_binary import demangle_table


def branch_kinds(target: str, symbol: str, lines: list[str], primitive_ids: list[str]) -> list[tuple[str, str]]:
  body = FunctionBody(symbol=symbol, path=Path("fixture.o.disasm.txt"), address=0,
                      lines=list(enumerate(lines, start=1)))
  findings = scan_symbol(target, symbol, body, {symbol}, scope="ct_intended_call_closure",
                         primitive_ids=primitive_ids, roots=[symbol])
  apply_select_branch_enforcement(findings, {"kx.p256_ecdh"})
  return [(row["kind"], row["severity"]) for row in findings
          if row["kind"] in {"conditional_branch", "select_branch"}]


def multiply_kinds(target: str, symbol: str, line: str, primitive_ids: list[str]) -> list[tuple[str, str]]:
  body = FunctionBody(symbol=symbol, path=Path("fixture.o.disasm.txt"), address=0, lines=[(1, line)])
  findings = scan_symbol(target, symbol, body, {symbol}, scope="ct_intended_call_closure",
                         primitive_ids=primitive_ids, roots=[symbol])
  return [(row["kind"], row["severity"]) for row in findings if "multiply" in row["kind"]]


def immediate_symbol_regressions(body: FunctionBody, known: set[str]) -> None:
  """llvm-objdump names the symbol that contains an absolute immediate's value.

  Moving unrelated code renames that symbol without changing the instruction; the 2026-11-04 RISC-V
  RSA waivers went stale only because `lui a1, 0x33333 <Projective::add+0x841>` became `+0x869`.
  The value must stay bound, and PC-relative names, which identify real targets, must stay.
  """
  cases = {
    "riscv64gc-unknown-linux-gnu": (
      "1000: 333335b7\tlui\ta1, 0x33333 <{}>",
      "1000: 555555b7\tlui\ta1, 0x55555 <{}>",
      ["1000: 000080e7\tjalr\tra <{}>"],
    ),
    "aarch64-unknown-linux-gnu": (
      "1000: 92403c08\tand\tx8, x0, #0xffff <{}>",
      "1000: 92403808\tand\tx8, x0, #0x7fff <{}>",
      ["1000: b00009c8\tadrp\tx8, 0x140000 <{}>", "1000: 82 80 00 00\t.word\t0x8082 <{}>"],
    ),
    "x86_64-unknown-linux-gnu": (
      "1000: 48 81 e3 ff ff 0f 00\tandq\t$0xfffff, %rbx  # imm = 0xfffff <{}>",
      "1000: 48 81 e3 ff ff 07 00\tandq\t$0x7ffff, %rbx  # imm = 0x7ffff <{}>",
      ["1000: 48 8d 3d 59 ef 07 00\tleaq\t0x7ef59(%rip), %rdi  # 0x7ff60 <{}>"],
    ),
  }
  for target, (immediate, other_value, pc_relative) in cases.items():
    identity = lambda line: instruction_identity(target, body, line, known)
    assert identity(immediate.format("callee+0x841")) == identity(immediate.format("other+0x869")), target
    assert "<" not in identity(immediate.format("callee")), target
    assert identity(immediate.format("callee")) != identity(other_value.format("callee")), target
    for line in pc_relative:
      assert identity(line.format("callee")) != identity(line.format("other")), (target, line)
  # The shifted AArch64 form keeps the computed value objdump prints after `// =`.
  target = "aarch64-unknown-linux-gnu"
  shifted = "1000: 714142ff\tcmp\tw23, #0x50, lsl #12  // =0x50000 <{}>"
  moved = instruction_identity(target, body, shifted.format("other"), known)
  assert instruction_identity(target, body, shifted.format("callee"), known) == moved
  assert moved.endswith("// =0x50000"), moved
  # Only `lui` loads an absolute RISC-V immediate; other annotated forms stay exact.
  target = "riscv64gc-unknown-linux-gnu"
  added = "1000: 00a50513\taddi\ta0, a0, 0xa <{}>"
  assert instruction_identity(target, body, added.format("callee"), known) != \
    instruction_identity(target, body, added.format("other"), known)


def linker_stub_regressions() -> None:
  """Calls to PLT entries and POWER call stubs bind the import, not where the linker placed it.

  The 2026-11-04 POWER RSA waivers went stale when `bl 0x83c0 <0000004f.plt_call.free@@GLIBC_2.17>`
  moved to 0x8360: the lookup stripped the symbol version, so the call stayed byte-exact.
  """
  body = FunctionBody("caller", Path("linked.binary.disasm.txt"), 0x1000, [])
  stubs = {"caller", "memcpy@plt", "memset@plt", "0000004f.plt_call.free@@GLIBC_2.17",
           "00000037.plt_call.free@@GLIBC_2.17", "0000004f.plt_call.memset@@GLIBC_2.17"}
  power = lambda line: instruction_identity("powerpc64le-unknown-linux-gnu", body, line, stubs)
  free = power("1000: c5 e5 f7 4b\tbl 0x83c0 <0000004f.plt_call.free@@GLIBC_2.17>")
  assert free == power("1000: e5 ef f7 4b\tbl 0x8360 <0000004f.plt_call.free@@GLIBC_2.17>")
  assert free == power("1000: 85 ef f7 4b\tbl 0x8300 <00000037.plt_call.free@@GLIBC_2.17>")
  assert free != power("1000: c5 e5 f7 4b\tbl 0x83c0 <0000004f.plt_call.memset@@GLIBC_2.17>")
  assert "0000004f" not in free and "plt_call.free@@GLIBC_2.17" in free, free
  arm = lambda line: instruction_identity("aarch64-unknown-linux-gnu", body, line, stubs)
  memcpy = arm("1000: 94001000\tbl 0x5000 <memcpy@plt>")
  assert memcpy == arm("1000: 94000c00\tbl 0x4000 <memcpy@plt>")
  assert memcpy != arm("1000: 94001000\tbl 0x5000 <memset@plt>")
  # An unknown destination stays byte-exact.
  assert arm("1000: 94001000\tbl 0x5000 <bcmp@plt>") != arm("1000: 94000c00\tbl 0x4000 <bcmp@plt>")


def multiply_regressions() -> None:
  # Owner decisions 2026-10-21: secret ECDSA limbs use the hardware multiplier on riscv64, and the
  # RSA blinding inverse on riscv64 and IBM Z, reviewed like ECDH and the RSA exponentiation. RV32
  # and IBM Z ECDSA stay fixed-work: IBM Z MLGR failed native DudeCT (Constant-Time 39717348652).
  ecdsa = "rscrypto::auth::ecdsa::montgomery_mul"
  riscv = "  10: 02b50533     \tmul\ta0, a0, a1"
  ibm_z = "  10: b9 86 00 24  \tmlgr\t%r2, %r4"
  review = [("variable_latency_multiply_review", "warn")]
  blocking = [("variable_latency_multiply", "fail")]
  assert multiply_kinds("riscv64gc-unknown-linux-gnu", ecdsa, riscv, ["signature.ecdsa_p256"]) == review
  assert multiply_kinds("s390x-unknown-linux-gnu", ecdsa, ibm_z, ["signature.ecdsa_p384"]) == blocking
  assert multiply_kinds("riscv32imac-unknown-none-elf", ecdsa, riscv, ["signature.ecdsa_p256"]) == blocking
  inverse = "rscrypto::auth::rsa::private_modular_inverse_odd_i31"
  assert multiply_kinds("riscv64gc-unknown-linux-gnu", inverse, riscv, ["rsa.private_ops"]) == review
  assert multiply_kinds("s390x-unknown-linux-gnu", inverse, ibm_z, ["rsa.private_ops"]) == review
  assert multiply_kinds("riscv32imac-unknown-none-elf", inverse, riscv, ["rsa.private_ops"]) == blocking


def select_branch_regressions() -> None:
  # riscv64 `Projective::select_signed` before 543fa738: LLVM skipped one
  # store on the entry matching the secret digit (Constant-Time 38072816018).
  select_signed = "<rscrypto::auth::p256_portable::Projective>::select_signed"
  before = [
    f"  f8: 04a48d63     \tbeq\ts1, a0, 0x152 <{select_signed}+0x152>",
    f" 14a: e0aa         \tsd\ta0, 0x40(sp)",
    f" 14c: d811         \tbeqz\ts0, 0x60 <{select_signed}+0x60>",
    f" 14e: fc56         \tsd\ts4, 0x38(sp)",
    f" 150: bf21         \tj\t0x60 <{select_signed}+0x60>",
    " 152: 01c03533     \tsnez\ta0, t3",
  ]
  assert branch_kinds("riscv64gc-unknown-linux-gnu", select_signed, before, ["kx.p256_ecdh"]) == [
    ("conditional_branch", "warn"),
    ("select_branch", "fail"),
  ]
  # The same shape outside an enforced primitive is reported, not blocking.
  assert branch_kinds("riscv64gc-unknown-linux-gnu", select_signed, before, ["kem.mlkem768"]) == [
    ("conditional_branch", "warn"),
    ("select_branch", "warn"),
  ]
  # After the fix only the public table loop edge remains.
  after = [
    f" 13a: e0aa         \tsd\ta0, 0x40(sp)",
    f" 13c: f0a495e3     \tbne\ts1, a0, 0x5e <{select_signed}+0x5e>",
    " 140: 01c03533     \tsnez\ta0, t3",
  ]
  assert branch_kinds("riscv64gc-unknown-linux-gnu", select_signed, after, ["kx.p256_ecdh"]) == [
    ("conditional_branch", "warn"),
  ]

  # s390x prints address-only targets and compare-and-branch forms. This is
  # the P-256 signed-digit correction at 543fa738 (`jh` on the secret window).
  recode = "<rscrypto::auth::p256_ecdh::P256EphemeralSecret>::diffie_hellman"
  assert is_conditional_branch("s390x-unknown-linux-gnu", "cgije")
  assert is_conditional_branch("s390x-unknown-linux-gnu", "clgrjhe")
  assert is_conditional_branch("s390x-unknown-linux-gnu", "jnle")
  assert not is_conditional_branch("s390x-unknown-linux-gnu", "jg")
  assert branch_kinds("s390x-unknown-linux-gnu", recode, [
    " 1c4: a7 84 00 54  \tje\t0x284",
    " 26a: c2 4f 00 00 00 10    \tclfi\t%r4, 16",
    " 274: a7 58 00 e0  \tlhi\t%r5, 224",
    " 278: a7 24 ff a6  \tjh\t0x1c4",
    " 27c: a7 58 00 00  \tlhi\t%r5, 0",
    " 280: a7 f4 ff a2  \tj\t0x1c4",
    " 284: e3 d0 f8 e8 00 24    \tstg\t%r13, 2280(%r15)",
  ], ["kx.p256_ecdh"]) == [("conditional_branch", "warn"), ("select_branch", "fail")]
  # An address operand followed by a relocation names another function.
  assert branch_kinds("s390x-unknown-linux-gnu", recode, [
    "   0: ec 28 00 03 00 7c    \tcgije\t%r2, 0, 0x6",
    "   6: c0 f4 00 00 00 00    \tjg\t0x6",
    "\t\t0000000000000008:  R_390_PC32DBL\trscrypto::other+0x2",
    "   c: a7 f4 00 00  \tj\t0x0",
  ], ["kx.p256_ecdh"]) == [("conditional_branch", "warn")]

  # POWER uses `bt`/`bf` condition-register branches.
  zeroize_tail = "<rscrypto::auth::p256_ecdh::P256EphemeralSecret>::diffie_hellman"
  assert is_conditional_branch("powerpc64le-unknown-linux-gnu", "bt")
  assert is_conditional_branch("powerpc64le-unknown-linux-gnu", "bdnz")
  assert not is_conditional_branch("riscv64gc-unknown-linux-gnu", "bt")
  assert branch_kinds("powerpc64le-unknown-linux-gnu", zeroize_tail, [
    "1354: 05 00 83 98  \tstb 4, 5(3)",
    f"1358: 08 00 82 41  \tbt\t2, 0x1360 <{zeroize_tail}+0x1360>",
    "135c: 06 00 83 98  \tstb 4, 6(3)",
    "1360: ac 04 00 7c  \tsync",
  ], ["kx.p256_ecdh"]) == [("select_branch", "fail")]

  # x86 `jg` is conditional, unlike s390x `jg`. An inner conditional
  # does not make the enclosing branch a straight-line selection diamond.
  symbol = "nested_conditions"
  assert branch_kinds("x86_64-unknown-linux-gnu", symbol, [
    f"  0: 74 0e  \tje 0x10 <{symbol}+0x10>",
    "  2: b8 01 00 00 00  \tmovl $1, %eax",
    f"  7: 7f 0f  \tjg 0x18 <{symbol}+0x18>",
    "  9: c3  \tretq",
    " 10: b8 02 00 00 00  \tmovl $2, %eax",
    f" 15: eb 01  \tjmp 0x18 <{symbol}+0x18>",
    " 18: c3  \tretq",
  ], ["kx.p256_ecdh"]) == [("conditional_branch", "warn"), ("conditional_branch", "warn")]

  # `jrcxz` branches on rcx without reading the flags. The RSA squaring loop uses it as its exit,
  # and the same instruction can also skip a selection arm.
  symbol = "count_loop"
  assert branch_kinds("x86_64-unknown-linux-gnu", symbol, [
    "  0: 48 83 c1 20  \taddq $0x20, %rcx",
    f"  4: e3 03  \tjrcxz 0x9 <{symbol}+0x9>",
    "  6: 90  \tnop",
    f"  7: eb f7  \tjmp 0x0 <{symbol}>",
    "  9: c3  \tretq",
  ], ["kx.p256_ecdh"]) == [("conditional_branch", "warn")]
  symbol = "count_select"
  assert branch_kinds("x86_64-unknown-linux-gnu", symbol, [
    f"  0: e3 05  \tjrcxz 0x7 <{symbol}+0x7>",
    "  2: b8 01 00 00 00  \tmovl $1, %eax",
    "  7: c3  \tretq",
  ], ["kx.p256_ecdh"]) == [("select_branch", "fail")]

  # A two-arm diamond really does select data; a call in either arm
  # requires separate control-flow analysis instead of this shape rule.
  diamond = [
    f"  0: 54000061  \tb.ne 0xc <{symbol}+0xc>",
    "  4: d2800020  \tmov x0, #1",
    f"  8: 14000002  \tb 0x10 <{symbol}+0x10>",
    "  c: d2800040  \tmov x0, #2",
    " 10: d65f03c0  \tret",
  ]
  assert branch_kinds("aarch64-unknown-linux-gnu", symbol, diamond, ["kx.p256_ecdh"]) == [
    ("select_branch", "fail"),
  ]
  diamond[3] = "  c: 94000000  \tbl 0x20 <callee>"
  assert branch_kinds("aarch64-unknown-linux-gnu", symbol, diamond, ["kx.p256_ecdh"]) == [
    ("conditional_branch", "warn"),
  ]


def linked_fixture(
  text: int, lines: list[str], data: dict[int, bytes], *, units: list[tuple[int, int, int | None]] | None = None,
  names: dict[int, str] | None = None, slots: dict[int, Slot] | None = None, toc_base: int | None = None,
) -> tuple[FunctionBody, LinkedImage]:
  """Build one linked function and the read-only data it references."""
  body = FunctionBody("rscrypto::fixture", Path("fixture.binary.disasm.txt"), text, list(enumerate(lines, start=1)))
  image_bytes = bytearray()
  sections = [Section(".text", text, 0x100, None, True, False)]
  for address, contents in sorted(data.items()):
    sections.append(Section(".rodata", address, len(contents), len(image_bytes), False, False))
    image_bytes.extend(contents)
  starts = dict(names or {})
  image = LinkedImage(bytes(image_bytes), sections, True, starts=starts, extents=units or [], slots=slots or {},
                      toc_base=toc_base)
  image.set_functions([(text, text + 0x100, body.symbol)])
  return body, image


def fixture_hash(target: str, fixture: tuple[FunctionBody, LinkedImage]) -> str:
  body, image = fixture
  return function_identity(instruction_identities(target, body, {body.symbol} | {name for *_, name in image.functions},
                                                  image))


DETECT = "rscrypto::platform::detect::detect_uncached"


def public_input_branch(condition: list[str], symbol: str = DETECT, target: str = "x86_64-unknown-linux-gnu",
                        waivers: list[dict] | None = None) -> tuple[dict, list[str]]:
  """The enforced select-shaped branch on `condition`, and the waiver errors.

  The rule classifies detection on x86-64 Linux, with leaves as in `ct.toml`.
  """
  rule = {"target": "x86_64-unknown-linux-gnu", "symbol": DETECT, "source": "src/platform/detect/arch/x86.rs",
          "rationale": "fixture", "leaves": ["cpuid", "call core::core_arch::x86::xsave::_xgetbv"]}
  address = 0x1000 + sum(len(line.split("\t")[0].split(": ")[1].split()) for line in condition)
  lines = [*condition, f"{address:x}: 74 05\tje\t0x{address + 7:x} <{symbol}+0x{address + 7 - 0x1000:x}>",
           f"{address + 2:x}: b8 01 00 00 00\tmovl\t$0x1, %eax", f"{address + 7:x}: c3\tretq"]
  body = FunctionBody(symbol, Path("fixture.binary.disasm.txt"), 0x1000, list(enumerate(lines, start=1)))
  local = {symbol, "core::core_arch::x86::xsave::_xgetbv", "g"}
  findings = scan_symbol(target, symbol, body, local, scope="ct_intended_call_closure",
                         primitive_ids=["kx.p256_ecdh"], roots=["ct_entry_p256_ecdh_agree"])
  apply_select_branch_enforcement(findings, {"kx.p256_ecdh"})
  attach_condition_identities(target, findings, {symbol: body}, "artifacts/fixture.binary.disasm.txt", local, None,
                              {"select_branch"})
  apply_public_inputs(findings, [rule], target)
  errors = apply_waivers(findings, waivers or [], target)
  [branch] = [row for row in findings if row["kind"] == "select_branch"]
  return branch, errors


def public_input_regressions() -> None:
  # A feature test on a CPUID leaf: its only leaf is the CPUID result, which the rule allows.
  cpuid = ["1000: b8 07 00 00 00\tmovl\t$0x7, %eax", "1005: 31 c9\txorl\t%ecx, %ecx", "1007: 0f a2\tcpuid",
           "1009: f6 c3 08\ttestb\t$0x8, %bl"]
  accepted, errors = public_input_branch(cpuid)
  assert accepted["disposition"] == "accepted" and accepted["slice_leaves"] == ["cpuid"] and not errors, accepted
  # XCR0 from core's `_xgetbv` call is an allowed query.
  xgetbv = ["1000: 31 c9\txorl\t%ecx, %ecx",
            "1002: e8 f9 0f 00 00\tcallq\t0x2000 <core::core_arch::x86::xsave::_xgetbv>",
            "1007: a8 06\ttestb\t$0x6, %al"]
  assert public_input_branch(xgetbv)[0]["disposition"] == "accepted"
  # A function argument, writable memory, and an unlisted call each block, so the enforced branch still fails.
  blocked = {
    "entry rdi": ["1000: 40 f6 c7 08\ttestb\t$0x8, %dil"],
    "entry memory": ["1000: f6 05 00 10 00 00 08\ttestb\t$0x8, 0x1000(%rip)"],
    "call g": ["1000: e8 fb 0f 00 00\tcallq\t0x2000 <g>", "1005: a8 08\ttestb\t$0x8, %al"],
  }
  for leaf, condition in blocked.items():
    branch = public_input_branch(condition)[0]
    assert branch["disposition"] == "needs-fix" and branch["public_input_blocked"] == [leaf], branch
  # The rule names one function on one target.
  assert public_input_branch(cpuid, symbol="rscrypto::platform::detect::detect")[0]["disposition"] == "needs-fix"
  assert public_input_branch(cpuid, target="x86_64-pc-windows-msvc")[0]["disposition"] == "needs-fix"
  # A waiver of a branch that the rule accepts waives nothing, so it fails the verdict.
  waiver = {"target": "x86_64-unknown-linux-gnu", "primitive": "kx.p256_ecdh", "symbol": DETECT,
            "kind": "select_branch", "slice_sha256": accepted["slice_sha256"]}
  assert public_input_branch(cpuid, waivers=[waiver])[1] == [
    "asm_waiver[0] matches a finding that a public classification accepts"
  ]
  # Only processor-state instructions, named calls, and numbered system calls can be allowed leaves.
  for leaves in (["entry memory"], ["call"], ["syscall"], ["unknown store"], ["cpuid", "cpuid"]):
    rule = {"target": "t", "symbol": "f", "source": "s", "rationale": "r", "leaves": leaves}
    assert validate_public_inputs([rule]) == [
      "asm_public_input[0] leaves must be unique processor-state instructions, `call <symbol>`, or `syscall <number>`"
    ], leaves


def constant_branches(lines: list[str], waivers: list[dict] | None = None) -> tuple[list[dict], list[str]]:
  """The enforced select-shaped branches of an x86-64 fixture, and the waiver errors."""
  target, symbol = "x86_64-unknown-linux-gnu", "rscrypto::fixture"
  body = FunctionBody(symbol, Path("fixture.binary.disasm.txt"), 0x1000, list(enumerate(lines, start=1)))
  findings = scan_symbol(target, symbol, body, {symbol}, scope="ct_intended_call_closure",
                         primitive_ids=["kx.p256_ecdh"], roots=["ct_entry_p256_ecdh_agree"])
  apply_select_branch_enforcement(findings, {"kx.p256_ecdh"})
  attach_condition_identities(target, findings, {symbol: body}, "artifacts/fixture.binary.disasm.txt", {symbol}, None,
                              {"select_branch"})
  apply_constant_conditions(findings)
  errors = apply_waivers(findings, waivers or [], target)
  return [row for row in findings if row["kind"] == "select_branch"], errors


def constant_condition_regressions() -> None:
  # A loop counter from an immediate, tested inside its loop: the branch decides from constants alone.
  loop = ["1000: b9 00 00 00 00\tmovl\t$0x0, %ecx", "1005: 83 c1 01\taddl\t$0x1, %ecx",
          "1008: 83 f9 32\tcmpl\t$0x32, %ecx", "100b: 74 05\tje\t0x1012 <rscrypto::fixture+0x12>",
          "100d: b8 01 00 00 00\tmovl\t$0x1, %eax", "1012: 83 f9 34\tcmpl\t$0x34, %ecx",
          "1015: 75 ee\tjne\t0x1005 <rscrypto::fixture+0x5>", "1017: c3\tretq"]
  [accepted], errors = constant_branches(loop)
  assert accepted["slice_constant"] and accepted["disposition"] == "accepted" and not errors, accepted
  # A waiver of that branch waives nothing, so it fails the verdict.
  waiver = {"target": "x86_64-unknown-linux-gnu", "primitive": "kx.p256_ecdh", "symbol": "rscrypto::fixture",
            "kind": "select_branch", "slice_sha256": accepted["slice_sha256"]}
  assert constant_branches(loop, [waiver])[1] == ["asm_waiver[0] matches a finding that a public classification accepts"]
  # The counter advances only when an argument is nonzero. Its data slice has no leaves, but the argument
  # decides its value, so the second branch stays blocked.
  guarded = ["1000: b9 00 00 00 00\tmovl\t$0x0, %ecx", "1005: 85 ff\ttestl\t%edi, %edi",
             "1007: 74 03\tje\t0x100c <rscrypto::fixture+0xc>", "1009: 83 c1 01\taddl\t$0x1, %ecx",
             "100c: 83 f9 01\tcmpl\t$0x1, %ecx", "100f: 74 05\tje\t0x1016 <rscrypto::fixture+0x16>",
             "1011: b8 01 00 00 00\tmovl\t$0x1, %eax", "1016: c3\tretq"]
  argument, counter = constant_branches(guarded)[0]
  assert argument["slice_leaves"] == ["entry rdi"] and not argument["slice_constant"], argument
  assert not counter["slice_open"] and counter["slice_leaves"] == [] and not counter["slice_constant"], counter
  assert counter["disposition"] == "needs-fix", counter
  # An argument bounds the loop, so the count that the branch after the loop tests is the argument.
  bounded = ["1000: b9 00 00 00 00\tmovl\t$0x0, %ecx", "1005: 83 c1 01\taddl\t$0x1, %ecx",
             "1008: 39 f9\tcmpl\t%edi, %ecx", "100a: 75 f9\tjne\t0x1005 <rscrypto::fixture+0x5>",
             "100c: 83 f9 03\tcmpl\t$0x3, %ecx", "100f: 74 05\tje\t0x1016 <rscrypto::fixture+0x16>",
             "1011: b8 01 00 00 00\tmovl\t$0x1, %eax", "1016: c3\tretq"]
  [after] = constant_branches(bounded)[0]
  assert after["slice_leaves"] == [] and not after["slice_constant"] and after["disposition"] == "needs-fix", after


def exrl_template_regressions() -> None:
  """IBM Z `exrl` binds the instruction that it executes, not where its template sits."""
  target = "s390x-unknown-linux-gnu"
  mvc = bytes.fromhex("d20030008000")
  xc = bytes.fromhex("d700e000e000")

  def build(offset: int, template: bytes, after: bytes):
    text, island = 0x10000, 0x30000
    encoded = "c6 10 " + ((island + offset - text) // 2).to_bytes(4, "big").hex(" ")
    return linked_fixture(text, [
      f"{text:x}: {encoded}\texrl\t%r1, 0x{island + offset:x}",
      f"{text + 6:x}: 07 fe\tbr\t%r14",
    ], {island: bytes(offset) + template + after})

  original = fixture_hash(target, build(0x142, mvc, xc))
  # The template island moved by 6 bytes, and another template follows it: the identity stays.
  assert fixture_hash(target, build(0x148, mvc, mvc)) == original
  # A different template is a different instruction.
  assert fixture_hash(target, build(0x142, xc, xc)) != original


def data_reference_regressions() -> None:
  """Data references bind the referenced object, not its distance from the code."""
  table = bytes(range(32))
  changed = table[:17] + b"\xff" + table[18:]

  def x86(text: int, rodata: int, contents: bytes, name: str = "rscrypto::TABLE"):
    displacement = rodata + 0x10 - (text + 7)
    return linked_fixture(text, [
      f"{text:x}: 48 8d 05 {displacement.to_bytes(4, 'little').hex(' ')}\tleaq\t0x{displacement:x}(%rip), %rax"
      f" # 0x{rodata + 0x10:x} <{name}+0x10>",
      f"{text + 7:x}: c3\tretq",
    ], {rodata: contents}, units=[(rodata, rodata + len(contents), None)], names={rodata: name})

  def aarch64(text: int, rodata: int, contents: bytes):
    return linked_fixture(text, [
      f"{text:x}: 90000108\tadrp\tx8, 0x{rodata:x}",
      f"{text + 4:x}: 91004108\tadd\tx8, x8, #0x10",
      f"{text + 8:x}: d65f03c0\tret",
    ], {rodata: contents}, units=[(rodata, rodata + len(contents), None)])

  def riscv(text: int, rodata: int, contents: bytes):
    return linked_fixture(text, [
      f"{text:x}: 00020517\tauipc\ta0, 0x{(rodata - text) >> 12:x}",
      f"{text + 4:x}: 01050513\taddi\ta0, a0, 0x10",
      f"{text + 8:x}: 00008067\tret",
    ], {rodata: contents}, units=[(rodata, rodata + len(contents), None)])

  def power(text: int, rodata: int, contents: bytes):
    # The global entry computes the TOC from r12; the pair then addresses TOC-relative data.
    # The TOC sits at a placement-dependent distance from the data, as after relinking.
    toc = rodata + 0x10000 + (text >> 4)
    high = (toc - text + 0x8000) >> 16
    low = toc - text - (high << 16)
    delta = rodata + 0x10 - toc
    data_high = (delta + 0x8000) >> 16
    data_low = delta - (data_high << 16)
    return linked_fixture(text, [
      f"{text:x}: 3c4c0002\taddis 2, 12, {high}",
      f"{text + 4:x}: 38428000\taddi 2, 2, {low}",
      f"{text + 8:x}: 3c62ffff\taddis 3, 2, {data_high}",
      f"{text + 12:x}: 38630010\taddi 3, 3, {data_low}",
      f"{text + 16:x}: 4e800020\tblr",
    ], {rodata: contents}, units=[(rodata, rodata + len(contents), None)], toc_base=toc)

  def s390x(text: int, rodata: int, contents: bytes):
    encoded = "c0 10 " + ((rodata + 0x10 - text) // 2).to_bytes(4, "big").hex(" ")
    return linked_fixture(text, [
      f"{text:x}: {encoded}\tlarl\t%r1, 0x{rodata + 0x10:x}",
      f"{text + 6:x}: 07 fe\tbr\t%r14",
    ], {rodata: contents}, units=[(rodata, rodata + len(contents), None)])

  for target, build in (
    ("x86_64-unknown-linux-gnu", x86),
    ("aarch64-unknown-linux-gnu", aarch64),
    ("riscv64gc-unknown-linux-gnu", riscv),
    ("powerpc64le-unknown-linux-gnu", power),
    ("s390x-unknown-linux-gnu", s390x),
  ):
    original = fixture_hash(target, build(0x10000, 0x30000, table))
    # Relinking moves both the code and the data; the identity stays.
    assert fixture_hash(target, build(0x50000, 0x91000, table)) == original, target
    # One changed byte anywhere in the referenced object is a different instruction.
    assert fixture_hash(target, build(0x10000, 0x30000, changed)) != original, target
    # Without an allocation unit, the reference keeps its exact displacement.
    body, image = build(0x10000, 0x30000, table)
    image.extents.clear()
    image.__post_init__()
    exact = fixture_hash(target, (body, image))
    body, image = build(0x50000, 0x91000, table)
    image.extents.clear()
    image.__post_init__()
    assert fixture_hash(target, (body, image)) != exact, target

  target = "x86_64-unknown-linux-gnu"
  assert fixture_hash(target, x86(0x10000, 0x30000, table, "rscrypto::OTHER")) != fixture_hash(
    target, x86(0x10000, 0x30000, table))

  # A relocated pointer inside read-only data binds its target's contents, not its address.
  def location(text: int, rodata: int, path: bytes, line: int):
    contents = bytes(8) + len(path).to_bytes(8, "little") + line.to_bytes(8, "little")
    strings = rodata + 0x100
    displacement = rodata - (text + 7)
    return linked_fixture(text, [
      f"{text:x}: 48 8d 3d {displacement.to_bytes(4, 'little').hex(' ')}\tleaq\t0x{displacement:x}(%rip), %rdi",
      f"{text + 7:x}: c3\tretq",
    ], {rodata: contents, strings: path + b"\0" + b"padding"},
      units=[(rodata, rodata + 24, None), (strings, strings + len(path) + 8, 0)], slots={rodata: Slot(address=strings)})

  original = fixture_hash(target, location(0x10000, 0x30000, b"src/auth/rsa.rs", 7))
  assert fixture_hash(target, location(0x50000, 0x92000, b"src/auth/rsa.rs", 7)) == original
  assert fixture_hash(target, location(0x10000, 0x30000, b"src/auth/rsb.rs", 7)) != original
  assert fixture_hash(target, location(0x10000, 0x30000, b"src/auth/rsa.rs", 8)) != original

  # An `adrp` whose register is read after a branch cannot be attributed to
  # one consumer, so the page stays exact and relinking changes the identity.
  def branching(text: int, rodata: int):
    return linked_fixture(text, [
      f"{text:x}: 90000109\tadrp\tx9, 0x{rodata:x}",
      f"{text + 4:x}: b4000040\tcbz\tx0, 0x{text + 0xc:x} <rscrypto::fixture+0xc>",
      f"{text + 8:x}: f9400920\tldr\tx0, [x9, #0x10]",
      f"{text + 0xc:x}: d65f03c0\tret",
    ], {rodata: table}, units=[(rodata, rodata + 32, None)])

  target = "aarch64-unknown-linux-gnu"
  assert fixture_hash(target, branching(0x10000, 0x30000)) != fixture_hash(target, branching(0x50000, 0x91000))


# The assertion message of `rsa_aarch64_pair::exponentiate_pair`, as the Apple builds at 6f571652 and 868dcbca
# link it, and the panic call that passes it with its length.
EXPONENTIATE_PAIR_MESSAGE = (
  b"assertion failed: digits.is_multiple_of(BLOCK_DIGITS) && digits >= BLOCK_DIGITS.strict_mul(3) &&\n"
  b"                p.m.len() == q.m.len() &&\n            digits.strict_mul(28) >= p.m.len().strict_mul(64)"
  b".strict_add(2) &&\n        p.exponent.len() == q.exponent.len() &&\n    workspace.len() >= pair_workspace_words(digits)")


# The IR parameters of `core::panicking::panic(&'static str, &Location)` and of a function that takes a `&[u64]`
# and its length, as rustc emits them (nightly-2026-10-24, ddf3a965 x86-64 Linux `rscrypto_ct_evidence.ll`).
PANIC_PARAMETERS = ("ptr noalias nofree noundef nonnull readonly captures(address, read_provenance) %0",
                    "i64 noundef range(i64 32, 143) %1",
                    "ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %2")
WORD_SLICE_PARAMETERS = ("ptr noalias nofree noundef nonnull readonly align 8 captures(address) %0", "i64 noundef %1")
# `Sha256::update_impl(&mut self, data: &[u8])` as the two modules of the Windows build at ddf3a965 define it.
UPDATE_IMPL_FORMS = frozenset({
  ("ptr noalias nofree noundef nonnull align 8 captures(address) dereferenceable(120) %0",
   "ptr noalias nofree noundef nonnull readonly captures(address) %1",
   "i64 noundef range(i64 0, -9223372036854775808) %2"),
  ("ptr noalias nofree noundef nonnull align 8 dereferenceable(120) %self",
   "ptr noalias nofree noundef nonnull readonly captures(address, read_provenance) %data.0",
   "i64 noundef range(i64 0, -9223372036854775808) %data.1"),
})


def ld64_atom_regressions() -> None:
  """An ld64 atom can end in the alignment padding before the next atom; a reference binds it unless its extent is
  known (2026-11-15 Apple renewal of `exponentiate_pair`)."""
  target = "aarch64-apple-darwin"
  assert len(EXPONENTIATE_PAIR_MESSAGE) == 317

  def linked(code: int, page_offset: int, size: int, gap: int, length: str = "528027a1\tmov\tw1, #0x13d",
             message: bytes = EXPONENTIATE_PAIR_MESSAGE, parameters: tuple[str, ...] = PANIC_PARAMETERS):
    atom = 0x1000a6000 + page_offset
    lines = [f"{code:x}: f0000360\tadrp\tx0, 0x1000a6000", f"{code + 4:x}: 91323000\tadd\tx0, x0, #0x{page_offset:x}",
             f"{code + 8:x}: {length}", f"{code + 12:x}: 94013ef7\tbl\t0x1000878ec <core::panicking::panic>"]
    body = FunctionBody("rscrypto::fixture", Path("fixture.binary.disasm.txt"), code, list(enumerate(lines, start=1)))
    data = bytes(0x1000)
    data = data[:page_offset] + message + bytes(size - len(message)) + data[page_offset + size:]
    image = LinkedImage(data, [Section("__TEXT,__text", 0x100037000, 0x1000, None, True, False),
                               Section("__TEXT,__const", 0x1000a6000, 0x1000, 0, False, False)], True,
                        extents=[(atom, atom + size, None), (atom + size + gap, atom + size + gap + 0x100, None)],
                        padded_units=True)
    image.set_functions([(code, code + 0x10, body.symbol), (0x1000878ec, 0x100087900, "core::panicking::panic")])
    image.parameters = {"core::panicking::panic": frozenset({parameters})}
    return body, image

  def hashed(fixture) -> str:
    body, image = fixture
    return function_identity(instruction_identities(target, body, {body.symbol, "core::panicking::panic"}, image))

  # 6f571652: the atom holds one byte of padding (0x13e); 868dcbca: it holds none (0x13d), and a gap follows it.
  original = hashed(linked(0x100037cfc, 0xc8c, 0x13e, 0))
  assert hashed(linked(0x100037d18, 0xd24, 0x13d, 0x27)) == original
  changed = EXPONENTIATE_PAIR_MESSAGE.replace(b"digits)", b"digitz)")
  assert hashed(linked(0x100037cfc, 0xc8c, 0x13e, 0, message=changed)) != original
  # A length in a register is no extent, so the padding byte binds.
  register = "2a1403e1\tmov\tw1, w20"
  assert hashed(linked(0x100037cfc, 0xc8c, 0x13e, 0, length=register)) != hashed(
    linked(0x100037d18, 0xd24, 0x13d, 0x27, length=register))
  # The length of a `&[u64]` counts words, not bytes, so it is no byte extent and the padding byte binds.
  assert hashed(linked(0x100037cfc, 0xc8c, 0x13e, 0, parameters=WORD_SLICE_PARAMETERS)) != hashed(
    linked(0x100037d18, 0xd24, 0x13d, 0x27, parameters=WORD_SLICE_PARAMETERS))
  # A length past the end of the atom is not a byte length, such as `len << 1 | 1` of a `fmt::Arguments` literal
  # (0x27b for 317 bytes), so the reference binds the whole atom.
  arguments = "52804f61\tmov\tw1, #0x27b"
  assert hashed(linked(0x100037cfc, 0xc8c, 0x13e, 0, length=arguments)) != hashed(
    linked(0x100037d18, 0xd24, 0x13d, 0x27, length=arguments))


def location_bytes(line: int, column: int, path: bytes) -> bytes:
  """A 64-bit little-endian `core::panic::Location` with an unrelocated name pointer."""
  return bytes(8) + len(path).to_bytes(8, "little") + line.to_bytes(4, "little") + column.to_bytes(4, "little")


def placement_regressions() -> None:
  """Identities bind what an address designates, never where the linker or the source tree put it."""
  target = "x86_64-unknown-linux-gnu"

  # A panic location describes as `loc`: its file, checkout path, and line record source placement.
  def location(text: int, rodata: int, path: bytes, line: int):
    strings = rodata + 0x100
    displacement = rodata - (text + 7)
    return linked_fixture(text, [
      f"{text:x}: 48 8d 3d {displacement.to_bytes(4, 'little').hex(' ')}\tleaq\t0x{displacement:x}(%rip), %rdi",
      f"{text + 7:x}: c3\tretq",
    ], {rodata: location_bytes(line, 9, path), strings: path + b"\0"},
      units=[(rodata, rodata + 24, None), (strings, strings + len(path), None)], slots={rodata: Slot(address=strings)})

  original = fixture_hash(target, location(0x10000, 0x30000, b"/home/runner/work/rscrypto/src/auth/rsa.rs", 7))
  assert fixture_hash(target, location(0x50000, 0x92000, b"/Users/someone/rscrypto/src/auth/rsa.rs", 7)) == original
  assert fixture_hash(target, location(0x10000, 0x30000, b"/home/runner/work/rscrypto/src/auth/rsa.rs", 9)) == original
  body, image = location(0x10000, 0x30000, b"src/auth/rsa.rs", 7)
  assert "<loc>" in "\n".join(instruction_identities(target, body, {body.symbol}, image).values())
  # Text that is not a Rust source path is data, so it still binds.
  assert fixture_hash(target, location(0x10000, 0x30000, b"src/auth/rsa.txt", 7)) != fixture_hash(
    target, location(0x10000, 0x30000, b"src/auth/rsa.txt", 9))

  # Inside other read-only contents a location is `<loc>` too, so only the surrounding bytes bind.
  def table(text: int, rodata: int, line: int, tag: int):
    strings = rodata + 0x100
    displacement = rodata - (text + 7)
    contents = bytes([tag] * 8) + location_bytes(line, 3, b"src/lib.rs")
    return linked_fixture(text, [
      f"{text:x}: 48 8d 3d {displacement.to_bytes(4, 'little').hex(' ')}\tleaq\t0x{displacement:x}(%rip), %rdi",
      f"{text + 7:x}: c3\tretq",
    ], {rodata: contents, strings: b"src/lib.rs"},
      units=[(rodata, rodata + 32, None), (strings, strings + 10, None)], slots={rodata + 8: Slot(address=strings)})

  assert fixture_hash(target, table(0x10000, 0x30000, 7, 1)) == fixture_hash(target, table(0x50000, 0x91000, 70, 1))
  assert fixture_hash(target, table(0x10000, 0x30000, 7, 1)) != fixture_hash(target, table(0x10000, 0x30000, 7, 2))

  # Without map sizes (PE), trailing zero bytes before the next object may be alignment padding. Only a reference
  # with a known extent drops them, and never below that extent: an access width, or an immediate length that a
  # call receives with the address (`&str`, `&[u8]`) in the next argument register of the Windows x64 ABI, here
  # rcx and rdx. Any other reference binds the padding.
  pe = "x86_64-pc-windows-msvc"

  def padded(text: int, rodata: int, contents: bytes, padding: int, slot: bool = False, use: str = "paired",
             length: int = 3, following_bytes: bytes = b"next", second: bool = False):
    following = rodata + len(contents) + padding
    displacement = rodata - (text + 7)
    pointer = ("15", "%rdx") if second else ("0d", "%rcx")
    rows = [(f"48 8d {pointer[0]} {displacement.to_bytes(4, 'little').hex(' ')}", f"leaq\t0x{displacement:x}(%rip), {pointer[1]}")]
    rows += {"paired": [(f"41 b8 {length:02x} 00 00 00", f"movl\t$0x{length:x}, %r8d") if second else
                        (f"ba {length:02x} 00 00 00", f"movl\t$0x{length:x}, %edx")],
             "register": [("44 89 c2", "movl\t%r8d, %edx")]}.get(use, [])
    if use in {"paired", "register"}:
      rows.append(("e8 00 00 00 00", f"callq\t0x{text + 0x80:x} <rscrypto::consume>"))
    lines, address = [], text
    for encoding, instruction in [*rows, ("c3", "retq")]:
      lines.append(f"{address:x}: {encoding}\t{instruction}")
      address += len(encoding.split())
    body = FunctionBody("rscrypto::fixture", Path("fixture.binary.disasm.txt"), text, list(enumerate(lines, start=1)))
    data = contents + bytes(padding) + following_bytes
    image = LinkedImage(data, [Section(".text", text, 0x100, None, True, False),
                               Section(".rdata", rodata, len(data), 0, False, False)], True,
                        starts={rodata: None, following: None}, bounded_by_starts=True,
                        slots={rodata: Slot(symbol="HeapFree")} if slot else {})
    image.set_functions([(text, text + 0x80, body.symbol), (text + 0x80, text + 0x100, "rscrypto::consume")])
    image.parameters = {"rscrypto::consume": UPDATE_IMPL_FORMS if second else frozenset({PANIC_PARAMETERS})}
    return body, image

  constant = b"\x01\x02\x03"
  assert fixture_hash(pe, padded(0x10000, 0x30000, constant, 5)) == fixture_hash(
    pe, padded(0x50000, 0x91000, constant, 13))
  assert fixture_hash(pe, padded(0x10000, 0x30000, constant, 5)) != fixture_hash(
    pe, padded(0x10000, 0x30000, b"\x01\x02\x04", 5))
  # Without a known extent, or with a length in a register, the reference can read the padding, so it binds it.
  for use in ("formed", "register"):
    assert fixture_hash(pe, padded(0x10000, 0x30000, constant, 5, use=use)) != fixture_hash(
      pe, padded(0x10000, 0x30000, constant, 13, use=use)), use
  # A length past the object is not a byte length, such as `len << 1 | 1` of a `fmt::Arguments` literal (7 for 3
  # bytes). The reference binds its object, not the next one.
  assert fixture_hash(pe, padded(0x10000, 0x30000, constant, 2, length=7)) == fixture_hash(
    pe, padded(0x10000, 0x30000, constant, 2, length=7, following_bytes=b"NEXT"))
  # Each module defines `update_impl` with other attributes, and both forms declare a byte slice, so the 8-byte zero
  # prefix of PSS M' that `encode_pss_with_mask` hashes (RFC 8017 9.1.1, step 5) binds without its padding.
  assert fixture_hash(pe, padded(0x10000, 0x30000, bytes(8), 8, length=8, second=True)) == fixture_hash(
    pe, padded(0x50000, 0x91000, bytes(8), 0, length=8, second=True))

  # A broadcast reads one element (Intel SDM Vol. 2, VPBROADCAST): `vpbroadcastq` of `__real@000fffffffffffff` in
  # `rsa_x86_64_ifma::build_table::<3>` (Windows, ddf3a965) binds its 8 bytes, with the top zero byte, and not the
  # constants after it, as a 64-byte read of the register width would.
  def broadcast(word: bytes, following_bytes: bytes):
    displacement = 0x30000 - (0x10000 + 10)
    body = FunctionBody("rscrypto::fixture", Path("fixture.binary.disasm.txt"), 0x10000, list(enumerate([
      f"10000: 62 e2 fd 48 59 05 {displacement.to_bytes(4, 'little').hex(' ')}\tvpbroadcastq\t0x{displacement:x}(%rip), %zmm16",
      "1000a: c3\tretq"], start=1)))
    image = LinkedImage(word + following_bytes, [Section(".text", 0x10000, 0x100, None, True, False),
                                                 Section(".rdata", 0x30000, 64, 0, False, False)], True,
                        starts={0x30000: "__real@000fffffffffffff", 0x30008: None}, bounded_by_starts=True)
    image.set_functions([(0x10000, 0x10100, body.symbol)])
    return body, image

  mask = (0x000FFFFFFFFFFFFF).to_bytes(8, "little")
  assert fixture_hash(pe, broadcast(mask, bytes(range(56)))) == fixture_hash(pe, broadcast(mask, bytes(range(1, 57))))
  assert fixture_hash(pe, broadcast(mask, bytes(56))) != fixture_hash(pe, broadcast(mask[:7] + b"\x01", bytes(56)))
  # A zero byte inside the extent binds: it can become nonzero, and when it leaves the object, the extent reads
  # the next object instead.
  zero = b"\x01\x02\x00"
  assert fixture_hash(pe, padded(0x10000, 0x30000, zero, 5)) != fixture_hash(
    pe, padded(0x10000, 0x30000, b"\x01\x02\x05", 5))
  assert fixture_hash(pe, padded(0x10000, 0x30000, zero, 0)) != fixture_hash(
    pe, padded(0x10000, 0x30000, b"\x01\x02", 0))
  # An import slot is never cut by the padding bound, so it describes by its import: the raw
  # entry (a placement-dependent RVA whose high bytes are zero) and the padding after it are free.
  assert fixture_hash(pe, padded(0x10000, 0x30000, (0x1234).to_bytes(8, "little"), 0, slot=True)) == fixture_hash(
    pe, padded(0x50000, 0x91000, (0x5678).to_bytes(8, "little"), 8, slot=True))
  ld64_atom_regressions()

  # A small extent that starts inside a larger one (a TLS template offset) does not hide the larger one.
  image = LinkedImage(bytes(64), [Section(".data.rel.ro", 0x1000, 64, 0, False, False)], True,
                      extents=[(0x1000, 0x1040, None), (0x1008, 0x1010, None)])
  assert image.region(0x1020)[:2] == (0x1000, 0x1040)

  power_merged_regressions()
  merged_call_regressions()
  power_branch_regressions()
  transfer_form_regressions()
  closed_world_regressions()
  jump_table_regressions()
  merge_table_regressions()
  riscv_live_high_part_regression()
  riscv_zero_low_part_regression()
  riscv_annotated_jump_table_regression()


POWER = "powerpc64le-unknown-linux-gnu"
SYNTHETIC = "60 00 00 00"


def power_entry(address: int, toc: int) -> list[tuple[str, str]]:
  """The TOC setup of a POWER global entry point: `addis 2, 12, high` and `addi 2, 2, low` reach the TOC base."""
  high = (toc - address + 0x8000) >> 16
  return [(SYNTHETIC, f"addis 2, 12, {high}"), (SYNTHETIC, f"addi 2, 2, {toc - address - (high << 16)}")]


def merged_assembly(size: int, objects: list[tuple[int, str, str, int]], name: str = ".L_MergedGlobals.2") -> str:
  """Compiler assembly of one global-merge unit: (offset, data directives, alias, size) objects, zero padding between."""
  lines, position = [f"{name}:"], 0
  for offset, directives, _alias, length in objects:
    if offset > position:
      lines.append(f"\t.zero\t{offset - position}")
    lines.extend(f"\t{directive}" for directive in directives.split("\n"))
    position = offset + length
  if size > position:
    lines.append(f"\t.zero\t{size - position}")
  lines.append(f"\t.size\t{name}, {size}")
  for offset, _directives, alias, length in objects:
    lines += [f"{alias} = {name}+{offset}", f"\t.size\t{alias}, {length}"]
  return "\n".join(lines) + "\n"


def power_link(toc: int, unit: int, assembly: str, functions: list[tuple[str, int, list[tuple[str, str]]]],
               target: str = POWER) -> tuple[dict[str, FunctionBody], LinkedImage]:
  """Link functions of (symbol, address, (encoding, text) rows) with the global-merge unit of `assembly` at `unit`."""
  [definition] = assembly_merge_units(assembly, True)
  bodies, number = {}, 1
  for symbol, address, rows in functions:
    lines = []
    for index, (encoding, text) in enumerate(rows):
      lines.append((number, f"{address + 4 * index:x}: {encoding}\t{text}"))
      number += 1
    bodies[symbol] = FunctionBody(symbol, Path("fixture.binary.disasm.txt"), address, lines)
  data = definition.contents
  image = LinkedImage(data, [Section(".text", 0x10000, 0x8000, None, True, False),
                             Section(".data.rel.ro", unit, len(data), 0, False, False)], True,
                      extents=[(unit, unit + len(data), MERGED)], toc_base=toc)
  image.set_members(merge_unit_members(image, [definition]))
  prepare_linked_image(target, bodies, image)
  return bodies, image


def linked_identities(target: str, fixture: tuple[dict[str, FunctionBody], LinkedImage], symbol: str) -> list[str]:
  bodies, image = fixture
  return list(instruction_identities(target, bodies[symbol], set(bodies), image).values())


RSA_ROOT = "ct_entry_rsa_pkcs1v15_decrypt_fixed_blinding"
UPDATE = "<rscrypto::hashes::crypto::sha256::Sha256>::update_impl"
HMAC = "rscrypto::auth::rsa::hmac_sha256_parts"
MEMCPY = "0000004f.plt_call.memcpy@@GLIBC_2.17"
MEMMOVE = "0000004f.plt_call.memmove@@GLIBC_2.17"
SHA256_IV = r'.ascii	"g\346\tj\205\256g\273r\363n<:\365O\245\177R\016Q\214h\005\233\253\331\203\037\031\315\340["'
# Excerpts of POWER CT 43670264860 (6f571652) and CT 43699511183 (868dcbca). 868dcbca added HPKE labels to
# `.L_MergedGlobals.2`: "dkp_prk" before "message" and two after it. Each row of the root is the linked
# instruction of `ct_entry_rsa_pkcs1v15_decrypt_fixed_blinding`, in order; the TOC setups are synthetic.
RSA_DECRYPT = {
  "6f571652": {
    "size": 20640, "base": ("a0 84 e3 3a", -31584), "li5": ("60 50 a0 38", 20576), "li7": ("70 50 e0 38", 20592),
    "message": ("74 01 c5 39", 372),
    "objects": [
      (358, '.ascii	" { .. }"', ".Lanon.a93706019fb9a4aab351026df9960745.35", 7),
      (365, '.ascii	"IOError"', ".Lanon.dd9cad2fdbdfc3f7f7080943c244926c.135", 7),
      (372, '.ascii	"message"', ".Lanon.3c5503918be1719b3333892c6703e024.1043", 7),
      (379, '.ascii	"unsafe "', ".Lanon.1c7fae5b1541fafbfe3fcff8c29715af.26", 7),
      (386, '.ascii	"closure"', ".Lanon.1c7fae5b1541fafbfe3fcff8c29715af.53", 7),
      (20576, SHA256_IV, ".Lanon.5db48453cec2c3d3d727b77dae34a150.16", 32),
    ],
  },
  "868dcbca": {
    "size": 20896, "base": ("a0 83 e3 3a", -31840), "li5": ("60 51 a0 38", 20832), "li7": ("70 51 e0 38", 20848),
    "message": ("9b 01 c5 39", 411),
    "objects": [
      (390, '.ascii	" { .. }"', ".Lanon.a93706019fb9a4aab351026df9960745.35", 7),
      (397, '.ascii	"IOError"', ".Lanon.dd9cad2fdbdfc3f7f7080943c244926c.135", 7),
      (404, '.ascii	"dkp_prk"', ".Lanon.8bd117a2bc6b8f81c36665eb8c4023c2.826", 7),
      (411, '.ascii	"message"', ".Lanon.8bd117a2bc6b8f81c36665eb8c4023c2.1049", 7),
      (418, '.ascii	"HPKE-v1"', ".Lanon.8bd117a2bc6b8f81c36665eb8c4023c2.1080", 7),
      (425, '.ascii	"eae_prk"', ".Lanon.8bd117a2bc6b8f81c36665eb8c4023c2.1220", 7),
      (432, '.ascii	"unsafe "', ".Lanon.1c7fae5b1541fafbfe3fcff8c29715af.26", 7),
      (439, '.ascii	"closure"', ".Lanon.1c7fae5b1541fafbfe3fcff8c29715af.53", 7),
      (20832, SHA256_IV, ".Lanon.911c70f77840cbbdf87f006a729f1b0e.21", 32),
    ],
  },
}


def rsa_decrypt(version: str, *, update_call: list[tuple[str, str]] | None = None, message: int | None = None,
                index: int | None = None, contents: dict[str, str] | None = None):
  """Link the RSA root excerpt of `version` with its callees and global-merge objects.

  `update_call` replaces the `memcpy` call of `update_impl`; `message` and `index` replace the immediates of
  `addi 14, 5, ...` and `li 7, ...`; `contents` replaces object directives.
  """
  layout = RSA_DECRYPT[version]
  toc, root, update, hmac, stub = 0x200000, 0x10000, 0x11000, 0x12000, 0x13000
  unit = toc - 0x10000 + layout["base"][1]
  objects = [(offset, (contents or {}).get(directive, directive), alias, size)
             for offset, directive, alias, size in layout["objects"]]
  li7 = layout["li7"] if index is None else (SYNTHETIC, index)
  addi14 = layout["message"] if message is None else (SYNTHETIC, message)
  rows = [*power_entry(root, toc),
          ("ff ff 62 3c", "addis 3, 2, -1"), (layout["li5"][0], f"li 5, {layout['li5'][1]}"),
          ("48 00 9c e8", "ld 4, 72(28)"), (li7[0], f"li 7, {li7[1]}"), ("70 01 c1 ea", "ld 22, 368(1)"),
          ("40 01 01 eb", "ld 24, 320(1)"), ("00 00 40 3b", "li 26, 0"), ("78 33 d9 7c", "mr\t25, 6"),
          (layout["base"][0], f"addi 23, 3, {layout['base'][1]}"), ("40 00 60 38", "li 3, 64"),
          ("00 05 41 b3", "sth 26, 1280(1)"), ("98 2e 17 7c", "lxvd2x 0, 23, 5"), ("50 00 bc e8", "ld 5, 80(28)"),
          ("90 04 81 3b", "addi 28, 1, 1168"), ("98 3e 37 7c", "lxvd2x 1, 23, 7"),
          ("98 1f 1c 7c", "stxvd2x 0, 28, 3"), ("d0 04 00 f0", "xxlxor 0, 0, 0"), ("30 00 60 38", "li 3, 48"),
          ("98 1f 1c 7c", "stxvd2x 0, 28, 3"), ("20 00 60 38", "li 3, 32"), ("98 e7 00 7c", "stxvd2x 0, 0, 28"),
          ("98 1f 1c 7c", "stxvd2x 0, 28, 3"), ("10 00 60 38", "li 3, 16"), ("98 1f 1c 7c", "stxvd2x 0, 28, 3"),
          ("50 00 60 38", "li 3, 80"), ("98 1f 3c 7c", "stxvd2x 1, 28, 3"), ("60 00 60 38", "li 3, 96"),
          ("98 1f 1c 7c", "stxvd2x 0, 28, 3"), ("78 e3 83 7f", "mr\t3, 28"),
          ("21 db fa 4b", f"bl 0x{update + 8:x} <{UPDATE}+0x8>"), ("c0 05 61 3b", "addi 27, 1, 1472"),
          ("48 00 e1 fa", "std 23, 72(1)"), ("48 00 a1 e8", "ld 5, 72(1)"), (addi14[0], f"addi 14, 5, {addi14[1]}"),
          ("78 db 63 7f", "mr\t3, 27"), ("03 00 c0 38", "li 6, 3"), ("78 d3 44 7f", "mr\t4, 26"),
          ("78 93 45 7e", "mr\t5, 18"), ("c0 05 21 fa", "std 17, 1472(1)"), ("c8 05 81 fb", "std 28, 1480(1)"),
          ("d0 05 c1 f9", "std 14, 1488(1)"), ("d8 05 c1 fa", "std 22, 1496(1)"),
          ("59 30 fe 4b", f"bl 0x{hmac + 8:x} <{HMAC}+0x8>"), ("20 00 80 4e", "blr")]
  call = update_call if update_call is not None else [("79 4e fd 4b", f"bl 0x{stub:x} <{MEMCPY}>")]
  update_rows = [*power_entry(update, toc), ("a6 02 08 7c", "mflr 0"), ("78 23 9c 7c", "mr\t28, 4"),
                 ("78 1b 7d 7c", "mr\t29, 3"), ("78 2b be 7c", "mr\t30, 5"), ("40 00 7d 38", "addi 3, 29, 64"),
                 ("78 eb a4 7f", "mr\t4, 29"), ("40 00 a0 38", "li 5, 64"), *call, ("18 00 41 e8", "ld 2, 24(1)"),
                 ("a6 03 08 7c", "mtlr 0"), ("20 00 80 4e", "blr")]
  stub_rows = [("18 00 41 f8", "std 2, 24(1)"), ("ff ff 82 3d", "addis 12, 2, -1"), ("c8 7d 8c e9", "ld 12, 32200(12)"),
               ("a6 03 89 7d", "mtctr 12"), ("20 04 80 4e", "bctr")]
  return power_link(toc, unit, merged_assembly(layout["size"], objects), [
    (RSA_ROOT, root, rows), (UPDATE, update, update_rows),
    (HMAC, hmac, [*power_entry(hmac, toc), ("a6 02 08 7c", "mflr 0"), ("20 00 80 4e", "blr")]),
    (MEMCPY, stub, stub_rows), (MEMMOVE, stub + 0x100, stub_rows)])


def rsa_row(fixture, text: str) -> str:
  """The identity, without its offset, of the root's row whose disassembly starts with `text`."""
  bodies, image = fixture
  body = bodies[RSA_ROOT]
  identities = instruction_identities(POWER, body, set(bodies), image)
  return next(identities[number].split(": ", 1)[1] for number, line in body.lines
              if line.split("\t", 1)[1].startswith(text))


def power_merged_regressions() -> None:
  """A global-merge reference binds the object that the compiler merged, not the unit layout.

  The 2026-11-15 HPKE renewal (POWER CT 43699511183) had two causes, which these real excerpts pin.
  """
  def root(fixture) -> str:
    return function_identity(dict(enumerate(linked_identities(POWER, fixture, RSA_ROOT))))

  old, new = rsa_decrypt("6f571652"), rsa_decrypt("868dcbca")
  assert root(old) == root(new)

  # Cause 2: `addi 14, 5, 372` became `411`. The member came from the addresses that every function formed, so
  # it grew from "message" to 21 bytes when no listed function formed the new labels after it. The compiler's
  # own object table keeps "message" at 7 bytes.
  assert rsa_row(old, "addi 14, 5,") == rsa_row(new, "addi 14, 5,")
  assert "/7:" in rsa_row(new, "addi 14, 5,"), rsa_row(new, "addi 14, 5,")
  # A changed loaded byte or an address of other data is a changed identity.
  changed = {'.ascii	"message"': '.ascii	"massage"'}
  assert rsa_row(rsa_decrypt("868dcbca", contents=changed), "addi 14, 5,") != rsa_row(new, "addi 14, 5,")
  assert rsa_row(rsa_decrypt("868dcbca", message=418), "addi 14, 5,") != rsa_row(new, "addi 14, 5,")

  # Cause 1: `li 7, 20592` became `20848`. It only selects the object that `lxvd2x 1, 23, 7` loads, but it is still
  # in r7 at `bl update_impl`. `update_impl` reads r3-r5 and passes r3-r5 to `memcpy`, so r7 is no argument.
  assert rsa_row(old, "li 7,") == rsa_row(new, "li 7,") == "4B li 7, <member-offset>"
  assert rsa_row(old, "lxvd2x 1,") == rsa_row(new, "lxvd2x 1,")
  iv = {SHA256_IV: SHA256_IV.replace(r'"g\346', r'"h\346')}
  assert root(rsa_decrypt("868dcbca", contents=iv)) != root(new)
  assert rsa_row(rsa_decrypt("868dcbca", index=20864), "lxvd2x 1,") != rsa_row(new, "lxvd2x 1,")
  # An import without a prototype, and an indirect call, can read every argument register, so the immediate stays.
  for call in ([("79 4e fd 4b", f"bl 0x13100 <{MEMMOVE}>")], [("a6 03 89 7d", "mtctr 12"), ("21 04 80 4e", "bctrl")]):
    old_call, new_call = rsa_decrypt("6f571652", update_call=call), rsa_decrypt("868dcbca", update_call=call)
    assert rsa_row(old_call, "li 7,") != rsa_row(new_call, "li 7,"), call
    assert rsa_row(new_call, "li 7,").endswith("\tli 7, 20848"), rsa_row(new_call, "li 7,")

  power_transfer_regressions()


def power_transfer_regressions() -> None:
  """A global-merge base that leaves the function binds its whole unit."""
  toc, text, unit = 0x200000, 0x10000, 0x1e0000
  objects = [(0, '.ascii	"index out of range"', ".Lanon.a.1", 18), (24, '.ascii	"other"', ".Lanon.a.2", 5)]

  def link(leave: list[tuple[str, str]], other: str = '.ascii	"other"', callee: list[tuple[str, str]] | None = None):
    rows = [*power_entry(text, toc), (SYNTHETIC, "addis 3, 2, -1"), (SYNTHETIC, f"addi 3, 3, {unit - toc + 0x10000}"),
            (SYNTHETIC, "li 4, 0"), *leave]
    callee_rows = callee or [*power_entry(0x11000, toc), (SYNTHETIC, "blr")]
    return power_link(toc, unit, merged_assembly(32, [objects[0], (24, other, ".Lanon.a.2", 5)]), [
      ("rscrypto::fixture", text, rows), ("rscrypto::callee", 0x11000, callee_rows),
      ("rscrypto::reads_r3", 0x11100, [*power_entry(0x11100, toc), (SYNTHETIC, "lbz 5, 0(3)"), (SYNTHETIC, "blr")])])

  def bound(leave: list[tuple[str, str]]) -> bool:
    """Whether the other object, which no instruction addresses, binds the function."""
    first = linked_identities(POWER, link(leave), "rscrypto::fixture")
    return first != linked_identities(POWER, link(leave, '.ascii	"OTHER"'), "rscrypto::fixture")

  # The base is still in r3 at a call to a callee that reads no argument, and the call clobbers r3.
  assert not bound([(SYNTHETIC, "bl 0x11008 <rscrypto::callee+0x8>"), (SYNTHETIC, "blr")])
  # The base leaves in r3: it is returned, passed to a callee that reads r3, tail-called with, or the function
  # continues through a jump table, which the dataflow cannot follow.
  for leave in ([(SYNTHETIC, "blr")], [(SYNTHETIC, "bclr 12, 2"), (SYNTHETIC, "li 3, 0"), (SYNTHETIC, "blr")],
                [(SYNTHETIC, "bl 0x11108 <rscrypto::reads_r3+0x8>"), (SYNTHETIC, "blr")],
                [(SYNTHETIC, "b 0x11108 <rscrypto::reads_r3+0x8>")],
                [(SYNTHETIC, "mtctr 4"), (SYNTHETIC, "bctr"), (SYNTHETIC, "addis 5, 2, -1"),
                 (SYNTHETIC, f"addi 5, 5, {unit - toc + 0x10000}"), (SYNTHETIC, "blr")]):
    assert bound(leave), leave


def merged_call_regressions() -> None:
  """AArch64 and RISC-V: a global-merge base reaches a callee only through an argument that the callee reads."""
  objects = [(0, '.ascii	"index out of range"', ".Lanon.d.1", 18), (24, '.ascii	"other"', ".Lanon.d.2", 5)]

  def bound(target: str, rows: list[tuple[str, str]], callee: list[tuple[str, str]]) -> bool:
    """Whether the object that no instruction addresses binds the caller."""
    identities = []
    for other in ('.ascii	"other"', '.ascii	"OTHER"'):
      fixture = power_link(0, 0x30000, merged_assembly(32, [objects[0], (24, other, ".Lanon.d.2", 5)]), [
        ("rscrypto::fixture", 0x10000, rows), ("rscrypto::callee", 0x11000, callee)], target=target)
      identities.append(linked_identities(target, fixture, "rscrypto::fixture"))
    return identities[0] != identities[1]

  aarch64 = "aarch64-unknown-linux-gnu"
  base = [("90000108", "adrp\tx8, 0x30000"), ("91000108", "add\tx8, x8, #0x0"), ("39400109", "ldrb\tw9, [x8]")]
  call = [("94000400", "bl\t0x11000 <rscrypto::callee>"), ("d65f03c0", "ret")]
  # x8 is still the base at the call. The callee reads no argument, and the call clobbers x8.
  assert not bound(aarch64, base + call, [("d65f03c0", "ret")])
  # The callee writes its result through x8, so the base reaches it.
  assert bound(aarch64, base + call, [("f900011f", "str\txzr, [x8]"), ("d65f03c0", "ret")])

  riscv = "riscv64gc-unknown-linux-gnu"
  base = [("00020797", "auipc\ta5, 0x20"), ("00078793", "addi\ta5, a5, 0x0"), ("0007c803", "lbu\ta6, 0x0(a5)")]
  # `jalr a1` is an indirect call, so every argument register reaches it.
  assert bound(riscv, base + [("000580e7", "jalr\ta1"), ("00008067", "ret")], [("00008067", "ret")])
  # A tail jump leaves the function: its destination reads a5 as an argument.
  reads_a5 = [("0007c503", "lbu\ta0, 0x0(a5)"), ("00008067", "ret")]
  assert bound(riscv, base + [("0000006f", "j\t0x11000 <rscrypto::callee>")], reads_a5)
  assert not bound(riscv, base + [("0000006f", "j\t0x11000 <rscrypto::callee>")], [("00008067", "ret")])


def power_branch_regressions() -> None:
  """Each POWER branch form (Power ISA 3.1 Book I 2.4) is a finding of its kind."""
  # `ct_entry_owner_eq_32` in POWER CT 43707625493 (ddf3a965) returns early through `bclr 12, 2` when an
  # input pointer is null. The scanner knew only conditional branches to labels, so it reported neither return.
  owner = FunctionBody("ct_entry_owner_eq_32", Path("fixture.binary.disasm.txt"), 0x7fe80, list(enumerate([
    "7fe80: 00 00 23 2c  \tcmpdi\t3, 0",
    "7fe84: 78 1b 65 7c  \tmr\t5, 3",
    "7fe88: 00 00 60 38  \tli 3, 0",
    "7fe8c: 20 00 82 4d  \tbclr\t12, 2",
    "7fe90: 00 00 24 2c  \tcmpdi\t4, 0",
    "7fe94: 20 00 82 4d  \tbclr\t12, 2",
    "7fe98: 10 00 60 38  \tli 3, 16",
    "7fe9c: 20 00 80 4e  \tblr",
  ], start=1)))

  def kinds(body: FunctionBody) -> list[tuple[str, int]]:
    rows = scan_symbol(POWER, body.symbol, body, {body.symbol}, scope="entry", primitive_ids=["kx.p256_ecdh"],
                       roots=[body.symbol])
    return [(row["kind"], row["offset"]) for row in rows]

  # A return is no finding; each conditional return is a conditional branch, whose slice reads cr0.
  assert kinds(owner) == [("conditional_branch", 0xc), ("conditional_branch", 0x14)], kinds(owner)
  rows = decode_lines(owner)
  assert row_successors(POWER, owner, rows)[3] == [4] and row_successors(POWER, owner, rows)[7] == []
  from condition_slice import condition_slice
  bounded = condition_slice(POWER, owner, None, 4, {owner.symbol})
  assert bounded.reasons == [] and [node["offset"] for node in bounded.nodes] == [0xc, 0x0], bounded.canonical
  # A call or a jump through CTR is indirect, and a conditional jump through CTR has unknown successors.
  dispatch = FunctionBody("rscrypto::fixture", owner.path, 0x1000, list(enumerate([
    "1000: a6 03 89 7d\tmtctr 12", "1004: 21 04 80 4e\tbctrl", "1008: 20 04 82 4d\tbcctr 12, 2",
    "100c: 20 04 80 4e\tbctr"], start=1)))
  assert kinds(dispatch) == [("indirect_call", 0x4), ("conditional_branch", 0x8), ("indirect_jump", 0xc)]
  assert row_successors(POWER, dispatch, decode_lines(dispatch))[2:] == [None, None]


def transfer_form_regressions() -> None:
  """Each control transfer of an ISA manual is a finding of its kind, with a matching control-flow edge."""
  cases = {
    # z/Architecture Principles of Operation, chapter 7. `ber %r1` jumps through a register, and
    # `blhr %r14` (`core::ptr::drop_glue` at ddf3a965) is a conditional return.
    "s390x-unknown-linux-gnu": [
      ("0d e1\tbasr\t%r14, %r1", "indirect_call", [1]), ("05 e1\tbalr\t%r14, %r1", "indirect_call", [1]),
      ("07 6e\tblhr\t%r14", "conditional_branch", [1]), ("07 81\tber\t%r1", "conditional_branch", None),
      ("06 18\tbctr\t%r1, %r8", "conditional_branch", None), ("47 80 10 00\tbe\t0(%r1)", "conditional_branch", None),
      ("b9 60 80 12\tcgrte\t%r1, %r2", "conditional_branch", [1]),
      ("e3 10 20 00 00 85\tlgat\t%r1, 0(%r2)", "conditional_branch", [1]),
    ],
    # Arm ARM C6.2: pointer-authenticated BLR and BR, and BC.cond.
    "aarch64-unknown-linux-gnu": [
      ("d73f0900\tblraa\tx8, x0", "indirect_call", [1]), ("d71f0a30\tbraa\tx17, x16", "indirect_jump", None),
      # A short forward branch over straight-line code is select-shaped.
      ("54000050\tbc.eq\t0x1008 <rscrypto::fixture+0x8>", "select_branch", [1, 2]),
    ],
    # Intel SDM Vol. 2: far CALL and JMP, and XBEGIN, whose operand is the abort path.
    "x86_64-unknown-linux-gnu": [
      ("ff 18\tlcallq\t*(%rax)", "indirect_call", [1]), ("ff 28\tljmpq\t*(%rax)", "indirect_jump", None),
      ("c7 f8 02 00 00 00\txbegin\t0x1008 <rscrypto::fixture+0x8>", "select_branch", [1, 2]),
    ],
    # Power ISA 3.1 Book I 3.3.11: TRAP WORD and its extended forms.
    "powerpc64le-unknown-linux-gnu": [
      ("08 20 83 7c\ttweq 3, 4", "conditional_branch", [1]), ("00 00 83 0c\ttwgti 3, 0", "conditional_branch", [1]),
    ],
    # RISC-V Unprivileged ISA, "Control Transfer Instructions": `jr a5` jumps through a register.
    "riscv64gc-unknown-linux-gnu": [("00078067\tjr\ta5", "indirect_jump", None), ("9782\tjalr\ta5", "indirect_call", [1])],
  }
  for target, rows in cases.items():
    for line, kind, following in rows:
      body = FunctionBody("rscrypto::fixture", Path("fixture.binary.disasm.txt"), 0x1000, list(enumerate([
        f"1000: {line}", "1004: 00000000\tnop", "1008: 00000000\tnop"], start=1)))
      found = [row["kind"] for row in scan_symbol(target, body.symbol, body, {body.symbol}, scope="entry",
                                                  primitive_ids=["kx.p256_ecdh"], roots=[body.symbol])]
      assert found == [kind], (target, line, found)
      assert row_successors(target, body, decode_lines(body))[0] == following, (target, line)

  # z/Architecture Principles of Operation, chapter 7, BRANCH ON CONDITION: with R2 zero, BCR does not branch, and
  # BCR 15,0 serializes. The rayon scope of Argon2 fences after `Injector::push` (IBM Z, ddf3a965, +0x164). The same
  # row with R2 = 1 jumps through r1.
  scope = "rayon_core::scope::scope::<rscrypto::auth::argon2::fill_slice_parallel::{closure#0}, ()>::{closure#0}"
  for row, kinds, following in [("07 f0        \tbr\t0", [], [2]), ("07 f1        \tbr\t%r1", ["indirect_jump"], None)]:
    body = FunctionBody(scope, Path("fixture.binary.disasm.txt"), 0x397b4, list(enumerate([
      "397b4: c0 e5 ff ff 46 9e    \tbrasl\t%r14, 0x224f0 <<crossbeam_deque::deque::Injector<rayon_core::job::JobRef>>::push>",
      f"397ba: {row}", "397bc: e3 10 d3 70 00 04    \tlg\t%r1, 880(%r13)"], start=1)))
    found = [finding["kind"] for finding in scan_symbol("s390x-unknown-linux-gnu", scope, body, {scope}, scope="entry",
                                                        primitive_ids=["password.argon2i"], roots=[scope])
             if finding["offset"] == 6]
    assert found == kinds, (row, found)
    assert row_successors("s390x-unknown-linux-gnu", body, decode_lines(body))[1] == following, row

  # An x86 conditional tail transfer into another function is a call-graph edge: `je` in
  # `rscrypto_chacha20_poly1305_open_x86_64` (x86-64 Linux, ddf3a965) continues in its AVX2 form.
  root, avx2 = "rscrypto_chacha20_poly1305_open_x86_64", "rscrypto_chacha20_poly1305_open_avx2_x86_64"
  functions = {
    root: FunctionBody(root, Path("fixture.binary.disasm.txt"), 0x537b0, [
      (1, f"537b7: 0f 84 03 44 00 00            \tje\t0x57bc0 <{avx2}>"), (2, "537bd: c3\tretq")]),
    avx2: FunctionBody(avx2, Path("fixture.binary.disasm.txt"), 0x57bc0, [(3, "57bc0: c3\tretq")]),
  }
  assert reachable_closure(root, build_call_graph(functions)) == {root, avx2}


def closed_world_regressions() -> None:
  """The scanner stops on a row that it cannot model: reachable bytes that do not decode, a transfer of control that
  no rule classifies, and, in an enforced closure, an instruction without an effect rule."""
  def fixture(lines: list[str]) -> FunctionBody:
    return FunctionBody("rscrypto::fixture", Path("fixture.binary.disasm.txt"), 0x1000, list(enumerate(lines, start=1)))

  # The Keccak round constants that follow the `ret` of `rscrypto_keccakf1600_aarch64_sve2_sha3_x4` (AArch64 Linux,
  # f4f0d6b0) are data that no edge reaches. The same word after a fall-through could run.
  aarch64 = "aarch64-unknown-linux-gnu"
  constants = ["1000: d65f03c0\tret", "1004: d503201f\tnop", "1008: 01 00 00 00\t.word\t0x00000001"]
  assert control_errors(aarch64, fixture(constants), None) == []
  assert control_errors(aarch64, fixture(constants[1:]), None) == [
    "rscrypto::fixture+0x8: .word 0x00000001: an edge can reach bytes that do not decode"]
  # Arm ARM C6.2, FEAT_CMPBR: CBGT encodes its destination, and no rule names it.
  assert control_errors(aarch64, fixture(["1000: 74000041\tcbgt\tx1, x0, 0x1008 <rscrypto::fixture+0x8>",
                                          "1004: d65f03c0\tret"]), None) == [
    "rscrypto::fixture+0x0: cbgt x1, x0, 0x1008 <rscrypto::fixture+0x8>: a transfer of control without a control rule"]
  # z/Architecture BRANCH RELATIVE AND SAVE calls its operand, and no rule names it.
  assert control_errors("s390x-unknown-linux-gnu", fixture(["1000: a7 e5 00 80\tbras\t%r14, 0x1100 <rscrypto::g>",
                                                            "1004: 07 fe\tbr\t%r14"]), None) == [
    "rscrypto::fixture+0x0: bras %r14, 0x1100 <rscrypto::g>: a transfer of control without a control rule"]
  # x86 VPERMQ has no effect rule; VPXORQ has one.
  x86 = "x86_64-unknown-linux-gnu"
  vector = fixture(["1000: 62 f3 fd 48 00 c1 4e\tvpermq\t$0x4e, %zmm1, %zmm0",
                    "1007: 62 f1 f5 48 ef c2\tvpxorq\t%zmm2, %zmm1, %zmm0", "100d: c3\tretq"])
  assert control_errors(x86, vector, None) == []
  assert effect_errors(x86, vector, {vector.symbol}, None) == [
    "rscrypto::fixture+0x0: vpermq $0x4e, %zmm1, %zmm0: no effect rule"]

  # In an object listing, the relocation after an IBM Z `exrl` names its template, a row of `.text`; its addend counts
  # from the relocated field, 2 bytes into the `exrl` (`Sha512::update` at 466c3556). An MVC continues; a BRC jumps.
  def executed(template: str) -> list[str]:
    with TemporaryDirectory() as directory:
      listing = Path(directory) / "fixture.o.disasm.txt"
      listing.write_text("\n".join([
        "Disassembly of section .text:", "", "0000000000000000 <.text>:", f"      48: {template}", "",
        "Disassembly of section .text.update:", "", "0000000000000000 <rscrypto::fixture>:",
        "     174: c6 20 00 00 00 00    \texrl\t%r2, 0x174", "\t\t0000000000000176:  R_390_PC32DBL\t.text+0x4a",
        "     17a: 07 fe    \tbr\t%r14"]) + "\n")
      functions, _ = parse_disassembly(listing)
      return control_errors("s390x-unknown-linux-gnu", functions["rscrypto::fixture"], None)
  assert executed("d2 00 30 00 40 00    \tmvc\t0(1,%r3), 0(%r4)") == []
  assert executed("a7 f4 00 08          \tj\t0x58") == [
    "rscrypto::fixture+0x174: exrl %r2, 0x174: a transfer of control without a control rule"]

  # z/Architecture BRANCH ON CONDITION: with R2 zero, BCR does not branch. `bnor 0` (BCR 14,0) serializes, so it is
  # no conditional branch, and it continues at the next row.
  target = "s390x-unknown-linux-gnu"
  fence = fixture(["1000: 07 e0\tbnor\t0", "1002: 07 fe\tbr\t%r14"])
  assert scan_symbol(target, fence.symbol, fence, {fence.symbol}, scope="entry", primitive_ids=["kx.p256_ecdh"],
                     roots=[fence.symbol]) == []
  assert row_successors(target, fence, decode_lines(fence))[0] == [1]


def jump_finding(target: str, body: FunctionBody) -> dict:
  """The finding of the jump in `body`; the table load also has a register-indexed operand."""
  [found] = [row for row in scan_symbol(target, body.symbol, body, {body.symbol}, scope="entry",
                                        primitive_ids=["mac.hmac_verify"], roots=[body.symbol])
             if row["kind"] in {"conditional_branch", "indirect_jump"}]
  return found


def jump_table_regressions() -> None:
  """An indirect jump through a read-only table on an index is a multi-way branch on that index; any other indirect
  jump fails."""
  from condition_slice import FunctionSlices

  cases = {
    # `HkdfSha256::extract` in `rscrypto_ct_evidence.o` (RISC-V, ddf3a965): the `match padded` of
    # `HmacSha256::mac_into` (src/auth/hmac.rs), on the block count (s1 + 0x88) >> 6.
    "riscv64gc-unknown-linux-gnu": (0x382, [
      "     382: 08848513     \taddi\ta0, s1, 0x88", "     386: 8119         \tsrli\ta0, a0, 0x6",
      "0000000000000388 <.Lpcrel_hi1689>:", "     388: 00000597     \tauipc\ta1, 0x0",
      "\t\t0000000000000388:  R_RISCV_PCREL_HI20\t.LJTI533_0", "     38c: 050a         \tslli\ta0, a0, 0x2",
      "     38e: 00058593     \tmv\ta1, a1", "\t\t000000000000038e:  R_RISCV_PCREL_LO12_I\t.Lpcrel_hi1689",
      "     392: 952e         \tadd\ta0, a0, a1", "     394: ff852503     \tlw\ta0, -0x8(a0)",
      "     398: 952e         \tadd\ta0, a0, a1", "     39a: 8502         \tjr\ta0",
    ], "entry s1", ("     38e: 00058593     \tmv\ta1, a1", "     38e: 0005b583     \tld\ta1, 0x0(a1)")),
    # `ct_entry_hmac_sha384_verify` in the x86-64 Linux harness object (ddf3a965): the same `match padded`.
    "x86_64-unknown-linux-gnu": (0x19d, [
      "     19d: 40 0f b6 c5                  \tmovzbl\t%bpl, %eax",
      "     1a1: 48 8d 0d 00 00 00 00         \tleaq\t(%rip), %rcx            # 0x1a8 <rscrypto::fixture+0xb>",
      "\t\t00000000000001a4:  R_X86_64_PC32\t.rodata.ct_entry_hmac_sha384_verify-0x4",
      "     1a8: 48 63 04 81                  \tmovslq\t(%rcx,%rax,4), %rax",
      "     1ac: 48 01 c8                     \taddq\t%rcx, %rax", "     1af: ff e0                        \tjmpq\t*%rax",
    ], "entry rbp", ("\t\t00000000000001a4:  R_X86_64_PC32\t.rodata.ct_entry_hmac_sha384_verify-0x4",
                     "\t\t00000000000001a4:  R_X86_64_PC32\t.data.rscrypto_dispatch-0x4")),
  }
  for target, (address, lines, index, (original, mutated)) in cases.items():
    for listing, expected in ((lines, "conditional_branch"), ([mutated if line == original else line for line in lines],
                                                              "indirect_jump")):
      body = FunctionBody("rscrypto::fixture", Path("fixture.o.disasm.txt"), address, list(enumerate(listing, start=1)))
      assert jump_finding(target, body)["kind"] == expected, (target, expected)
    # The table entry is public read-only data, so the slice reaches the index and no other input.
    body = FunctionBody("rscrypto::fixture", Path("fixture.o.disasm.txt"), address, list(enumerate(lines, start=1)))
    found = jump_finding(target, body)
    analysis = FunctionSlices(target, body, None, {body.symbol})
    result = analysis.slice(found["line"], frozenset(analysis.index_by_line[line] for line in found["table_loads"]))
    assert set(result.leaves) == {index, "incomplete control flow"}, (target, result.leaves, result.canonical)


def merge_table_regressions() -> None:
  """A global-merge unit takes its objects only from an assembly table that matches it exactly."""
  objects = [(0, '.ascii	"ab"', ".Lanon.b.1", 2), (2, '.ascii	"cd"', ".Lanon.b.2", 2)]
  [definition] = assembly_merge_units(merged_assembly(4, objects), True)

  def image(units: list[bytes]) -> LinkedImage:
    data = b"".join(units)
    return LinkedImage(data, [Section(".rodata", 0x1000, len(data), 0, False, False)], True,
                       extents=[(0x1000 + 4 * index, 0x1004 + 4 * index, MERGED) for index in range(len(units))])

  assert merge_unit_members(image([b"abcd"]), [definition]) == [(0x1000, 0x1002), (0x1002, 0x1004)]
  failures = {
    "unmatched": ([b"abce"], [definition]),
    # One table cannot give the objects of two units.
    "ambiguous unit": ([b"abcd", b"abcd"], [definition]),
    # Two tables with the bytes of one unit but different objects.
    "ambiguous table": ([b"abcd"], [definition, *assembly_merge_units(
      merged_assembly(4, [(0, '.ascii	"abcd"', ".Lanon.c.1", 4)], ".L_MergedGlobals.3"), True)]),
  }
  for name, (units, tables) in failures.items():
    try:
      merge_unit_members(image(units), tables)
    except ValueError:
      continue
    raise AssertionError(name)
  # Content that no object holds, or a directive that the reader does not know, is not a table.
  for text in (merged_assembly(4, objects[:1]).replace(".zero\t2", '.ascii\t"cd"'),
               merged_assembly(4, objects).replace('.ascii\t"cd"', ".p2align\t1")):
    try:
      assembly_merge_units(text, True)
    except ValueError:
      continue
    raise AssertionError(text)


def riscv_live_high_part_regression() -> None:
  """A high part whose low part resolves is described even when a call may read its register."""
  target = "riscv64gc-unknown-linux-gnu"
  table = bytes(range(32))

  def build(text: int, rodata: int):
    return linked_fixture(text, [
      f"{text:x}: 00020817\tauipc\ta6, 0x{(rodata - text) >> 12:x}",
      f"{text + 4:x}: 01080513\taddi\ta0, a6, 0x10",
      f"{text + 8:x}: 000000ef\tjal\tra, 0x{text:x} <rscrypto::fixture>",
      f"{text + 0xc:x}: 00008067\tret",
    ], {rodata: table}, units=[(rodata, rodata + 32, None)])

  assert fixture_hash(target, build(0x10000, 0x30000)) == fixture_hash(target, build(0x50000, 0x91000))


def riscv_zero_low_part_regression() -> None:
  """A low part that relinking leaves at offset zero, printed as `mv`, keeps its `addi` identity."""
  target = "riscv64gc-unknown-linux-gnu"
  table = bytes(range(32))

  def build(text: int, rodata: int):
    referenced = rodata + 0x10
    low = referenced & 0xFFF
    second = (f"{text + 4:x}: 00050513\tmv\ta0, a0" if low == 0 else
              f"{text + 4:x}: 01050513\taddi\ta0, a0, 0x{low:x}")
    return linked_fixture(text, [
      f"{text:x}: 00020517\tauipc\ta0, 0x{(referenced - text) >> 12:x}",
      second,
      f"{text + 8:x}: 00008067\tret",
    ], {rodata: table}, units=[(rodata, rodata + 32, None)])

  # 0x30010 and 0x91000: the same table member, with a nonzero and a zero low part.
  assert fixture_hash(target, build(0x10000, 0x30000)) == fixture_hash(target, build(0x50000, 0x90ff0))
  # The identity names the encoded `addi`, not the alias.
  body, image = build(0x50000, 0x90ff0)
  assert "mv" not in "\n".join(instruction_identities(target, body, {body.symbol}, image).values())


def riscv_annotated_jump_table_regression() -> None:
  """A linked `auipc` that llvm-objdump annotates with a nearby symbol still forms its table address.

  `HkdfSha256::extract` in the RISC-V linked evidence binary (d9776202): the `match padded` of `HmacSha256::mac_into`
  jumps through `.LJTI453_0` on the block count (s1 + 0x88) >> 6. Without the annotation the pair resolves, so the
  annotation must not change the identity or turn the table jump into an unknown indirect jump.
  """
  target = "riscv64gc-unknown-linux-gnu"

  def build(text: int, table: int, annotation: str):
    high = table - (text + 8)
    low = high & 0xFFF
    return linked_fixture(text, [
      f"{text:x}: 08848513\taddi\ta0, s1, 0x88",
      f"{text + 4:x}: 8119    \tsrli\ta0, a0, 0x6",
      f"{text + 8:x}: 00000597\tauipc\ta1, 0x{(high >> 12) & 0xFFFFF:x}{annotation}",
      f"{text + 0xc:x}: 050a    \tslli\ta0, a0, 0x2",
      f"{text + 0xe:x}: 00058593\taddi\ta1, a1, 0x{low:x}",
      f"{text + 0x12:x}: 952e    \tadd\ta0, a0, a1",
      f"{text + 0x14:x}: ff852503\tlw\ta0, -0x8(a0)",
      f"{text + 0x18:x}: 952e    \tadd\ta0, a0, a1",
      f"{text + 0x1a:x}: 8502    \tjr\ta0",
    ], {table: bytes(0x14)}, units=[(table, table + 0x14, None)])

  plain = build(0x10000, 0x30170, "")
  annotated = build(0x50000, 0x91170, " <<rscrypto::aead::ascon128::AsconAead128>::finalize+0x753>")
  assert fixture_hash(target, plain) == fixture_hash(target, annotated)
  body, image = annotated
  [found] = [row for row in scan_symbol(target, body.symbol, body, {body.symbol}, scope="entry",
                                        primitive_ids=["kdf.hkdf"], roots=[body.symbol], image=image)
             if row["kind"] in {"conditional_branch", "indirect_jump"}]
  assert found["kind"] == "conditional_branch", found


def main() -> None:
  relocation = "0000000000000060:  R_RISCV_CALL_PLT\trscrypto::auth::ecdsa::sign_digest_p256_blinded"
  assert is_call_relocation(relocation)
  assert is_riscv_conditional_branch("riscv64gc-unknown-linux-gnu", "bnez")
  assert not is_riscv_conditional_branch("x86_64-unknown-linux-gnu", "bnez")
  assert is_rsa_inverse_scope(
    ["rsa.private_ops"],
    "rscrypto::auth::rsa::private_i31_co_reduce_mod",
  )
  assert not is_rsa_inverse_scope(
    ["rsa.private_ops"],
    "rscrypto::auth::rsa::mont_mul_cios_portable",
  )
  assert not is_rsa_inverse_scope(
    ["rsa.private_key_material"],
    "rscrypto::auth::rsa::private_i31_co_reduce_mod",
  )

  callee = "rscrypto::auth::ecdsa::sign_digest_p256_blinded"
  root = FunctionBody(
    symbol="ct_entry_ecdsa_p256_sign",
    path=Path("fixture.disasm.txt"),
    address=0,
    lines=[
      (1, relocation),
      (2, "64: 000080e7\tjalr\tra <ct_entry_ecdsa_p256_sign+0x64>"),
    ],
  )
  functions = {
    root.symbol: root,
    callee: FunctionBody(symbol=callee, path=root.path, address=0x100, lines=[]),
  }
  assert direct_callees(root, functions) == {callee}

  final_root = FunctionBody(
    symbol="ct_entry_owner_eq_16",
    path=Path("final.binary.disasm.txt"),
    address=0,
    lines=[
      (1, "1000: e8 0b 00 00 00\tcallq\t0x1010 <rscrypto::fixed_eq>"),
      (2, "1005: 90\tadrp\tx0, 0x2000 <rscrypto::data::TABLE>"),
    ],
  )
  assert all_direct_callees(final_root) == {"rscrypto::fixed_eq"}

  # The linked ML-DSA mask and rounding roots use AArch64 `b`, not `bl`.
  # Missing that edge would scan only the null guard and silently omit the kernel.
  leaf = "rscrypto::auth::mldsa::diagnostics::diag_mldsa_mask"
  for instruction in ("b", "ba", "j", "jg", "jmp", "jmpq", "c.j"):
    tail_root = FunctionBody(
      symbol="ct_entry_mldsa_mask17", path=final_root.path, address=0,
      lines=[(1, f"1000: 17ffd7f8\t{instruction}\t0x1100 <{leaf}>"),
             (2, f"1004: 17fffffe\t{instruction}\t0x1000 <ct_entry_mldsa_mask17+0x4>")],
    )
    leaf_body = FunctionBody(symbol=leaf, path=final_root.path, address=0x1100,
                             lines=[(3, "1100: 9ac10800\tudiv\tx0, x0, x1")])
    functions = {tail_root.symbol: tail_root, leaf: leaf_body}
    closure = reachable_closure(tail_root.symbol, build_call_graph(functions))
    assert closure == {tail_root.symbol, leaf}
    findings = [finding for name in closure
                for finding in scan_symbol("aarch64-apple-darwin", name, functions[name], set(functions),
                                           scope="ct_intended_call_closure",
                                           primitive_ids=["signature.mldsa.secret_kernels"],
                                           roots=[tail_root.symbol])]
    assert any(row["symbol"] == leaf and row["kind"] == "variable_latency_division"
               and row["severity"] == "fail" for row in findings)

  relocated_tail = FunctionBody(
    symbol="ct_entry_mldsa_mask17", path=Path("fixture.o.disasm.txt"), address=0,
    lines=[(1, "0: 14000000\tb\t0x0 <ct_entry_mldsa_mask17>"),
           (2, f"0: R_AARCH64_JUMP26 {leaf}")],
  )
  assert all_direct_callees(relocated_tail) == {leaf}

  # Mach-O relocations retain mangled names even with llvm-objdump -C.
  # The P-256 root used to appear to have no production callees on Darwin.
  with TemporaryDirectory() as temporary:
    artifact_dir = Path(temporary)
    callee = "<rscrypto::auth::p256_ecdh::P256EphemeralSecret>::diffie_hellman"
    mangled = "__RNvM_fixture_P256EphemeralSecret_diffie_hellman"
    (artifact_dir / "fixture.o.raw-symbols.txt").write_text(f"00000100 t {mangled}\n")
    (artifact_dir / "fixture.o.symbols.txt").write_text(f"00000100 t {callee}\n")
    names = demangle_table(artifact_dir)
    artifact = artifact_dir / "fixture.o.disasm.txt"
    for relocation_kind, instruction in (
      ("ARM64_RELOC_BRANCH26", "bl"),
      ("X86_64_RELOC_BRANCH", "callq"),
      ("IMAGE_REL_AMD64_REL32", "callq"),
    ):
      artifact.write_text(
        "00000000 <_ct_entry_p256_ecdh_agree>:\n"
        f"  0: 94000000\t{instruction}\t0x0 <_ct_entry_p256_ecdh_agree>\n"
        f"  0: {relocation_kind}\t{mangled}\n"
        "  4: d65f03c0\tret\n"
        f"00000100 <{callee}>:\n"
        "100: d65f03c0\tret\n"
      )
      functions, _ = parse_disassembly(artifact, names)
      assert reachable_closure("ct_entry_p256_ecdh_agree", build_call_graph(functions)) == {
        "ct_entry_p256_ecdh_agree", callee,
      }

    # The next section's crate disambiguator is not part of this function.
    hashes = []
    for next_section in ("crate_hash_one", "crate_hash_two"):
      artifact.write_text(
        "Disassembly of section .text.ct_entry_p256_ecdh_agree:\n"
        "00000000 <ct_entry_p256_ecdh_agree>:\n"
        "  0: 54000040\tb.eq\t0x8 <ct_entry_p256_ecdh_agree+0x8>\n"
        "  4: d503201f\tnop\n"
        "  8: d65f03c0\tret\n"
        f"Disassembly of section .text.{next_section}:\n"
      )
      functions, _ = parse_disassembly(artifact)
      root = functions["ct_entry_p256_ecdh_agree"]
      findings = scan_symbol("aarch64-apple-darwin", root.symbol, root, set(functions),
                             scope="entry", primitive_ids=["kx.p256_ecdh"], roots=[root.symbol])
      hashes.append(findings[0]["function_sha256"])
    assert hashes[0] == hashes[1]

    # Object-local outlined names are reused for different bodies.
    combined = {}
    for name in ("one", "two"):
      outlined = artifact_dir / f"{name}.o.disasm.txt"
      outlined.write_text(
        f"00000000 <ct_entry_{name}>:\n"
        "  0: 94000001\tbl\t0x4 <_OUTLINED_FUNCTION_0>\n"
        "00000004 <_OUTLINED_FUNCTION_0>:\n"
        f"  4: 94000001\tbl\t0x8 <rscrypto::{name}>\n"
        f"00000008 <rscrypto::{name}>:\n"
        "  8: d65f03c0\tret\n"
      )
      parsed, _ = parse_disassembly(outlined)
      combined.update(parsed)
    for name in ("one", "two"):
      assert reachable_closure(f"ct_entry_{name}", build_call_graph(combined)) == {
        f"ct_entry_{name}", f"{name}.o.disasm.txt::OUTLINED_FUNCTION_0", f"rscrypto::{name}",
      }

    artifact.write_text(
      "00000000 <ct_entry_p256_ecdh_agree>:\n"
      "0: 94000008\tbl\t0x20 <.text+0x20>\n"
      "4: d65f03c0\tret\n"
    )
    functions, _ = parse_disassembly(artifact)
    unresolved = linked_closure({"ct_entry_p256_ecdh_agree"}, functions, artifact)
    assert unresolved["unresolved_internal_calls"] == [{"caller": "ct_entry_p256_ecdh_agree", "target": ".text"}]
    for section in (".text", ".plt", ".glink"):
      artifact.write_text(
        "00000000 <ct_entry_p256_ecdh_agree>:\n"
        f"0: 21 00 00 48\tbl 0x20 <{section}+0x20>\n"
        "4: 20 00 80 4e\tblr\n"
      )
      functions, _ = parse_disassembly(artifact)
      unresolved = linked_closure({"ct_entry_p256_ecdh_agree"}, functions, artifact, "powerpc64le-unknown-linux-gnu")
      assert unresolved["unresolved_internal_calls"] == [{"caller": "ct_entry_p256_ecdh_agree", "target": section}]

    # Missing Linux ELF GOT evidence must not quietly shrink the linked closure.
    # Register dispatch and Windows IAT transfers retain their existing policy.
    for transfer in ("callq *0x6852a(%rip) # 0x165150", "jmpq *0x6852a(%rip) # 0x165150"):
      opcode = "15" if transfer.startswith("call") else "25"
      for annotation, unresolved_count in (("", 1), (" <.got+0x10>", 1), (" <memcpy@plt>", 1)):
        artifact.write_text(
          "000fcc20 <ct_entry_p256_ecdh_agree>:\n"
          f"fcc20: ff {opcode} 2a 85 06 00\t{transfer}{annotation}\n"
        )
        functions, _ = parse_disassembly(artifact)
        result = linked_closure({"ct_entry_p256_ecdh_agree"}, functions, artifact, "x86_64-unknown-linux-gnu")
        assert len(result["unresolved_internal_calls"]) == unresolved_count
        result = linked_closure({"ct_entry_p256_ecdh_agree"}, functions, artifact, "x86_64-pc-windows-msvc")
        assert result["unresolved_internal_calls"] == []
      artifact.write_text(artifact.read_text().replace("# ", "# GOT "))
      functions, _ = parse_disassembly(artifact)
      result = linked_closure({"ct_entry_p256_ecdh_agree"}, functions, artifact, "x86_64-unknown-linux-gnu")
      assert result["unresolved_internal_calls"] == []
    artifact.write_text("000fcc20 <ct_entry_p256_ecdh_agree>:\nfcc20: ff d0\tcallq *%rax\n")
    functions, _ = parse_disassembly(artifact)
    assert linked_closure({"ct_entry_p256_ecdh_agree"}, functions, artifact, "x86_64-unknown-linux-gnu")["unresolved_internal_calls"] == []

  # Key import must stay outside the linked enforced closure. A reached symbol
  # reports its call path; an absent symbol cannot satisfy the check.
  linked = Path("linked.binary.disasm.txt")
  import_symbol = "<rscrypto::auth::rsa::RsaPrivateKey>::from_pkcs8_der_with_policy"
  linked_functions = {
    "ct_entry_rsa_pss_sign_fixed_blinding": FunctionBody(
      "ct_entry_rsa_pss_sign_fixed_blinding", linked, 0x1000,
      [(1, "1000: 94000400\tbl\t0x2000 <rscrypto::auth::rsa::private_operation>")]),
    "rscrypto::auth::rsa::private_operation": FunctionBody(
      "rscrypto::auth::rsa::private_operation", linked, 0x2000, [(2, "2000: d65f03c0\tret")]),
    "ct_entry_rsa_fixture_key_import": FunctionBody(
      "ct_entry_rsa_fixture_key_import", linked, 0x3000,
      [(3, f"3000: 14000400\tb\t0x4000 <{import_symbol}>")]),
    import_symbol: FunctionBody(import_symbol, linked, 0x4000, [(4, "4000: d65f03c0\tret")]),
  }
  roots = {"ct_entry_rsa_pss_sign_fixed_blinding"}
  unreachable = ["ct_entry_rsa_fixture_key_import", import_symbol]
  assert unreachable_symbol_errors(roots, unreachable, linked_functions) == []
  assert unreachable_symbol_errors(roots, ["rscrypto::auth::rsa::renamed_import"], linked_functions) == [
    "unreachable symbol rscrypto::auth::rsa::renamed_import is absent from the linked binary",
  ]
  linked_functions["rscrypto::auth::rsa::private_operation"].lines = [
    (2, f"2000: 94000800\tbl\t0x4000 <{import_symbol}>")]
  assert unreachable_symbol_errors(roots, unreachable, linked_functions) == [
    f"enforced CT closure reaches {import_symbol}: ct_entry_rsa_pss_sign_fixed_blinding -> "
    f"rscrypto::auth::rsa::private_operation -> {import_symbol}",
  ]

  # Identity is independent only of function placement and resolved direct
  # transfer displacement. Every condition, operand, destination and width stays.
  target = "aarch64-unknown-linux-gnu"
  body = FunctionBody("caller", Path("linked.binary.disasm.txt"), 0x1000, [])
  moved = FunctionBody("caller", body.path, 0x3000, [])
  known = {"caller", "callee", "other"}
  identity = lambda line: instruction_identity(target, body, line, known)
  original = identity("1000: b7000041\ttbnz\tx1, #0x20, 0x1008 <caller+0x8>")
  assert original == instruction_identity(
    target, moved, "3000: b7000041\ttbnz\tx1, #0x20, 0x3008 <caller+0x8>", known,
  )
  for mutation in (
    "1000: b6000041\ttbz\tx1, #0x20, 0x1008 <caller+0x8>",
    "1000: b7000042\ttbnz\tx2, #0x20, 0x1008 <caller+0x8>",
    "1000: b7080041\ttbnz\tx1, #0x21, 0x1008 <caller+0x8>",
    "1000: b7000061\ttbnz\tx1, #0x20, 0x100c <caller+0xc>",
    "1000: b7000041 0000\ttbnz\tx1, #0x20, 0x1008 <caller+0x8>",
  ):
    assert original != identity(mutation)
  call = identity("1000: 94000400\tbl\t0x2000 <callee>")
  assert call == instruction_identity(target, moved, "3000: 94000800\tbl\t0x5000 <callee>", known)
  assert call != identity("1000: 94000400\tbl\t0x2000 <other>")
  assert identity("1000: 90000000\tadrp\tx0, 0x4000") != identity("1000: 90000020\tadrp\tx0, 0x5000")
  assert identity("1000: d2800020\tmov\tx0, #1") != identity("1000: d2800040\tmov\tx0, #2")
  assert identity("1000: 94000400\tbl\t0x2000") != identity("1000: 94000800\tbl\t0x3000")
  # x86 `jrcxz` is a resolved direct transfer too: moving its function keeps the identity.
  target = "x86_64-unknown-linux-gnu"
  counted = instruction_identity(target, body, "1006: e3 01\tjrcxz\t0x1009 <caller+0x9>", known)
  assert counted == instruction_identity(target, moved, "3006: e3 01\tjrcxz\t0x3009 <caller+0x9>", known)
  assert counted != instruction_identity(target, body, "1006: e3 02\tjrcxz\t0x100a <caller+0xa>", known)
  immediate_symbol_regressions(body, known)
  linker_stub_regressions()
  multiply_regressions()
  select_branch_regressions()
  public_input_regressions()
  constant_condition_regressions()
  data_reference_regressions()
  exrl_template_regressions()
  placement_regressions()


if __name__ == "__main__":
  main()
