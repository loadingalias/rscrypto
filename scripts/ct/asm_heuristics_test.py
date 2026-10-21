#!/usr/bin/env -S uv run python
"""Regression tests for target-specific CT disassembly parsing."""

from pathlib import Path
from tempfile import TemporaryDirectory

from asm_heuristics import (
  FunctionBody,
  all_direct_callees,
  apply_select_branch_enforcement,
  build_call_graph,
  direct_callees,
  final_equality_closure,
  is_call_relocation,
  is_conditional_branch,
  is_riscv_conditional_branch,
  is_rsa_inverse_scope,
  instruction_identity,
  parse_disassembly,
  reachable_closure,
  scan_symbol,
  unreachable_symbol_errors,
)
from symbolize_linked_binary import demangle_table


def branch_kinds(target: str, symbol: str, lines: list[str], primitive_ids: list[str]) -> list[tuple[str, str]]:
  body = FunctionBody(symbol=symbol, path=Path("fixture.o.disasm.txt"), address=0,
                      lines=list(enumerate(lines, start=1)))
  findings = scan_symbol(target, symbol, body, {symbol}, scope="ct_intended_call_closure",
                         primitive_ids=primitive_ids, roots=[symbol])
  apply_select_branch_enforcement(findings, {"kx.p256_ecdh"})
  return [(row["kind"], row["severity"]) for row in findings
          if row["kind"] in {"conditional_branch", "select_branch"}]


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
    unresolved = final_equality_closure({"ct_entry_p256_ecdh_agree"}, functions, artifact)
    assert unresolved["unresolved_internal_calls"] == [{"caller": "ct_entry_p256_ecdh_agree", "target": ".text"}]
    for section in (".text", ".plt", ".glink"):
      artifact.write_text(
        "00000000 <ct_entry_p256_ecdh_agree>:\n"
        f"0: 21 00 00 48\tbl 0x20 <{section}+0x20>\n"
        "4: 20 00 80 4e\tblr\n"
      )
      functions, _ = parse_disassembly(artifact)
      unresolved = final_equality_closure({"ct_entry_p256_ecdh_agree"}, functions, artifact, "powerpc64le-unknown-linux-gnu")
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
        result = final_equality_closure({"ct_entry_p256_ecdh_agree"}, functions, artifact, "x86_64-unknown-linux-gnu")
        assert len(result["unresolved_internal_calls"]) == unresolved_count
        result = final_equality_closure({"ct_entry_p256_ecdh_agree"}, functions, artifact, "x86_64-pc-windows-msvc")
        assert result["unresolved_internal_calls"] == []
      artifact.write_text(artifact.read_text().replace("# ", "# GOT "))
      functions, _ = parse_disassembly(artifact)
      result = final_equality_closure({"ct_entry_p256_ecdh_agree"}, functions, artifact, "x86_64-unknown-linux-gnu")
      assert result["unresolved_internal_calls"] == []
    artifact.write_text("000fcc20 <ct_entry_p256_ecdh_agree>:\nfcc20: ff d0\tcallq *%rax\n")
    functions, _ = parse_disassembly(artifact)
    assert final_equality_closure({"ct_entry_p256_ecdh_agree"}, functions, artifact, "x86_64-unknown-linux-gnu")["unresolved_internal_calls"] == []

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
  select_branch_regressions()


if __name__ == "__main__":
  main()
