#!/usr/bin/env python3
"""Regression tests for target-specific CT disassembly parsing."""

from pathlib import Path

from asm_heuristics import (
  FunctionBody,
  all_direct_callees,
  build_call_graph,
  direct_callees,
  is_call_relocation,
  is_riscv_conditional_branch,
  is_rsa_inverse_scope,
  reachable_closure,
  scan_symbol,
)


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


if __name__ == "__main__":
  main()
