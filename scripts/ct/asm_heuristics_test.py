#!/usr/bin/env python3
"""Regression tests for target-specific CT disassembly parsing."""

from pathlib import Path

from asm_heuristics import FunctionBody, all_direct_callees, direct_callees, is_call_relocation, is_riscv_conditional_branch


def main() -> None:
  relocation = "0000000000000060:  R_RISCV_CALL_PLT\trscrypto::auth::ecdsa::sign_digest_p256_blinded"
  assert is_call_relocation(relocation)
  assert is_riscv_conditional_branch("riscv64gc-unknown-linux-gnu", "bnez")
  assert not is_riscv_conditional_branch("x86_64-unknown-linux-gnu", "bnez")

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


if __name__ == "__main__":
  main()
