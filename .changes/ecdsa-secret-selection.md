---
"rscrypto" = "patch"
---

Preserve masked ECDSA point selection on AArch64 and Windows
and masked secret selection in portable P-256.
Use fixed-bound ECDSA table traversal
and retain unconditional RISC-V generator-table loads under LLVM optimization without changing
signature semantics.
Accelerate portable P-256 public derivation and agreement with target-shaped RV64 multiplication,
sparse field arithmetic, and shared fixed-base tables without changing ECDH or signature semantics.
