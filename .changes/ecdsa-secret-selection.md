---
"rscrypto" = "patch"
---

Keep masked ECDSA point selection on AArch64 and Windows,
and masked secret selection in portable P-256.
Use ECDSA table traversal with fixed bounds.
Keep RISC-V generator-table loads unconditional under LLVM optimization.
Signature behavior does not change.

Make portable P-256 public derivation and agreement faster with RV64-shaped multiplication,
sparse field arithmetic, and shared fixed-base tables.
ECDH and signature behavior do not change.
