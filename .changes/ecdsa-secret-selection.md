---
"rscrypto" = "patch"
---

Preserve masked ECDSA point selection on AArch64 and Windows
and masked secret selection in portable P-256.
Use fixed-bound ECDSA table traversal
and retain unconditional RISC-V generator-table loads under LLVM optimization without changing
signature semantics.
