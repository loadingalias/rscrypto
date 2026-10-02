---
"rscrypto" = "patch"
---

Remove secret-dependent branches from Ed25519 scalar arithmetic on RISC-V.
On RV64GC, checked 128-bit additions on secret-derived carries compiled to branches.
The bounded arithmetic now uses wrapping operations.
Signatures and public APIs do not change.
