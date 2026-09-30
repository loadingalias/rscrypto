---
"rscrypto" = "patch"
---

Remove secret-dependent branches from Ed25519 scalar arithmetic on RISC-V.
Checked 128-bit additions on secret-derived carries compiled to branches on
RV64GC; the bounded arithmetic now uses wrapping operations. Signatures and
public APIs are unchanged.
