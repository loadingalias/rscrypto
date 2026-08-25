---
"rscrypto" = "patch"
---

Harden ECDSA limb arithmetic, endian conversion, diagnostic documentation, and fixed-work RISC-V and s390x multiplication while preserving P-256 and P-384 signature semantics. On s390x, wide nonce reduction now avoids secret-fed multiplication, while caller blinding masks projective, inversion, and final order arithmetic against operand-dependent timing.
