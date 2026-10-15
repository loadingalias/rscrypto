---
"rscrypto" = "patch"
---

Remove compiler-introduced secret branches from ML-KEM implicit rejection on RISC-V, ECDSA arithmetic on RISC-V and IBM Z, and Curve25519 field normalization on those targets. Replace the Curve25519 zero test's early exit with a fixed limb reduction. Signing, key agreement, and decapsulation results are unchanged.
