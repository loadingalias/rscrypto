---
"rscrypto" = "minor"
---

Set Arm data-independent timing (`PSTATE.DIT`) during X25519, Ed25519, ECDSA,
P-256 and P-384 ECDH, ML-KEM, ML-DSA, RSA private, Argon2, scrypt, and PBKDF2
operations on AArch64 cores with `FEAT_DIT`, restoring the caller's state
afterwards. Add `traits::ct::with_data_independent_timing`, which sets it for a
caller-chosen scope, so short symmetric operations can be covered without a
per-call toggle.
