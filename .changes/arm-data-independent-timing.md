---
"rscrypto" = "minor"
---

On AArch64 cores with `FEAT_DIT`, set Arm data-independent timing (`PSTATE.DIT`) during X25519, Ed25519, ECDSA,
P-256 and P-384 ECDH, ML-KEM, ML-DSA, RSA private operations, Argon2, scrypt, and PBKDF2.
The caller's state is restored after each operation.
Add `traits::ct::with_data_independent_timing`, which sets DIT for a scope that the caller selects.
Use it to cover short symmetric operations without a toggle on each call.
