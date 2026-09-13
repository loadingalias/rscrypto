---
"rscrypto" = "minor"
---

Limit ordinary `diag` builds to capability and backend-selection introspection. Remove benchmark,
forced-kernel, constant-time, zeroization, and component operations from ordinary public module paths,
re-exports, and associated methods across AEADs, MACs, KDFs, password hashing, signatures, key agreement,
RSA, ML-KEM, and cryptographic hashes. This includes RSA seeded diagnostic encryption methods,
BLAKE3 diagnostic selectors, the SHA-256 benchmark compression helper, and the diagnostic-only
`Argon2Error::BackendUnavailable` variant. Removing that variant changes the numeric discriminant of
`VerificationLimitTooLow` in ordinary `diag` builds with `phc-strings`.
Repository evidence tools retain explicit internal access. Internal PBKDF2 verification probes exercise
the primitive instead of rejecting their fixed parameters through the application password policy.
Application cryptographic operations are unchanged by this boundary cleanup.
