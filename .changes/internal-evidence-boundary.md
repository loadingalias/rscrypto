---
"rscrypto" = "minor"
---

Limit ordinary `diag` builds to capability and backend-selection introspection.
Remove benchmark, forced-kernel, constant-time, zeroization,
and component operations from ordinary public module paths, re-exports, and associated methods.
This covers AEADs, MACs, KDFs, password hashing, signatures, key agreement, RSA, ML-KEM,
and cryptographic hashes.
It includes the RSA seeded diagnostic encryption methods, the BLAKE3 diagnostic selectors,
the SHA-256 benchmark compression helper, and the diagnostic-only `Argon2Error::BackendUnavailable` variant.
Removing that variant changes the numeric discriminant of `VerificationLimitTooLow` in ordinary `diag` builds with `phc-strings`.

Repository evidence tools keep explicit internal access.
Internal PBKDF2 verification probes now exercise the primitive,
instead of being rejected by the application password policy because of their fixed parameters.
Application cryptographic operations do not change.
