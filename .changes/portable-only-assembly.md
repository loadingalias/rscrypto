---
"rscrypto" = "patch"
---

With the `portable-only` feature, or under Miri, ECDSA no longer calls its AArch64 and x86-64 assembly.
It uses the portable Rust implementation, as other targets do.
HKDF-SHA256 uses its AArch64 SHA2 single-block path only when the build enables `target_feature = "sha2"` and not
`portable-only`.
