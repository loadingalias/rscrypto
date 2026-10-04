---
"rscrypto" = "patch"
---

Precompute the AVX2 and IFMA fixed-base tables used by the Rust Ed25519 and X25519 backends, trading 160 KiB of read-only data for faster signing and public-key derivation. Keep both table selectors in separate AVX2 functions for binary constant-time verification.
