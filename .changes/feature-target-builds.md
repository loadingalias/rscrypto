---
"rscrypto" = "patch"
---

Restore Rust 1.91 Ed25519 and X25519 builds on x86-64 Linux when assembly owns fixed-base dispatch,
and keep standalone AEAD features lint-clean on Linux.
Exclude SIMD backends from scalar WebAssembly hash, AEAD,
and Argon2 dispatch when SIMD128 is disabled.
