---
"rscrypto" = "patch"
---

Keep Ed25519 and X25519 building on x86-64 Linux with every supported compiler
when the assembly backend owns fixed-base dispatch.
Keep standalone AEAD features lint-clean on Linux.
When SIMD128 is disabled, exclude SIMD backends from scalar WebAssembly hash, AEAD,
and Argon2 dispatch.
