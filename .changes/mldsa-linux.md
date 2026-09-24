---
"rscrypto" = "patch"
---

Accelerate ML-DSA transforms and matrix-product accumulation with runtime-gated
AVX2 on Linux x86-64 and compile-time NEON on Linux AArch64. Preserve portable
fallback, public APIs, encodings, and byte-for-byte results. Keep the Linux NEON
inverse middle stage out of line to prevent coefficient spills observed in the
reviewed build.
