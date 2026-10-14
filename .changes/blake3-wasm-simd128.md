---
"rscrypto" = "patch"
---

Speed up BLAKE3 on WebAssembly builds that enable `simd128` at compile time.
Chunk hashing, parent reduction, and equal-length `digest_batch` runs use four
SIMD128 lanes. Builds without `simd128`, and `portable-only` builds, keep the
portable path. Digests and keyed and derive-key cleanup are unchanged.
