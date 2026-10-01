---
"rscrypto" = "minor"
---

Add `Blake3::digest_batch`, which hashes many independent inputs in one call.
Runs of equal-length inputs of 1 to 1,024 bytes share SIMD lanes (SSE4.1, AVX2,
AVX-512, and NEON); other inputs, and targets without those kernels, take the
one-shot path. Outputs equal `Blake3::digest` for every input.
