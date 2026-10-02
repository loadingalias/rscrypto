---
"rscrypto" = "minor"
---

Add `Blake3::digest_batch`, which hashes many independent inputs in one call.
Runs of inputs with equal length, from 1 to 1,024 bytes, share SIMD lanes
(SSE4.1, AVX2, AVX-512, and NEON).
Other inputs, and targets without those kernels, use the one-shot path.
For every input, the output is equal to `Blake3::digest`.
