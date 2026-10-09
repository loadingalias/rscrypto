---
"rscrypto" = "patch"
---

Speed up BLAKE3 streaming updates that end inside a chunk after complete chunks, at
chunk counters divisible by 16. One SIMD pass now hashes those chunks and the leading
blocks of the partial chunk, and their chaining values reduce together; a later update or
finalization resumes the partial chunk. NEON covers 3, 7, 11 or 15 complete chunks;
AVX-512 covers 3 or 15. Digest bytes and public APIs are unchanged, and keyed and
derive-key scratch is cleared.
