---
"rscrypto" = "patch"
---

Add independently implemented four-lane BLAKE3 chunk hashing, four-parent reduction, and equal-length
digest batching for SIMD128 WebAssembly. Keep serial compression portable and
preserve the scalar and `portable-only` fallbacks. Specialize plain tiny hashes
without changing keyed or derive-key cleanup. Reuse round state for vector chaining
values to reduce guest scratch storage while preserving full remaining cleanup.
Skip padding wipes for complete blocks that never populate that scratch, while
retaining complete cleanup for secret partial blocks.
Reduce batch overhead by specializing plain roots separately from full chunks,
preserving keyed and derive-key cleanup.
