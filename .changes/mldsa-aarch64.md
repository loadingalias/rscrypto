---
"rscrypto" = "patch"
---

Accelerate ML-DSA forward and inverse transforms with NEON on macOS AArch64, preserving
portable fallback and byte-for-byte results. Fuse portable forward-transform
conversion with the first butterfly to remove redundant arithmetic.

Parallelize fixed-work ML-DSA challenge selection with a minimum reduction while
preserving secret-independent memory access and bounded failure behavior.

Accelerate matrix-product accumulation with four-lane NEON on macOS AArch64.

Batch public matrix expansion through paired SHAKE on macOS AArch64. Preserve
scalar expansion for portable-only and other targets, and keep private polynomial
cleanup unchanged. The accelerated row path uses one additional public polynomial
buffer.
