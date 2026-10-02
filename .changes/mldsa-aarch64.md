---
"rscrypto" = "patch"
---

Make ML-DSA forward and inverse transforms faster with NEON on macOS AArch64.
The portable fallback and byte-for-byte results do not change.
Merge the portable forward-transform conversion into the first butterfly to remove repeated
arithmetic.

Make fixed-work ML-DSA challenge selection parallel, with a minimum reduction.
Memory access stays independent of secrets, and failure stays bounded.

Make matrix-product accumulation faster with four-lane NEON on macOS AArch64.

Batch public matrix expansion through paired SHAKE on macOS AArch64.
`portable-only` and other targets keep scalar expansion.
Private polynomial cleanup does not change.
The accelerated row path uses one more public polynomial buffer.
