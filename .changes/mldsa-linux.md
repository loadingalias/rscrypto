---
"rscrypto" = "patch"
---

Make ML-DSA transforms and matrix-product accumulation faster: with AVX2, selected at run time,
on Linux x86-64, and with compile-time NEON on Linux AArch64.
The portable fallback, public APIs, encodings, and byte-for-byte results do not change.
The Linux NEON inverse middle stage stays out of line,
to prevent the coefficient spills seen in the reviewed build.
