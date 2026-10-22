---
"rscrypto" = "patch"
---

Speed up ML-DSA signing and verification. Verification decodes and bounds the response once and inverts one
number-theoretic transform per row, encodings pack fixed-width groups, signing hashes its mask polynomials two
SHAKE256 streams at a time, and matrix expansion pairs SHAKE128 streams, also across rows.

On Linux x86-64, the AVX2 transforms use signed Montgomery arithmetic with lazy reduction and merged stages.
On macOS and Linux AArch64, NEON runs Barrett-form transforms, the pointwise products, matrix rejection
sampling, hint application, and a fused row product for prepared signing. Both vector backends run the norm
checks, HighBits, the low part of Decompose, the hint loop, and the constant-time SampleInBall scans for each
challenge candidate and its placement. On Intel Granite Rapids, the AVX2 SampleInBall scans alone made
ML-DSA-65 signing about 20% faster and an ML-DSA-65 TLS 1.3 handshake about 12% faster.
Signatures and verification results are unchanged.
