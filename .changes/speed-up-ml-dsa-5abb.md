---
"rscrypto" = "patch"
---

Speed up ML-DSA signing and verification. Verification decodes and bounds the response once and inverts one number-theoretic transform per row, encodings pack fixed-width groups, and matrix expansion pairs SHAKE128 streams across rows. On AArch64, NEON also runs the pointwise products, Barrett-form transforms, matrix rejection sampling, hint application, a fused row product for prepared signing, and the constant-time challenge sampler. Signatures and verification results are unchanged.
