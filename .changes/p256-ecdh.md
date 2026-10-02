---
"rscrypto" = "minor"
---

Add standalone `no_std` P-256 ephemeral Diffie-Hellman under the `p256-ecdh` feature.
It validates canonical uncompressed SEC1 peer points,
generates scalars with a bounded fallible loop, and uses zeroizing secret owners that are not `Clone`.
Assembly derived from s2n-bignum accelerates it on Apple and Linux AArch64,
and on Linux and Windows x86-64 (baseline and BMI2/ADX).
RV64 uses accelerated field arithmetic.
Other targets use the safe Rust implementation.
