---
"rscrypto" = "patch"
---

Rework the ML-DSA kernels on Linux x86-64 and AArch64. The AVX2 and NEON number-theoretic transforms use signed
Montgomery arithmetic with lazy reduction and merged stages; signing hashes its mask polynomials two SHAKE256
streams at a time; vector backends run the norm checks, HighBits, the low part of Decompose and the hint loop;
and matrix expansion pairs SHAKE128 streams on Linux as it already did on macOS. Signatures and verification
results are unchanged.
