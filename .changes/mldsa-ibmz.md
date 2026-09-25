---
"rscrypto" = "patch"
---

Add original ML-DSA arithmetic for Linux IBM Z and little-endian POWER8 with
CPU and OS capability checks, and for RISC-V32/RISC-V64 with compile-time M
support. Share the vector transform schedule and preserve the portable fallback,
public APIs, encodings, canonical coefficients, and secret-owner cleanup.

Keep scalar ML-DSA reduction and selection masks opaque to the compiler on IBM Z
and RISC-V, including portable-only builds, without forcing masks through memory.
Protect the challenge sampler's secret-candidate minimum on these targets too.
