---
"rscrypto" = "patch"
---

Add original ML-DSA arithmetic for Linux IBM Z and little-endian POWER8,
with CPU and OS capability checks.
Add it also for RISC-V32 and RISC-V64 with compile-time M support.
The targets share the vector transform schedule.
The portable fallback, public APIs, encodings, canonical coefficients,
and secret-owner cleanup do not change.

Keep scalar ML-DSA reduction and selection masks opaque to the compiler on IBM Z and RISC-V,
including `portable-only` builds, without forcing the masks through memory.
Protect the secret-candidate minimum of the challenge sampler on these targets too.
