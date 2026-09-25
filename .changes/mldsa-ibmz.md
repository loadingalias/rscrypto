---
"rscrypto" = "patch"
---

Add original z/Vector ML-DSA transforms, products, and accumulation on Linux IBM Z
with CPU and OS capability checks and the existing portable fallback. Preserve
public APIs, encodings, canonical coefficients, and secret-owner cleanup.

Keep scalar ML-DSA reduction and selection masks opaque to the compiler on IBM Z
and RISC-V, including portable-only builds, without forcing masks through memory.
