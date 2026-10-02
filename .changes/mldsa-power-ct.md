---
"rscrypto" = "patch"
---

Protect POWER ML-DSA scalar selections and sampler masks from secret-dependent branches
that the compiler creates.
The register barrier applies to native and `portable-only` builds.
Arithmetic, sampling order, and timing-test requirements do not change.

Blind the POWER vector forward NTT of secret polynomials with a mask derived from the secret seed,
so that repeated small coefficients never reach the vector multiplier.
Signatures and keys do not change.
