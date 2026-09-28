---
"rscrypto" = "patch"
---

Protect POWER ML-DSA scalar selections and sampler masks from compiler-created
secret-dependent branches. Apply the register barrier to native and portable-only
builds without changing arithmetic, sampling order, or timing-test requirements.
Blind the POWER vector forward NTT of secret polynomials with a mask derived from
the secret seed, so repeated small coefficients never reach the vector multiplier.
Signatures and keys are unchanged.
