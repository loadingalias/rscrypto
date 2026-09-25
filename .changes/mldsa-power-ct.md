---
"rscrypto" = "patch"
---

Protect POWER ML-DSA scalar selections and sampler masks from compiler-created
secret-dependent branches. Apply the register barrier to native and portable-only
builds without changing arithmetic, sampling order, or timing-test requirements.
