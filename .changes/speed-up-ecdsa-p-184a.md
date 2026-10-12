---
"rscrypto" = "patch"
---

Speed up ECDSA P-256 and P-384 signature verification by taking the
generator term from the existing fixed-base signing table, with unchanged
keys and results.
