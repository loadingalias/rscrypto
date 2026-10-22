---
"rscrypto" = "patch"
---

RSA Montgomery arithmetic on AArch64 now uses rscrypto's own assembly instead of kernels adapted from AWS-LC, and AArch64 Windows gains the assembly path. Results are unchanged.
