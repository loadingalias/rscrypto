---
"rscrypto" = "patch"
---

RSA Montgomery arithmetic on AArch64 now uses rscrypto's own assembly instead of kernels adapted from AWS-LC, and AArch64 Windows gains the assembly path. Results are unchanged; on an Apple M1 Pro, RSA-4096 signing is about 1% faster and RSA-3072 signing about 2% slower than with the adapted kernels.
