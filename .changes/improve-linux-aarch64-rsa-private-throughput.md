---
"rscrypto" = "patch"
---

Reduce Linux AArch64 RSA private-operation overhead with paired Montgomery strip loads and fixed
five-bit exponent windows that reuse existing scratch capacity. Keep the original scalar kernels
and four-bit exponent policy on other targets.
