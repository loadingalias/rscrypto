---
"rscrypto" = "patch"
---

Speed up the single-state Keccak permutation on x86-64 CPUs with BMI1 and BMI2 that do not run the AVX-512
kernel, such as AMD CPUs and Intel CPUs without AVX-512: runtime dispatch runs the portable rounds compiled
with `RORX` and `ANDN`. On Intel Granite Rapids, before the AVX-512 kernel replaced it there, KMAC, cSHAKE,
and HMAC-SHA-3 ran 7–9% faster and SHAKE about 3% faster. AMD Zen 5 is unchanged.
