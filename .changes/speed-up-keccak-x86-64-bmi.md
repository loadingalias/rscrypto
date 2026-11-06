---
"rscrypto" = "patch"
---

Speed up KMAC, cSHAKE, and HMAC-SHA-3 by 7–9% on Intel Granite Rapids. On x86-64 CPUs with BMI1 and
BMI2, runtime dispatch runs the single-state Keccak permutation compiled with `RORX` and `ANDN`.
SHAKE is about 3% faster there and one-shot SHA-3 about 1%. AMD Zen 5 is unchanged.
