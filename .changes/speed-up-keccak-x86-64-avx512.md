---
"rscrypto" = "patch"
---

Speed up SHA-3, SHAKE, cSHAKE, KMAC, and HMAC-SHA-3 on Intel x86-64 CPUs with AVX-512. The single-state
Keccak permutation runs in AVX-512 registers and absorbs complete blocks without storing the state between
them. On Granite Rapids, SHA-3 of 1 KiB to 1 MiB runs 11–13% faster, 1.05–1.08x AWS-LC instead of
0.95–0.98x; on Sapphire Rapids it runs 3–4% faster. Other x86-64 CPUs with BMI1 and BMI2, such as AMD CPUs
and Intel CPUs without AVX-512, run the portable rounds compiled with `RORX` and `ANDN`. On Granite Rapids,
that kernel ran KMAC, cSHAKE, and HMAC-SHA-3 7–9% faster and SHAKE about 3% faster than the previous rounds.
AMD Zen 5 is unchanged, and AMD CPUs do not run the AVX-512 kernel, which is slower there.
