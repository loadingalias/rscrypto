---
"rscrypto" = "patch"
---

Speed up SHA-3, SHAKE, cSHAKE, KMAC, and HMAC-SHA-3 on Intel x86-64 CPUs with AVX-512. The single-state
Keccak permutation runs in AVX-512 registers and absorbs complete blocks without storing the state between
them. On Granite Rapids, SHA-3 of 1 KiB to 1 MiB runs 11–13% faster, 1.05–1.08x AWS-LC instead of
0.95–0.98x; on Sapphire Rapids it runs 3–4% faster. AMD CPUs keep the BMI kernel, which is faster there.
