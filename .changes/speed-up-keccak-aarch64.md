---
"rscrypto" = "patch"
---

Speed up SHA-3, SHAKE, cSHAKE, KMAC, and HMAC-SHA-3 by 16–24% on non-Apple AArch64, measured on
Graviton4. The scalar Keccak permutation applies every ρ rotation through the rotated operand of an
`EOR` or `BIC` instead of rotating lanes separately, and its 24 rounds are unrolled. ML-KEM and ML-DSA
run 2–6% faster there. Apple targets keep their existing Keccak paths.
