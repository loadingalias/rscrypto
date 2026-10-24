---
"rscrypto" = "patch"
---

Speed up standalone `Poly1305`. On 64-bit targets it now keeps a radix-2^64 accumulator and passes all
full blocks of each update to one kernel call. Short inputs take a scalar kernel. Longer inputs take an
AVX-512 IFMA kernel on x86-64 with AVX-512F and AVX-512 IFMA, a NEON kernel on AArch64 (beside four scalar
chains for long inputs), or four independent scalar chains on other 64-bit targets. Tags are unchanged.
32-bit targets and `portable-only` builds keep the radix-2^26 portable implementation.
