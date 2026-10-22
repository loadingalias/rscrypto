---
"rscrypto" = "patch"
---

Add an AVX-512 IFMA Poly1305 path for ChaCha20-Poly1305 and XChaCha20-Poly1305. On x86-64 with AVX-512F and
AVX-512 IFMA, text of at least 256 bytes that does not take the fused assembly is authenticated eight blocks
at a time in radix 2^44. Ciphertexts, plaintexts and tags are unchanged.
