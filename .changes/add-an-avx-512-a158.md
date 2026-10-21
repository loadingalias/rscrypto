---
"rscrypto" = "patch"
---

Add an AVX-512 IFMA Poly1305 path for ChaCha20-Poly1305 decryption. On x86-64 with AVX-512F and AVX-512 IFMA,
decryption authenticates text of at least 256 bytes eight blocks at a time in radix 2^44. Plaintexts and tags
are unchanged.
