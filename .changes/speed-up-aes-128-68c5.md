---
"rscrypto" = "patch"
---

Speed up AES-128-GCM and AES-256-GCM on POWER8 and later. Whole 128-byte chunks run eight AES blocks together with counters, data, and GHASH input kept in vector registers, then fold the eight ciphertext blocks into GHASH with one reduction. Ciphertexts and tags are unchanged.
