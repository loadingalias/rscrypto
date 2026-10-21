---
"rscrypto" = "patch"
---

Expand ML-KEM matrices two SHAKE128 streams at a time when importing encapsulation and decapsulation keys, and four at a time for ML-KEM-768 key generation on Apple silicon. Keys, ciphertexts, and shared secrets are unchanged.
