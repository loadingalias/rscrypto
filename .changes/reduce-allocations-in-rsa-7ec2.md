---
"rscrypto" = "patch"
---

Reduce allocations in RSA key encoding and generation. Reuse SHA-2 dispatch and HKDF length calculations.
