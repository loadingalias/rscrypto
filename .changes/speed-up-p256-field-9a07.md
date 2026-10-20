---
"rscrypto" = "patch"
---

Speed up ECDSA P-256 field arithmetic. The Montgomery reduction uses the field prime's special limbs, which
turns eight of its multiplications into shifts and additions, and squaring has its own kernel. Signatures and
verification results are unchanged.
