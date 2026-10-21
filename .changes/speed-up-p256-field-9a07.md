---
"rscrypto" = "patch"
---

Reduce the multiplications in ECDSA P-256 field arithmetic. The Montgomery reduction uses the field prime's
special limbs, which turns eight of its multiplications into shifts and additions, and squaring has its own
kernel, so a point doubling issues 98 hardware multiplications instead of 192. Signatures and verification
results are unchanged.
