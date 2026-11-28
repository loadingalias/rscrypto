---
"rscrypto" = "patch"
---

On AArch64, ML-KEM encapsulation and decapsulation, X-Wing encapsulation, and the HPKE `*_in` and single-shot `seal` and `open` forms build their secret results in place, so their data-independent timing scopes keep no extra stack copy. The HPKE key schedule builds its context without a moved copy on every target. Stack use falls by up to 6.8 KB for HPKE X-Wing seal and 4.4 KB for X-Wing encapsulation.
