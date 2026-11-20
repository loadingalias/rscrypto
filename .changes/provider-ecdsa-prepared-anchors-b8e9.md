---
"rscrypto-rustls" = "patch"
---

ECDSA trust anchors are kept as prepared keys, and a certificate key that is not an anchor is parsed and verified once with rscrypto 0.13 ECDSA public keys.
