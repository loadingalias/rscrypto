---
"rscrypto" = "patch"
---

Panic before absorbing SHA-384, SHA-512, or SHA-512/256 input that would
exceed the FIPS 180-4 length field instead of wrapping the encoded bit length.
