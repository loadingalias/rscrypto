---
"rscrypto" = "minor"
---

Add `EcdsaP256PublicKey::verify_sha384` and `EcdsaP384PublicKey::verify_sha256` for explicit
P-256/SHA-384 and P-384/SHA-256 signature verification. Existing `verify` methods keep their
P-256/SHA-256 and P-384/SHA-384 behavior.
