---
"rscrypto" = "minor"
---

Private RSA DER exports now return `SecretVec`, which wipes its allocation on
drop and requires `into_unprotected_vec()` for ordinary extraction. Secret
parsing, Serde deserialization, and generation use zeroizing temporary guards.
Secret keys, shared secrets, keypairs, and AEAD cipher contexts no longer
implement `Clone`; use `duplicate_secret()` where an additional owned secret is
required.
