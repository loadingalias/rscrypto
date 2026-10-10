---
"rscrypto" = "minor"
---

Add RFC 9881 PKCS #8 import and export for ML-DSA private keys. `SecretKey::from_pkcs8_der` and `from_pkcs8_der_in` accept the seed, expanded, and both forms in a version 1 container or a version 2 container with a public key, check that redundant fields agree, and reject attributes as the new `MlDsaKeyError::UnsupportedEncoding`. The new `MlDsa44Seed`, `MlDsa65Seed`, and `MlDsa87Seed` owners keep the recommended seed form: `generate`, `keypair`, `keypair_in`, `from_pkcs8_der`, and `to_pkcs8_der_into`. `SecretKey::to_pkcs8_der_into` writes the expanded form. Exports write into caller-owned fixed-size buffers, and nothing allocates except the `_in` constructors.
