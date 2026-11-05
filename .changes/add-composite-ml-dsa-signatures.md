---
"rscrypto" = "minor"
---

Add composite ML-DSA signatures from draft-ietf-lamps-pq-composite-sigs-19 (RFC-to-be 10073): `MlDsa65Ed25519` (`id-MLDSA65-Ed25519-SHA512`, feature `mldsa65-ed25519`) and `MlDsa65EcdsaP256` (`id-MLDSA65-ECDSA-P256-SHA512`, feature `mldsa65-ecdsa-p256`). Both build on the existing ML-DSA-65, Ed25519, and ECDSA P-256 implementations; a signature verifies only if both components verify. They provide key generation from caller randomness, deterministic and hedged signing with an application context of up to 255 bytes, external SHA-512 prehash signing and verification, prepared signing keys, and strict raw, SubjectPublicKeyInfo, and PKCS #8 encodings. The `signatures` umbrella feature includes both.
