---
"rscrypto" = "minor"
---

Add standard key encodings for ECDSA and Ed25519. `EcdsaP256SecretKey` and `EcdsaP384SecretKey` gain `from_pkcs8_der` (RFC 5958 with an RFC 5915 ECPrivateKey), `from_sec1_der`, and `to_pkcs8_der_into`, which writes the OpenSSL form with the public key. Every public key an encoding carries must match the key derived from the scalar. `EcdsaP256PublicKey` and `EcdsaP384PublicKey` gain `to_spki_der`, and `EcdsaP256Signature` and `EcdsaP384Signature` gain `to_der_into` for DER signatures. `Ed25519SecretKey` gains RFC 8410 `from_pkcs8_der` and `to_pkcs8_der_into`, and `Ed25519PublicKey` gains `from_spki_der` and `to_spki_der`, with the new `Ed25519KeyError`. PKCS #8 import accepts a version 1 container, or a version 2 container with a matching public key, and rejects attributes as the new `EcdsaError::UnsupportedEncoding` or `Ed25519KeyError::UnsupportedEncoding`. Exports write into caller-owned fixed-size buffers, and nothing allocates.
