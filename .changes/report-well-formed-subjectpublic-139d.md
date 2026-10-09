---
"rscrypto" = "patch"
---

Report well-formed SubjectPublicKeyInfo keys of other algorithms, such as RSA or Ed25519, as `EcdsaError::UnsupportedAlgorithm` instead of `EcdsaError::MalformedDer`.
