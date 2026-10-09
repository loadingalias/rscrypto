---
"rscrypto" = "minor"
---

Add RFC 9881 SubjectPublicKeyInfo import and export for ML-DSA public keys: `from_spki_der`, `to_spki_der`, and `SPKI_DER_LENGTH`. Import accepts only the unique DER encoding and reports `MlDsaKeyError::MalformedDer`, `UnsupportedAlgorithm`, or `InvalidPublicKey`. Neither direction allocates.
