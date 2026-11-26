---
"rscrypto" = "minor"
---

`rscrypto::expert::MlDsa65EcdsaP256PendingSignature` signs `id-MLDSA65-ECDSA-P256-SHA512` with external component signers, such as Windows CNG, a TPM, or a Secure Enclave. It gives the signers `M'` or its SHA-256 digest; ML-DSA-65 signs `M'` with `MlDsa65EcdsaP256::LABEL` as its context. It then assembles the composite signature from the two halves and verifies it. `MlDsa65EcdsaP256PublicKey::from_components` binds the two component public keys into one composite key.
