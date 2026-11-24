---
"rscrypto" = "minor"
---

`rscrypto::expert::MlDsa65EcdsaP256SplitSigner` signs `id-MLDSA65-ECDSA-P256-SHA512` with an external P-256 signer, such as a TPM or a Secure Enclave. It signs the ML-DSA-65 half, gives the external signer `M'` or its SHA-256 digest, and verifies the assembled signature. `MlDsa65EcdsaP256PublicKey::from_components` binds the two component public keys into one composite key.
