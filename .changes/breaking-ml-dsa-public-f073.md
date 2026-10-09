---
"rscrypto" = "minor"
---

Breaking: ML-DSA public- and secret-key imports (`try_from_slice`, `try_from_slice_in`) return the new `MlDsaKeyError`, which separates import failures, including matrix-expansion `RejectionLimit`, from generation, preparation, and signing failures. `MlDsaError::InvalidPublicKey` is removed; `MlDsaError::InvalidSecretKey` now reports only an invalid stored key found during signing or preparation.
