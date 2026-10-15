---
"rscrypto" = "minor"
---

Move legacy RSAES-PKCS1-v1_5 decryption behind `rscrypto::auth::rsa::expert::RsaPkcs1v15Decrypt`
and implement implicit rejection from draft-irtf-cfrg-rsa-guidance-10 section 7.
Import that trait to keep calling the four `decrypt_pkcs1v15*` methods on `RsaPrivateKey`.
The inherent methods and explicit padding-failure form are removed; the root decryption API is OAEP only.

Invalid padding now returns a deterministic synthetic message derived from the private key and ciphertext,
instead of an error. Authenticate the recovered message before using it. Public length and representative
errors remain errors, and `RsaPrivateOpError::DecryptionFailed` now applies only to OAEP.
TLS 1.2 fixed-length premaster-secret handling is not provided by this trait.
