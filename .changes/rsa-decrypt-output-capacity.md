---
"rscrypto" = "minor"
---

RSA decryption now requires `out` to hold the longest message the padding admits: `k - 11` bytes for PKCS#1 v1.5 and `k - 2 * profile.digest_len() - 2` bytes for OAEP, for a `k`-byte modulus.
A shorter `out` returns `RsaPrivateOpError::InvalidLength` before decryption, whatever the ciphertext.
Before, a short `out` returned `InvalidLength` only when the padding was valid and `DecryptionFailed` otherwise, so an attacker could tell valid padding from invalid padding.
An `out` of modulus length, as in the examples, still works.
