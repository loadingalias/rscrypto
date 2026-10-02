---
"rscrypto" = "minor"
---

Expose fallible P-256 and P-384 blinded signing
and public-key derivation with entropy from the caller.
Remove `public_key_blinded` and `try_sign_blinded`.
The supported entry points are `try_public_key_blinded_with` and `try_sign_blinded_with`.
They report an entropy failure before any private arithmetic.

Add direct-fill constructors for `SecretBytes` and `SecretVec`,
and ownership transfer from `Vec` and `String` that keeps the existing allocation.
