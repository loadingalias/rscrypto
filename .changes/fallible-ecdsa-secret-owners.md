---
"rscrypto" = "minor"
---

Expose fallible P-256/P-384 blinded signing and public-key derivation with caller-provided entropy.
Remove `public_key_blinded` and `try_sign_blinded`; the supported entry points are
`try_public_key_blinded_with` and `try_sign_blinded_with`, which report entropy failure before private arithmetic.
Add direct-fill constructors for SecretBytes and SecretVec,
plus allocation-preserving ownership transfer from Vec and String.
