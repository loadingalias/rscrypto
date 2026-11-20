---
"rscrypto" = "minor"
---

ECDSA public keys no longer build a verification table. `EcdsaP256PublicKey` and `EcdsaP384PublicKey` hold only the point, are `Copy`, and verify through the single-use path; `prepare()` returns the new `EcdsaP256PreparedPublicKey` or `EcdsaP384PreparedPublicKey`, which keeps the table for a key that verifies many signatures. `verify_sec1`, `verify_sec1_sha384`, and `verify_sec1_sha256` are removed: `from_sec1_bytes` followed by `verify` now does the same work. Deriving a public key from a secret key therefore builds no table either; on an Apple M1 Pro, P-256 derivation takes 7.9 µs instead of 91.6 µs, and P-384 derivation 53.3 µs instead of 238.0 µs.
