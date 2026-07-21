---
"rscrypto" = "minor"
---

Secret comparison is now owned by fixed-size cryptographic key, tag, and
shared-secret types. The public generic `ConstantTimeEq` trait, arbitrary-slice
comparison helper, and slice/array implementations have been removed;
`SecretBytes` and `SecretVec` no longer provide equality.
