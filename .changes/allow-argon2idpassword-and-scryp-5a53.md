---
"rscrypto" = "patch"
---

Allow Argon2idPassword and ScryptPassword to generate canonical PHC records from a caller-owned entropy source without enabling getrandom. Add exact-width HMAC-SHA256 verification for protocols that specify a 64-bit truncated tag.
