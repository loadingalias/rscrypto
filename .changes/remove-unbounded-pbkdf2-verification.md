---
"rscrypto" = "minor"
---

Remove `verify_with_policy` and `verify_password_with_policy` from PBKDF2-SHA256 and PBKDF2-SHA512.
Explicit password policies use `verify_with_policy_bounded` and `verify_password_with_policy_bounded`,
which require an upper iteration limit. Default password verification and explicit primitive operations remain.
