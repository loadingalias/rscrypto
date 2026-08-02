---
"rscrypto" = "patch"
---

Bound default password-record PBKDF2 verification work while keeping raw
derivation and compatibility policies unbounded. Custom record policies can
set an explicit ceiling with `verify_with_policy_bounded` or
`verify_password_with_policy_bounded`.
