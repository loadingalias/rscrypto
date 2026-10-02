---
"rscrypto" = "minor"
---

Remove `verify_with_policy` and `verify_password_with_policy` from PBKDF2-SHA256 and PBKDF2-SHA512.
Explicit password policies use `verify_with_policy_bounded` and `verify_password_with_policy_bounded`, which need an upper iteration limit.
Default password verification and explicit primitive operations stay.

Bounded PBKDF2 verification now limits total work, not only the iteration count.
Each `OUTPUT_SIZE` block of `expected` costs one full run of `iterations`, so verification fails before derivation when `iterations * expected.len().div_ceil(OUTPUT_SIZE)` exceeds the limit.
Before, a stored record with a long `expected` value could multiply verification time without bound.
`verify` and `verify_password` apply the same rule with `MAX_VERIFY_ITERATIONS`.
