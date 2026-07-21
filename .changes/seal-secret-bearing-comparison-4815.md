---
"rscrypto" = "minor"
---

Secret-bearing fixed-size keys, shared secrets, keypairs, authentication tags,
and keyed BLAKE3 outputs no longer implement `PartialEq` or `Eq`. Their inherent
`ct_eq` methods return an opaque `CtDecision`; callers must explicitly consume
it with `declassify()` when revealing equality is intended. Verification APIs
continue to return an opaque `Result`. Diagnostic HMAC and Ascon tag-comparison
helpers now return `CtDecision` instead of `bool`. Custom `Mac` implementations
must now provide `verify`; `Mac::Tag` no longer requires `Eq`.
