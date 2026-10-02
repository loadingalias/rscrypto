---
"rscrypto" = "minor"
---

Add `Blake3::digest_const`, which computes the unkeyed BLAKE3 digest of at most 1,024 bytes in a constant context.
Callers can use it to build digest tables at compile time.
It uses the same portable compression code as the runtime portable backend.
Longer inputs panic, which makes the build fail in a constant context.
