---
"rscrypto" = "minor"
---

Add `Blake3::digest_const`, which computes the unkeyed BLAKE3 digest of at most
1,024 bytes in constant context, so callers can build digest tables at compile
time. It shares the portable compression code with the runtime portable
backend; longer inputs panic, which fails the build in constant context.
