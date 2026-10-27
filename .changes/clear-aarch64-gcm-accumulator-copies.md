---
"rscrypto" = "patch"
---

Keep AArch64 AES-GCM authentication state in its clearing owner and borrow GHASH tables across
AAD and batch helpers. Avoid the reviewed saved-register and short-batch accumulator copies,
and construct padded GHASH tails without memory-helper calls while H and state are live.
