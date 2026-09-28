---
"rscrypto" = "minor"
---

Add `Argon2Block`, `Argon2Params::memory_blocks`, and `derive_with_memory`,
`derive_with_context_and_memory`, and `verify_with_memory` on `Argon2d`,
`Argon2i`, and `Argon2id`. They borrow caller-provided work memory instead of
allocating it, so callers can reuse it across operations or place it in memory
from any allocator. The operation ignores the initial contents and clears every
block it uses before returning. Short memory returns the new
`Argon2Error::MemoryTooSmall`.
