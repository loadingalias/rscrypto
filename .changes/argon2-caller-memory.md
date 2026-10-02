---
"rscrypto" = "minor"
---

Add `Argon2Block`, `Argon2Params::memory_blocks`, and these methods on `Argon2d`, `Argon2i`, and `Argon2id`: `derive_with_memory`, `derive_with_context_and_memory`, and `verify_with_memory`.
They borrow work memory from the caller instead of allocating it.
Callers can reuse the memory across operations, or take it from any allocator.
The operation ignores the initial contents, and clears every block it uses before it returns.
If the memory is too short, the operation returns the new `Argon2Error::MemoryTooSmall`.
