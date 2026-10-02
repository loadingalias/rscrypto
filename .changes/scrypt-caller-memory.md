---
"rscrypto" = "minor"
---

Add `ScryptBlock`, `ScryptParams::memory_blocks`, `Scrypt::derive_with_memory`, and `Scrypt::verify_with_memory`.
They borrow work memory from the caller instead of allocating it.
Callers can reuse the memory across operations, or take it from any allocator.
The operation ignores the initial contents, and clears every block it uses before it returns.
If the memory is too short, the operation returns the new `ScryptError::MemoryTooSmall`.
