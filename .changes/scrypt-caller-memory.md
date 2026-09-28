---
"rscrypto" = "minor"
---

Add `ScryptBlock`, `ScryptParams::memory_blocks`, `Scrypt::derive_with_memory`,
and `Scrypt::verify_with_memory`. They borrow caller-provided work memory
instead of allocating it, so callers can reuse it across operations or place it
in memory from any allocator. The operation ignores the initial contents and
clears every block it uses before returning. Short memory returns the new
`ScryptError::MemoryTooSmall`.
