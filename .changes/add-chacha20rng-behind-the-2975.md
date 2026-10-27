---
"rscrypto" = "minor"
---

Add ChaCha20Rng behind the new chacha20-rng leaf feature: a seeded, deterministic ChaCha20 random generator with fast key erasure. It reads no operating-system entropy, returns one byte stream through fill_bytes, next_u32, and next_u64, and clears returned output and replaced keys. Its first 480 output bytes are the RFC 8439 keystream for the seed. try_generate seeds it from the operating system with getrandom. The full feature includes it.
