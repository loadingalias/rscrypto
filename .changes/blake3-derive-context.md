---
"rscrypto" = "minor"
---

Add `Blake3DeriveContext`, a BLAKE3 key-derivation context that is hashed only once, with `Blake3::derive_key_with` and `Blake3::new_derive_key_from`.
Build it at compile time with `Blake3DeriveContext::new_const` (for contexts of at most 1,024 bytes), or at run time with `Blake3DeriveContext::new`.
The outputs are equal to `Blake3::derive_key` and `Blake3::new_derive_key`, without hashing the context on each call.
