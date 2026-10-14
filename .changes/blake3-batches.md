---
"rscrypto" = "minor"
---

Add `Blake3::keyed_digest_batch` and `Blake3DeriveContext::derive_key_batch`. They hash independent inputs into caller-provided outputs, in order, without allocation, through the existing one-shot tree paths.
