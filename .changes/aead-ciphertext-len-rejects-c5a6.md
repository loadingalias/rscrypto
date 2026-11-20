---
"rscrypto" = "patch"
---

Aead::ciphertext_len rejects a length above isize::MAX, because no byte buffer can hold it. Thus on 32-bit targets, encrypt_to_vec, seal_random_to_vec, and HpkeSender::seal_to_vec return a length error before they allocate. They no longer panic.
