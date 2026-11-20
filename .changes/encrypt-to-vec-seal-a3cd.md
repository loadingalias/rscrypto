---
"rscrypto" = "patch"
---

encrypt_to_vec, seal_random_to_vec, decrypt_to_vec, try_sign_to_vec, HpkeSender::seal_to_vec, and HpkeReceiver::open_to_vec clear their output vector on error and on unwind. Before, a panic in an external AEAD or signer implementation freed the vector with its plaintext or partial output.
