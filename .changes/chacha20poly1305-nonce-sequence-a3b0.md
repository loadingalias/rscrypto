---
"rscrypto" = "minor"
---

Existing `ChaCha20Poly1305::seal_random*` calls now fail at compile time.
Move a fresh key into `ChaCha20Poly1305SealingKey`. Its 12-byte nonces work with existing receivers.
For random 24-byte nonces, use XChaCha20-Poly1305 on both peers.
Generic random sealing requires `RandomNonceAead`.
