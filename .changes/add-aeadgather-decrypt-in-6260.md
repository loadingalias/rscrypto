---
"rscrypto" = "minor"
---

Add `AeadGather::decrypt_in_place_segments` for AES-128-GCM, AES-256-GCM, and ChaCha20-Poly1305.
It decrypts one ciphertext in place across ordered segments, such as a TLS record split between two
receive buffers, with one tag check and no copy or allocation. Each backend runs its single-buffer
kernels on every segment and stages only the block that crosses a seam. A failed tag check zeroes
every segment.
