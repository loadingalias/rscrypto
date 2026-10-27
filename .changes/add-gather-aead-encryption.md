---
"rscrypto" = "minor"
---

Add allocation-free gather encryption for AES-128-GCM, AES-256-GCM, and ChaCha20-Poly1305 through
`AeadGather`. Ordered plaintext slices share one counter sequence and authentication tag. The expert
nonce API supports initialized output or spare capacity without staging plaintext. AES-GCM nonce
counters and fresh-random-nonce encryption also support gathered plaintext.
