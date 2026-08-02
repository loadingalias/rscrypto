---
"rscrypto" = "major"
---

Make cryptographic boundaries misuse-resistant: keyed BLAKE2 uses validated borrowed key types and variable outputs fail with typed errors, normal AEAD sealing owns nonce issuance while caller nonces require an expert import, entropy and platform override failures no longer panic, and diagnostic or dangerous capabilities no longer clutter the crate root.
