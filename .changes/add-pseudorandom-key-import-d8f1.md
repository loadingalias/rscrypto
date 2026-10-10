---
"rscrypto" = "minor"
---

Add pseudorandom-key import and multi-part `info` expansion to `HkdfSha256` and `HkdfSha384`. `from_prk` builds the expansion state from an existing hash-length PRK, such as a TLS 1.3 secret, without repeating HKDF-Extract. `expand_vectored` expands over `info` given as several slices, such as a TLS 1.3 `HkdfLabel`, without concatenating them, and keeps the `HkdfOutputLengthError` bound of `expand`. `extract` now clears its working copy of the PRK once the state holds it.
