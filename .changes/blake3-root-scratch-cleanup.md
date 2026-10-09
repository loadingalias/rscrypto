---
"rscrypto" = "patch"
---

Clear decoded block words and compression-output scratch after keyed and derive-key
BLAKE3 root output in the shared fallback. Digest bytes and public APIs are unchanged.
