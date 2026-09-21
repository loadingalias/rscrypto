---
"rscrypto" = "patch"
---

Reduce keyed BLAKE3 one-shot overhead by removing redundant dispatch, input, secret-key, and full compression-output work while preserving cleanup.
