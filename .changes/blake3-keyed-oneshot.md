---
"rscrypto" = "patch"
---

Reduce keyed BLAKE3 one-shot overhead by removing redundant dispatch, input, and secret-key copies while preserving cleanup.
