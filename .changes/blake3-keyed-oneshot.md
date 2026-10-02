---
"rscrypto" = "patch"
---

Make keyed BLAKE3 one-shot hashing faster.
It no longer repeats dispatch, input handling, secret-key handling, or full compression-output work.
Cleanup does not change.
