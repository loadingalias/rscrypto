---
"rscrypto" = "patch"
---

Keep ML-DSA SHAKE256 finalization and squeezing inside a local zeroizing reader,
avoiding a returned secret-state temporary. Preserve sampler output, public
hash APIs, and existing cleanup and constant-time claim boundaries.
