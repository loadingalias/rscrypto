---
"rscrypto" = "patch"
---

Keep ML-DSA SHAKE256 finalization and squeezing inside a local zeroizing reader,
so that no temporary with secret state is returned.
Sampler output, public hash APIs,
and the existing cleanup and constant-time claim boundaries do not change.
