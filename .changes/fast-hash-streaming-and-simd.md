---
"rscrypto" = "minor"
---

Replaced heap-buffered RapidHash and XXH3 `Hasher` implementations with fixed-size,
allocation-free state. Added the collection-oriented `RapidHasher`, concatenating
`RapidStreamHasher`, deterministic `RapidBuildHasher`, and XXH3-128 streaming.
Deterministic builder documentation now restricts it to trusted keys.

Accelerated long XXH3 streams with AVX2, AVX-512, NEON, VSX, and z/Vector, and
added seeded partition fuzzing, RapidHash collection properties, backend
equivalence tests, and explicit allocation coverage.
