---
"rscrypto" = "major"
---

Collapse RapidHash to portable `RapidHash64`, `RapidStreamHasher`,
`RapidHasher`, `RapidSeededState`, and fallible `RapidRandomState`. Remove the
`RapidHash` alias, 128-bit/native-endian variants, `RapidBuildHasher`, duplicate
cores, placeholder dispatch, and direct/default `RapidHasher` construction;
collection hashers now come from an explicit state.
