---
"rscrypto" = "minor"
---

Add original portable ML-DSA-44, ML-DSA-65, and ML-DSA-87 signatures with explicit
deterministic and hedged signing, contexts, HashML-DSA, strict encodings,
prepared keys, and core-only operation under the `ml-dsa` feature.

Prepared keys fill caller-owned storage in place: construct
`MlDsa{44,65,87}Prepared{Secret,Public}KeyStorage::new()` where the storage
should live, then call `prepare(&mut storage)`. Preparation needs at most
4 KiB of measured stack instead of holding two copies of a prepared owner
(up to 163 KiB for ML-DSA-87).

Remove unguarded byte-array temporaries from partial Keccak lane extraction used
by secret SHAKE operations.

Decode prepared ML-DSA signing state directly into its caller-owned storage,
avoiding secret-bearing construction temporaries.

Remove redundant inverse-NTT reductions while preserving canonical outputs.

Avoid computing unused low remainders when ML-DSA signing needs only high bits.
