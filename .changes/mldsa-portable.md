---
"rscrypto" = "minor"
---

Add original portable ML-DSA-44, ML-DSA-65, and ML-DSA-87 signatures with explicit
deterministic and hedged signing, contexts, HashML-DSA, strict encodings,
prepared keys, and core-only operation under the `ml-dsa` feature.

Remove unguarded byte-array temporaries from partial Keccak lane extraction used
by secret SHAKE operations.

Decode prepared ML-DSA signing state directly into its retained owner, avoiding
an extra secret-bearing construction temporary.

Remove redundant inverse-NTT reductions while preserving canonical outputs.

Fuse the first two portable inverse-NTT stages while preserving canonical
arithmetic and strict overflow checks.

Avoid computing unused low remainders when ML-DSA signing needs only high bits.
