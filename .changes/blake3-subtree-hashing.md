---
"rscrypto" = "minor"
---

Add BLAKE3 subtree hashing in `hashes::expert::blake3_tree`.
`Blake3Tree` hashes chunk-aligned parts of one input independently into `Blake3ChainingValue`s.
It then merges them into the same root hash, keyed hash, derived key,
or XOF output as one-shot hashing.
Each chaining value records its input range and mode.
An invalid offset, an oversized subtree,
or a merge that is not a BLAKE3 parent node returns a `Blake3SubtreeError` instead of an unrelated hash.
