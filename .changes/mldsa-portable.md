---
"rscrypto" = "minor"
---

Add original portable ML-DSA-44, ML-DSA-65, and ML-DSA-87 signatures under the `ml-dsa` feature.
They support explicit deterministic and hedged signing, contexts, HashML-DSA, strict encodings,
prepared keys, and core-only operation.

With `alloc`, `keypair_from_seed_in`, `generate_keypair_in`, `try_generate_keypair_in`, and `SecretKey::try_from_slice_in` write the secret key directly into a box from an allocator
that the caller selects.
Moves then leave no by-value copies of the key.

Prepared keys fill storage that the caller owns, in place.
Construct `MlDsa{44,65,87}Prepared{Secret,Public}KeyStorage::new()` where the storage should live, then call `prepare(&mut storage)`.
Preparation needs at most 4 KiB of measured stack, instead of holding two copies of a prepared owner
(up to 163 KiB for ML-DSA-87).

Remove unguarded byte-array temporaries from the partial Keccak lane extraction
that secret SHAKE operations use.

Decode prepared ML-DSA signing state directly into its caller-owned storage,
without secret-bearing construction temporaries.

Remove repeated inverse-NTT reductions.
Outputs stay canonical.

Do not compute unused low remainders when ML-DSA signing needs only the high bits.
