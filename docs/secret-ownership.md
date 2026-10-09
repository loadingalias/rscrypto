# Secret ownership

This inventory tells users which public values keep secrets,
and which generic operations can copy or expose them.
It is a type-level contract.
It does not prove that the compiler erases every copy it makes.

## Types that own secrets

| Owner | Copying | Exposure |
| --- | --- | --- |
| `SecretBytes<N>`, `SecretVec`, `SecretString` | Not `Clone` or `Copy`. | A consuming export moves the bytes or UTF-8 text to the caller, and the caller becomes responsible for cleanup. |
| AEAD keys and contexts | Keys copy only through explicit `duplicate_secret`. Contexts do not copy. | Key export is explicit. `Debug` is redacted. |
| Header-protection keys and contexts | No generic copy. | No public export. `Debug` is redacted. |
| ECDSA and Ed25519 secret keys and keypairs | Explicit `duplicate_secret`. | Secret-key export is explicit. Keypair `Debug` shows only public data. |
| ML-DSA expanded and prepared secret keys | Not `Clone` or `Copy`. See [ML-DSA storage](#ml-dsa-storage). | Expanded export returns `SecretBytes`. Key, handle, and storage `Debug` are redacted. Serialization needs `serde-secrets`. |
| X25519 secrets; ML-KEM decapsulation keys and shared secrets | Explicit `duplicate_secret`. See [ML-KEM allocation](#ml-kem-allocation). | Secret export is explicit. `Debug` is redacted. |
| `P256EphemeralSecret`, `P256SharedSecret` | Not `Clone` or `Copy`. | The ephemeral scalar has no export or import API. `P256SharedSecret::expose_secret` makes an explicit `SecretBytes<32>` copy. `as_bytes` gives borrowed access. Both types redact `Debug`. |
| `P384EphemeralSecret`, `P384SharedSecret` | Not `Clone` or `Copy`. | The ephemeral scalar has no export or import API. `P384SharedSecret::expose_secret` makes an explicit `SecretBytes<48>` copy. `as_bytes` gives borrowed access. Both types redact `Debug`. |
| `RsaPrivateKey`, `RsaPrivateScratch` | Not `Clone` or `Copy`. | Private DER export returns `SecretVec`. `Debug` shows only public metadata. |
| HMAC, HKDF, KMAC, PBKDF2, and Poly1305 state | No generic copy. | `Debug` is redacted. Keyed state is not serialized. |
| Keyed BLAKE2 state | `Clone` where the shared `Digest` contract needs it. | `Debug` is redacted. A clone copies the keyed state. |
| `Blake3`, `Blake3XofReader` | `Clone`. | In keyed or derive-key mode, a clone copies secret-derived state. |
| `Blake3Tree`, `Blake3Subtree`, `Blake3ChainingValue` (`hashes::expert::blake3_tree`) | `Clone`. | See [BLAKE3 trees](#blake3-trees). |
| Password-hashing state and work memory | Borrowed contexts can be `Copy`. Owned state is not. See [password-hashing memory](#password-hashing-memory). | Block `Debug` is redacted. |

Typed private keys, shared secrets, keyed states, expanded schedules,
and private-operation scratch follow the same rules, also when the table does not name them.

### ML-DSA storage

- The `*_in` constructors write the key directly into an allocation from a caller-selected `Allocator` and return `Box<SecretKey, A>`.
- Preparation writes transformed secrets into storage that the caller owns.
  It returns a handle that borrows the key and the storage.
- The handle clears the secrets on drop and when preparation fails.
  The storage clears them again on drop.

### ML-KEM allocation

With `alloc`, these functions write the key
or prepared key directly into an allocation from a caller-selected `Allocator`, and return `Box<_, A>`: `generate_keypair_in`, `try_generate_keypair_in`, `DecapsulationKey::try_from_slice_in`, and `DecapsulationKey::prepare_in`.

### BLAKE3 trees

- A keyed tree holds its key.
- In keyed or derive-key mode, subtrees and chaining values hold secret-derived state.
  A clone copies that state.
- Drop clears the tree key and the chaining-value bytes.
- `Debug` shows only the mode and the input range.
- `Blake3ChainingValue::as_bytes` borrows the bytes.

`Blake3Tree::merge_level` owns bounded scratch for 32 child and 16 parent CVs
(1,536 bytes). Its destructor clears both arrays and fences. All shape and
pair validation precedes scratch population and output mutation. The tree's
existing outer by-value key arguments and backend scratch retain their separate
cleanup limits.

### BLAKE3 mixed batches

Keyed batching borrows the caller's key and outputs typed `Blake3KeyedHash`
values. Derive-key batching borrows its reusable context and writes ordinary
caller-owned arrays. Neither API clones or serializes its internal scratch.
Keyed batching owns a cleared key guard. Each serial/tree call owns key and
digest guards; secret-mode destruction clears both and then fences.
Input indexes, lengths and flags are public.
Returned outputs and unchanged fallback/outer-key copies are separate owners.

### BLAKE3 reader input

`Blake3::update_reader` and `Blake3Subtree::update_reader` own a temporary heap
buffer under `std`. The buffer is initialized before reading, never grows after
input arrives, and has no clone, formatting, serialization, or export path.
Drop clears its full initialized length, including alignment padding and bytes
written by a reader that returns an error. This applies in every hash mode.
The source reader and any copies it owns remain the caller's responsibility.

The plain parallel reader on Linux AArch64 also owns a 64-byte `ParentBlock`.
Its destructor clears all 64 bytes after the two child chaining values enter the
hasher, including unwinding after construction. The recursive helper's separate
return temporary, a caller-side left-CV argument and recursive plain arrays remain
outside this claim. Keyed and derive-key readers do not enter this path.

### BLAKE3 WASM scratch

- The SIMD128 backend borrows keys and input. Its round state also carries the vector
  chaining values; it owns no separate vector CV array. It also owns transposed message
  words, padded tails, and temporary output arrays.
- Keyed and derive-key operations clear these explicit copies after their last use.
- Padding is populated only by a partial final block. Secret-mode calls clear all four
  padding lanes when that block exists; full-block calls leave the padding owner zero.
- Returned chaining values remain owned by their callers. Arithmetic locals and compiler-created
  copies retain the documented machine-code evidence boundary.

### BLAKE3 root-output scratch

- Shared root-output helpers own decoded block words and compression-output scratch.
- Keyed and derive-key operations clear these explicit copies after their last use.
- Returned chaining values and XOF output remain owned by their existing callers.
  This adds no public secret owner or serialization path.

### Password-hashing memory

- Caller-provided `Argon2Block` and `ScryptBlock` memory is `Clone`, not `Copy`.
  It holds operation state only while an operation borrows it.
- A copy of a borrowed context copies references, not password or pepper bytes.
- The operation clears every block it used before it returns.
- PHC caller-memory verification borrows the same block storage and keeps its
  computed 32-byte digest in the existing `ZeroizingBytes` stack owner. It adds
  no owned workspace or password/pepper copy. Storage from `Vec<Block, A>` stays
  owned by the caller's allocator.

### ML-DSA internal helpers

- The private SHAKE256 helper borrows its output buffer.
  It copies the absorbed state into a zeroizing reader that already exists,
  then finalizes and squeezes inside that owner.
  It drops the absorbing core and the reader locally.
  It returns no state or reader that holds secret values.
- The secret-noise sampler keeps its acceptance mask with the input bit planes,
  and its accepted count with the output bit planes.
  It clears the input owner after each fixed block.
  It clears the output owner on success and on exhaustion.
- Copies that the compiler makes still need review for each target.

## Public authentication values

AEAD tags, HMAC tags, `Poly1305Tag`, and `Blake3KeyedHash` are visible in protocols.
They can implement `Clone`, `Copy`, raw `Debug`, or public serialization.
Where the concrete type supplies it, their verification still compares every byte.

Public keys, signatures, nonces, ciphertexts, PHC records, unkeyed hash state, `Blake3DeriveContext`,
and checksums do not own secrets.
Callers can still put sensitive data in these buffers.
`rscrypto` cannot manage memory that the caller owns.

## Explicit escape hatches

- `duplicate_secret()` creates a second secret lifetime.
- `SecretBytes::expose()` clears its source, then returns ordinary bytes.
- `SecretVec::into_unprotected_vec()` moves the allocation without clearing it.
  The caller becomes responsible for that memory.
- `SecretString::into_unprotected_string()` moves its UTF-8 allocation without clearing it.
  The caller becomes responsible for that memory.
- `serde-secrets` allows secret serialization.
- `expert::DisplaySecret` prints borrowed secret bytes on purpose.
- `as_bytes` and similar borrows expose bytes for the life of the borrow.
- `P256SharedSecret::expose_secret()` and `P384SharedSecret::expose_secret()` create a second zeroizing owner.
  The original shared secret stays live until it is dropped.

Do not log, format, or serialize a secret unless the integration needs that exact transfer.

`SecretBytes::try_fill_with` and `SecretVec::try_fill_with` give a filler direct access to the zero-initialized storage of the owner.
`SecretVec::from_vec` and `SecretString::from_string` move an existing allocation without copying or reallocating it.
These constructors create no second plaintext buffer that `rscrypto` owns.

## Changing a secret owner

A change to a secret owner needs review of `Clone`, `Copy`, `Debug`, Serde, export, allocation, comparison,
and cleanup behavior.
See [`secret-lifecycle.md`](secret-lifecycle.md) for cleanup evidence and [`constant-time.md`](constant-time.md) for timing claims.
