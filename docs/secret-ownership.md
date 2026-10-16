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
| ECDSA and Ed25519 secret keys and keypairs | Explicit `duplicate_secret`. | Secret-key export is explicit. PKCS #8 export writes into a fixed-size buffer that the caller owns and clears. PKCS #8 and SEC1 import borrow the caller's input and do not clear it. Keypair `Debug` shows only public data. |
| ML-DSA seeds, expanded secret keys, and prepared secret keys | Not `Clone` or `Copy`. See [ML-DSA storage](#ml-dsa-storage). | Raw export returns `SecretBytes`. PKCS #8 export writes into a fixed-size buffer that the caller owns and clears. Seed, key, handle, and storage `Debug` are redacted. Secret-key serialization needs `serde-secrets`; seeds have no Serde support. |
| SLH-DSA secret keys, pure and HashSLH-DSA | Not `Clone` or `Copy`. | Raw export returns `SecretBytes`. PKCS #8 export writes into a fixed-size buffer that the caller owns and clears. Raw and PKCS #8 import borrow the caller's input and do not clear it. `Debug` is redacted. No Serde support. |
| X25519 secrets; ML-KEM decapsulation keys and shared secrets | Explicit `duplicate_secret`. See [ML-KEM allocation](#ml-kem-allocation). | Secret export is explicit. ML-KEM decapsulation-key PKCS #8 export writes into a fixed-size buffer that the caller owns and clears. `Debug` is redacted. |
| ML-KEM seeds | Not `Clone` or `Copy`. | Raw export returns `SecretBytes<64>`. PKCS #8 export writes into a fixed-size buffer that the caller owns and clears. `Debug` is redacted. No Serde support. |
| `P256EphemeralSecret`, `P256SharedSecret` | Not `Clone` or `Copy`. | The ephemeral scalar has no export or import API. `P256SharedSecret::expose_secret` makes an explicit `SecretBytes<32>` copy. `as_bytes` gives borrowed access. Both types redact `Debug`. |
| `P384EphemeralSecret`, `P384SharedSecret` | Not `Clone` or `Copy`. | The ephemeral scalar has no export or import API. `P384SharedSecret::expose_secret` makes an explicit `SecretBytes<48>` copy. `as_bytes` gives borrowed access. Both types redact `Debug`. |
| `RsaPrivateKey`, `RsaPrivateScratch` | Not `Clone` or `Copy`. | Private DER export returns `SecretVec`. `Debug` shows only public metadata. |
| HMAC, HKDF, KMAC, PBKDF2, and Poly1305 state | No generic copy. | `Debug` is redacted. Keyed state is not serialized. HKDF `prk()` gives borrowed access to the PRK. `from_prk` borrows the caller's PRK and does not clear it. |
| Keyed BLAKE2 state | `Clone` where the shared `Digest` contract needs it. | `Debug` is redacted. A clone copies the keyed state. |
| `Blake3`, `Blake3XofReader` | `Clone`. | In keyed or derive-key mode, a clone copies secret-derived state. |
| `Blake3Tree`, `Blake3Subtree`, `Blake3ChainingValue` (`hashes::expert::blake3_tree`) | `Clone`. | See [BLAKE3 trees](#blake3-trees). |
| Password-hashing state and work memory | Borrowed contexts can be `Copy`. Owned state is not. See [password-hashing memory](#password-hashing-memory). | Block `Debug` is redacted. |

Typed private keys, shared secrets, keyed states, expanded schedules,
and private-operation scratch follow the same rules, also when the table does not name them.

### RSA implicit rejection

`RsaPrivateKey` retains the imported private exponent's value, padded to the
public modulus width. It retains the CRT coefficient at the public factor width,
so steady-state copies do not use its significant-byte count. Legacy
`expert::RsaPkcs1v15Decrypt` borrows the stored exponent to derive a rejection key
for each ciphertext; neither the exponent hash nor that key is cached, exported,
or formatted.
The local owners are the exponent's 32-byte SHA-256 hash, the 32-byte key
derivation key (KDK), 256 candidate-length bytes, and the modulus-width
alternative message (AM). The allocating path owns AM in a `Vec`;
scratch paths reuse `RsaPrivateScratch::blinding_power` after private arithmetic.

The internal HMAC helper borrows a 32-byte key and owns its 64-byte key block,
ipad and opad, SHA-256 states, inner digest, and returned 32-byte block.
The caller clears each returned derivation block after use. The selected
plaintext or synthetic message is copied into caller-owned `out`; its cleanup
is the caller's responsibility. See [cleanup boundaries](secret-lifecycle.md#rsa-implicit-rejection).

### rustls symmetric adapters

The `rscrypto-rustls` companion owns its HMAC keys in `SecretVec` and its HKDF
expansion state in `HkdfSha256` or `HkdfSha384`. Neither adapter is cloneable.
HKDF imports borrow rustls's `OkmBlock`; block expansion uses a `SecretBytes`
scratch owner before copying into rustls's clearing `OkmBlock`.
HMAC results cross the boundary through rustls's clearing `hmac::Tag`.

Record, packet, and header-protection keys own rscrypto AEAD or header-protection
contexts. Construction borrows rustls's clearing `AeadKey`, then creates typed
rscrypto key owners and expanded contexts. The adapters expose no key formatting
or serialization. `Tls13AeadAlgorithm::extract_keys` explicitly transfers
rustls-owned traffic secrets to the caller, as rustls requires.
Transcript contexts can be cloned; they contain public handshake data.

### rustls key exchange

Each active exchange owns an rscrypto `X25519SecretKey`, `P256EphemeralSecret`,
or `P384EphemeralSecret`. The hybrid also owns a boxed `MlKem768DecapsulationKey`,
generated directly into its allocation. Exchanges are neither cloneable nor
formattable. Completing or abandoning an exchange drops those clearing owners.

Primitive shared secrets remain in rscrypto's clearing owners until copied into
rustls's clearing `SharedSecret`. Hybrid concatenation creates one 64-byte vector
and moves it into that rustls owner. The borrowed component secrets clear on return.
Public shares and ciphertexts carry no secret ownership obligation.

### rustls authentication

The provider shares `EcdsaP256SecretKey`, `EcdsaP384SecretKey`,
`Ed25519SecretKey`, `RsaPrivateKey`, and ML-DSA keys (prepared by default,
compact with `sign::Provider::compact`) through `Arc`; their primitive owners
clear when the last signing key or signer releases them. RSA signers each own
one clearing `RsaPrivateScratch` behind a mutex. They borrow the key and scratch
for the blinded operation and return only the resulting public signature.
ECDSA signatures and cached SPKI are public. Key Debug formatting delegates to
redacted primitive owners and the caller's selected entropy source; that source
must not expose secret state through Debug. RSA signer formatting exposes only
its signature scheme. Groups and signing keys borrow a static entropy source;
the provider neither owns nor clones its state.

The ML-DSA loader constructs compact keys directly in allocator-backed storage.
By default it then prepares each key once with `prepare_in`, keeps the prepared
key, which owns its `MlDsa44/65/87PreparedSecretKeyStorage` box, and drops the
compact key, which clears it. Signers share the prepared key through `Arc` and
return public signatures. With `sign::Provider::compact`, the loader keeps the
compact key instead, and each signature prepares a temporary prepared key with
`prepare_in`. A prepared key clears its transformed secret polynomials and `K`
when it drops, and its storage clears them again before deallocation; a failed
signature leaves a retained prepared key unchanged.
`PreparedServerVerifier` and `PreparedClientVerifier` retain prepared ML-DSA
anchor keys; their storage is public. PKCS#8 input uses rustls-pki-types' borrowed or owned
DER containers. `Provider::load_private_key` immediately wraps the input in
`zeroize::Zeroizing`, before the size check or any parser call. With
rustls-pki-types 1.15.1 and zeroize 1.9.0, dropping this guard clears the owned
vector's full allocation, then frees it. Success, rejection, and unwinding
share that guard. Immutable borrowed static DER cannot be cleared and remains
the caller's responsibility. Aborting the process does not run destructors.

### rustls session tickets

Each private ticket key owns a clearing `Aes256Gcm` context behind a
`RwLock<Option<_>>`, a monotonic creation instant, and an atomic public nonce
counter. On its first use at or after twelve hours, it takes and drops the cipher.
The missing cipher is terminal; neither encryption nor decryption can revive it. A `SecretBytes<32>` protects entropy acquisition; the typed key
and temporary owner clear after context construction. `TicketRotator` owns the
current and previous keys and releases displaced keys on rotation. The adapter
neither clones keys nor serializes them. Plaintext copies during seal/open stay
in `SecretVec` until successful transfer to ciphertext or rustls-owned plaintext.
The incoming borrowed plaintext and the returned plaintext belong to rustls.

### rustls QUIC integration

Each endpoint configuration owns a shared clearing `SecretBytes<32>` reset key.
Its retained 64-bit connection-ID filter key is a non-cryptographic Quinn filter,
not an authentication secret. Server configurations own a clearing `HkdfSha256`
address-token root. Each token derives an `Aes256Gcm` context through a clearing
`SecretBytes<32>` scratch. A derived context seals once; Quinn supplies the fresh
128-bit derivation nonce. Incoming token buffers belong to Quinn.

Sessions own rustls QUIC connections, queued handshake/key-change events, and
next-generation traffic secrets. At handshake completion, each session takes
rustls's `KeyingMaterialExporter` owner to support Quinn's repeated exporter calls.
The move transfers ownership without exporting the secret bytes.
Packet and header wrappers own rustls trait objects backed by the same clearing
rscrypto contexts as the symmetric adapters. Cloning endpoint or server
configuration extends the corresponding root-key lifetime. Exporter buffers
and successful plaintext belong to the caller. No adapter key export, clone,
or Debug implementation exposes these keys.

### ML-DSA storage

- The `*_in` constructors, including `SecretKey::from_pkcs8_der_in` and `Seed::keypair_in`,
  write the key directly into an allocation from a caller-selected `Allocator` and return `Box<SecretKey, A>`.
- PKCS #8 import borrows the caller's DER and does not clear it.
  A seed owner copies its 32 bytes out of the DER; secret-key import copies or expands the key into its owner.
- Preparation writes transformed secrets, the public matrix, and the key's rho, `K`, and tr into storage.
  `prepare` uses storage that the caller owns and returns a prepared key that borrows it.
  `prepare_in` allocates the storage from a caller-selected `Allocator`, prepares it in place,
  and returns a prepared key that owns the box.
  Neither prepared key borrows the source key.
- The prepared key clears the transformed secrets and `K` on drop and when preparation fails.
  The storage clears them again on drop, before an allocated box is freed.

### ML-KEM allocation

With `alloc`, these functions write the key
or prepared key directly into an allocation from a caller-selected `Allocator`, and return `Box<_, A>`: `generate_keypair_in`, `try_generate_keypair_in`, `DecapsulationKey::try_from_slice_in`, `DecapsulationKey::from_pkcs8_der_in`, `DecapsulationKey::prepare_in`, and `Seed::keypair_in`.
PKCS #8 import borrows the caller's DER and does not clear it.

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

### Bao decoder input

The unkeyed Bao decoder owns a fixed 1,024-byte input buffer and a bounded
stack of subtree chaining values. Returned errors clear the input buffer;
drop clears it in all states. A caught reader panic makes the decoder terminal
but retains its internal buffer until drop. It exposes no content or unverified
length through `Debug`, and has no clone or serialization path. Previously
returned verified bytes and the underlying reader remain caller-owned.

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

### BLAKE3 partial-chunk forest

- With NEON or AVX-512 dispatch, a streaming update can start at an empty chunk
  whose counter is divisible by 16. When it holds complete chunks followed by a
  partial chunk (3, 7, 11 or 15 on NEON; 3 or 15 on AVX-512), it owns scratch for
  up to 16 leaf, 8 parent and 4 frontier chaining values (896 bytes).
- Keyed and derive-key updates clear every slot that shape populates on return or
  unwind, then fence. Plain updates hold chaining values of public input and skip
  the clear.
- The NEON lane worker clears its named vector working storage in every mode. The
  AVX-512 lane workers have no named working owner; their register copies and spills
  retain the documented machine-code evidence boundary.
- Frontier chaining values pass by value to the existing stack merge, which clears
  its copies in secret modes. The partial chunk's chaining value moves into the
  hasher's existing chunk state. This adds no public secret owner or serialization path.

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
- The [signing-residue qualification](../benchmark_results/OVERVIEW.md#2026-10-16-ml-dsa-optimization-qualification)
  checks masks, NTT masks, rejected candidates, commitments and hint maps in the
  measured macOS AArch64 stack. It does not change their ownership contract or
  establish cleanup for every compiler-created copy.
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
