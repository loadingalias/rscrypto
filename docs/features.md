# Features

Select the smallest feature set that gives you the primitives you use.
[`Cargo.toml`](../Cargo.toml) defines the complete feature graph.

## Start here

The default feature is `std`, and `std` enables `alloc`.
For `no_std`, disable the default features, then name each primitive you need:

```toml
# no_std SHA-2
rscrypto = { version = "0.13", default-features = false, features = ["sha2"] }

# All primitives, with operating-system randomness
rscrypto = { version = "0.13", features = ["full", "getrandom"] }
```

Umbrella features are convenient, but they make the build larger:

| Feature            | Includes |
| ------------------ | -------- |
| `checksums`        | CRC-16, CRC-24, CRC-32, CRC-64, and Adler-32 |
| `crypto-hashes`    | SHA-2, SHA-3, BLAKE2, BLAKE3, and Ascon-Hash |
| `fast-hashes`      | XXH3, XXH32, XXH64, and RapidHash |
| `hashes`           | Cryptographic hashes and fast hashes |
| `macs`             | HMAC-SHA-2, HMAC-SHA-3, KMAC, Poly1305, and SipHash |
| `kdfs`             | HKDF and PBKDF2 |
| `password-hashing` | Argon2, scrypt, and PHC password records |
| `signatures`       | ECDSA, Ed25519, RSA, ML-DSA, SLH-DSA, and composite ML-DSA |
| `key-exchange`     | X25519, P-256 ECDH, P-384 ECDH, ML-KEM, X-Wing, and HPKE |
| `auth`             | MACs, KDFs, password hashing, signatures, and key exchange |
| `aead`             | All AEADs |
| `full`             | Checksums, hashes, authentication, AEADs, and the ChaCha20 generator |

In libraries and constrained builds, use leaf features such as `sha2`, `blake3`, `aes-gcm`, `ed25519`, `p256-ecdh`, `p384-ecdh`, `ml-dsa`, `mldsa65-ed25519`, `slh-dsa`, `ml-kem`, or `x-wing`.

`chacha20-rng` gives `ChaCha20Rng`, a ChaCha20 random generator seeded by the
caller. It reads no operating-system entropy, works without `alloc`, and its
output stream is a stable function of the seed. `ChaCha20Rng::child` derives a
generator from the seed, an 8-byte label, and an index, so one recorded root
seed replays every owner's stream. `ChaCha20Rng::try_generate` seeds it from the
operating system and also needs `getrandom`. Without `getrandom`,
`try_generate_with` takes the caller's own operating-system read as its seed
source. With `rand_core`, it implements the `rand_core` 0.10 generator traits,
so `rand` and other `rand_core` consumers can use it directly.

`adler32` gives Adler-32 from [RFC 1950](https://www.rfc-editor.org/rfc/rfc1950.html),
the checksum of zlib streams and of HTTP `deflate`.
zlib stores it big-endian: after the compressed data for the uncompressed bytes,
and in the header for a preset dictionary.
It is non-cryptographic: it detects accidental corruption and gives no authentication.
`Adler32` implements `Checksum` and `ChecksumCombine`, with zlib's `adler32_combine` semantics.
It needs neither `alloc` nor `std` and keeps no input in its state.
It has a portable Rust kernel and per-architecture kernels; [`platforms.md`](platforms.md) lists them.
`rscrypto` has no zlib parser or inflate; a codec reads the checksum fields and compares them.

`xxh32` and `xxh64` give XXH32 and XXH64 from the
[xxHash specification](https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md), not XXH3.
LZ4 frames use seed-zero XXH32 for their descriptor, block, and content checksums.
Zstandard frames store the low 32 bits of the seed-zero XXH64 of the content.
Both are non-cryptographic: a frame checksum detects accidental corruption,
but gives no authentication and no resistance to chosen collisions or hash flooding.
Both are portable Rust and need neither `alloc` nor `std`.
`Xxh32Hasher` and `Xxh64Hasher` hash incrementally in a fixed-size state that keeps at most one partial stripe of input.
`rscrypto` has no frame parser; a codec reads the checksum fields and compares them.

`siphash` gives SipHash-1-3 and SipHash-2-4, keyed with a 128-bit `SipHashKey` that clears on drop.
`SipHash13` and `SipHash24` hash one byte string or act as a `core::hash::Hasher`.
`SipRandomState` is a `BuildHasher` for `HashMap` and `HashSet` that keys SipHash-1-3.
Use it for tables whose keys an attacker can choose, such as QUIC connection IDs.
`try_new_with` takes the key from caller entropy, and `try_new` also needs `getrandom`.
A 64-bit SipHash output is neither a collision-resistant hash nor a message authentication tag;
use HMAC or KMAC to authenticate messages.
The feature needs neither `alloc` nor `std`.

`websocket-sha1` gives only the compatibility digest for WebSocket handshakes.
No umbrella feature enables it, including `full`.
Enable it by name.

## Capability features

| Feature         | Effect |
| --------------- | ------ |
| `alloc`         | Enables APIs that own heap memory, including `SecretVec` and `SecretString`. |
| `std`           | Enables runtime CPU detection and standard-library integration. Enables `alloc`. |
| `getrandom`     | Enables fallible helpers that get randomness from the operating system, such as keys, nonces, salts, seeds, hedged signatures, ML-KEM and X-Wing encapsulation, and RSA blinding. |
| `parallel`      | Enables Rayon-based BLAKE3 and Argon2 work. Enables `std`, `blake3`, and `argon2`. |
| `rand_core`     | Implements `rand_core` 0.10's `TryRng`, `TryCryptoRng`, and `SeedableRng` for `ChaCha20Rng`. Enables `chacha20-rng`, adds only the `rand_core` dependency, and works without `alloc`. `SeedableRng::seed_from_u64` panics, because a 64-bit value is not a 256-bit seed. |
| `serde`         | Serializes public types. |
| `serde-secrets` | Also serializes secret keys and shared secrets. Use it only at an explicit key-storage boundary. |
| `portable-only` | Uses portable Rust instead of accelerated backends, with the exceptions below. |
| `diag`          | Exposes capability and backend-selection introspection. Enables `std`. |

Benchmark, constant-time, zeroization, forced-kernel,
and component hooks need both `diag` and the repository-only `rscrypto_internal` compiler cfg.
The ML-DSA signing-attempt counter needs the repository-only `rscrypto_mldsa_attempts` cfg instead.
No Cargo feature combination exposes them, including `--all-features`.
Applications must not use either cfg.
It has no compatibility guarantee.

With `chacha20poly1305`, move a fresh key into `ChaCha20Poly1305SealingKey::new(key)`.
Keep one sealing key per sender. Its `seal` methods return a unique 12-byte nonce with each ciphertext.
They work without `getrandom`. Use a fresh key when migrating from random nonces or restarting a sender.
Receivers open messages with `ChaCha20Poly1305`.

For random nonces, use `xchacha20poly1305` with `getrandom`. Both peers must support its 24-byte nonce.
Generic code that calls `seal_random*` needs the `RandomNonceAead` bound.

The `aes-gcm` and `chacha20poly1305` leaves implement `AeadGather`. It encrypts ordered plaintext
slices into one ciphertext without staging plaintext or allocating.
Protocols that derive unique nonces use `aead::expert::AeadWithNonce`. Its append helper
requires `alloc` and writes ciphertext directly into a vector's spare capacity.
`AeadGather::decrypt_in_place_segments` decrypts one ciphertext in place across ordered segments,
such as a TLS record split between two receive buffers. It needs no allocation. A failed tag check
zeroes every segment.

The `aes-gcm` leaf also gives `aead::expert::quic_lb`, the connection-ID cipher of
[draft-ietf-quic-load-balancers-22](https://datatracker.ietf.org/doc/draft-ietf-quic-load-balancers/22/).
`QuicLbCipher::new` takes a 16-byte `QuicLbKey`, a server-ID length of at least 1, and a nonce length of
at least 4, with a total of at most 19. It returns `None` for other lengths.
`encrypt` and `decrypt_server_id` work in place on a 19-byte buffer, do not allocate, and do not fail.
The module gives no general AES block encryption or decryption.

With `std` and `blake3`, `Blake3::update_reader` hashes through EOF with bounded
input buffering. `Blake3::update_file` reads from a file's current position through
actual EOF and uses regular-file size and position hints to choose buffer sizes.
Unavailable or stale hints never limit the input. Small files stay on the stack;
larger files may reserve one bounded heap window. `Blake3Subtree::update_reader` also stops at the subtree's
remaining capacity, allowing callers to schedule independent readers and merge
their results through `hashes::expert::blake3_tree`. All three methods preserve
successfully read input on an I/O error and clear their owned input buffer.
None requires `parallel`. On little-endian Linux AArch64, `parallel`
allows the plain hasher to process complete aligned buffers in the current Rayon
pool, initializing the global pool if needed. Keyed and derive-key readers use
ordinary updates. The subtree reader creates no threads; callers own its scheduling.

With the `blake3` leaf feature, `digest_batch`, `keyed_digest_batch`, and
`Blake3DeriveContext::derive_key_batch` accept independent mixed-length inputs
and caller-provided output storage. Plain equal-length runs can use SIMD;
other plain inputs and keyed/derive-key batches use individual calls. Larger messages use
the existing tree path. These APIs and
`Blake3Tree::merge_level` require neither allocation nor `std`. A merge level
validates every child pair before writing any output.

With `std`, `hashes::expert::bao::Decoder` verifies Bao combined encodings
against an independently trusted BLAKE3 root. It returns only verified chunks
and authenticates the final chunk before reporting EOF. It supports sequential
unkeyed decoding; outboard encodings, slices and seeking are outside this API.

`hashes::expert::BLAKE3_PORTABLE_SOURCE` holds the source text of the portable BLAKE3 compression
function that rscrypto compiles, for code that builds with Rust 1.85 and cannot link rscrypto.

The [BLAKE3 performance matrix](blake3-performance.md) records qualified
results and the remaining target and workload gaps.

`getrandom` changes how `rscrypto` gets entropy.
It does not change which algorithms are available.
APIs that accept entropy from the caller work without it.

`argon2`, `scrypt`, and `phc-strings` enable `alloc`, including when default
features are disabled. Their caller-memory methods reuse the work buffer;
raw `verify_with_memory` still allocates a temporary digest. See
[password-hashing memory APIs](password-hashing-memory.md) for PHC verification
with reusable, allocator-backed storage and the allocation contract.

`ml-dsa` supports all three ML-DSA parameter sets without allocation and without operating-system entropy.
See [ML-DSA](mldsa.md) for the API, the memory costs, and the open qualification gates.

`mldsa65-ed25519` and `mldsa65-ecdsa-p256` each give one composite signature from
[draft-ietf-lamps-pq-composite-sigs-19](https://datatracker.ietf.org/doc/draft-ietf-lamps-pq-composite-sigs/19/)
(RFC-to-be 10073): `id-MLDSA65-Ed25519-SHA512` and `id-MLDSA65-ECDSA-P256-SHA512`.
They enable `ml-dsa` and `ed25519` or `ecdsa-p256`, need neither allocation nor operating-system entropy,
and import and export the draft's raw, SubjectPublicKeyInfo, and PKCS #8 encodings.
A signature is valid only if both components verify. `prepare_in` and the allocator-backed
secret-key constructors (`generate_keypair_in`, `try_from_slice_in`, and `from_pkcs8_der_in`) need `alloc`.
The `rscrypto::auth::composite_mldsa` module documentation describes the construction and its limits.

`x-wing` gives the X-Wing hybrid KEM of ML-KEM-768 and X25519 from
[draft-connolly-cfrg-xwing-kem-11](https://datatracker.ietf.org/doc/draft-connolly-cfrg-xwing-kem/11/),
which draft-irtf-cfrg-concrete-hybrid-kems-04 names MLKEM768-X25519 (HPKE KEM `0x647A`).
It enables `ml-kem` and `x25519` and needs neither allocation nor operating-system entropy.
Keys and ciphertexts use the draft's raw encodings. The decapsulation key is the 32-byte seed.
It has one difference from the -11 text: encapsulation and decapsulation reject an all-zero X25519 shared secret,
as X-Wing pull request 39 at `782c880e` and RFC 9180 specify.
X25519 clamps its scalar, so that result depends only on the public key or ciphertext.
`prepare_in` needs `alloc`.
The `rscrypto::auth::xwing` module documentation describes the construction.

`hpke` gives HPKE Base mode from [RFC 9180](https://www.rfc-editor.org/rfc/rfc9180.html), with its
verified errata except 7121. draft-ietf-hpke-hpke-06 will replace RFC 9180, with a byte-for-byte identical Base mode.
The `rscrypto::auth::hpke` module documentation says why erratum 7121 does not apply.
`hpke` enables only `hkdf`. Each suite is available only when its KEM and AEAD features are also enabled,
so `hpke` alone gives no suite:

| Suite | Features |
| --- | --- |
| `HpkeX25519HkdfSha256Aes128Gcm`: DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, AES-128-GCM | `hpke`, `x25519`, `aes-gcm` |
| `HpkeXWingHkdfSha256ChaCha20Poly1305`: X-Wing (`0x647A`), HKDF-SHA256, ChaCha20Poly1305 | `hpke`, `x-wing`, `chacha20poly1305` |

The X-Wing suite follows draft-ietf-hpke-pq-05 for its KEM identifier and key derivation.
HPKE needs neither allocation nor operating-system entropy. `seal_to_vec`, `open_to_vec`, `setup_in`, and `derive_keypair_in` need `alloc`.
The `rscrypto::auth::hpke` module documentation describes the API.

`slh-dsa` supports all 12 FIPS 205 SLH-DSA parameter sets, each as a pure profile and an RFC 9909
HashSLH-DSA profile, without allocation and without operating-system entropy. It enables `hmac` and `sha3`.
Signatures are 7,856 to 49,856 bytes and go into buffers that the caller provides. The small (`s`)
sets compute larger trees than the fast (`f`) sets, so they sign much more slowly, and importing a
secret key costs one key generation. The allocator-backed secret-key constructors
(`generate_keypair_in`, `try_from_slice_in`, and `from_pkcs8_der_in`) need `alloc`.
See [SLH-DSA](slhdsa.md) for the profile choice, the API, and the open qualification gates.

`p256-ecdh` and `p384-ecdh` are standalone leaf features.
They do not enable ECDSA, HMAC, `alloc`, or `std`.
See [`platforms.md`](platforms.md) for backend selection, [`constant-time.md`](constant-time.md) for timing claims,
and [`test-vector-coverage.md`](test-vector-coverage.md) for independent vectors.

`portable-only` makes runtime detection report no accelerated capabilities.
It also turns off assembly that the crate selects at compile time from the target alone,
such as the RSA, elliptic-curve, ML-KEM, and ML-DSA backends.
It excludes nightly-only POWER, IBM Z, and RISC-V backends from compilation,
so these targets can build on the declared MSRV.
It does not promise to remove every accelerated backend from the binary.
Some hash backends that a compile-time `target_feature` setting enables still run,
for example SHA-256 on Apple Silicon.
See [`platforms.md`](platforms.md).

## Check a selection

```bash
cargo check --no-default-features --features sha2
just plan
just check
```

- `just check` repairs formatting and lints, then checks this Mac. It lints the combined native and
  portable feature sets, and runs the CI tooling checks.
- CI checks every other target without repairs. Its compatibility row (`scripts/check/compat.py`) checks
  each feature alone on the development compiler and on the MSRV. It also builds the bare-metal targets
  and runs the scalar and SIMD WebAssembly vectors in Wasmtime.

To check one feature selection by itself, use the `cargo check` command above.

[docs.rs](https://docs.rs/rscrypto) shows which items each feature exposes.
