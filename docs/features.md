# Features

Select the smallest feature set that gives you the primitives you use.
[`Cargo.toml`](../Cargo.toml) defines the complete feature graph.

## Start here

The default feature is `std`, and `std` enables `alloc`.
For `no_std`, disable the default features, then name each primitive you need:

```toml
# no_std SHA-2
rscrypto = { version = "0.10", default-features = false, features = ["sha2"] }

# All primitives, with operating-system randomness
rscrypto = { version = "0.10", features = ["full", "getrandom"] }
```

Umbrella features are convenient, but they make the build larger:

| Feature         | Includes |
| --------------- | -------- |
| `checksums`     | CRC-16, CRC-24, CRC-32, and CRC-64 |
| `crypto-hashes` | SHA-2, SHA-3, BLAKE2, BLAKE3, and Ascon-Hash |
| `fast-hashes`   | XXH3 and RapidHash |
| `hashes`        | Cryptographic hashes and fast hashes |
| `auth`          | MACs, KDFs, password hashing, signatures, and key exchange |
| `aead`          | All AEADs |
| `full`          | Checksums, hashes, authentication, and AEADs |

In libraries and constrained builds, use leaf features such as `sha2`, `blake3`, `aes-gcm`, `ed25519`, `p256-ecdh`, `p384-ecdh`, `ml-dsa`, or `ml-kem`.

`websocket-sha1` gives only the compatibility digest for WebSocket handshakes.
No umbrella feature enables it, including `full`.
Enable it by name.

## Capability features

| Feature         | Effect |
| --------------- | ------ |
| `alloc`         | Enables APIs that own heap memory, including `SecretVec` and `SecretString`. |
| `std`           | Enables runtime CPU detection and standard-library integration. Enables `alloc`. |
| `getrandom`     | Enables fallible helpers that get keys, nonces, salts, or seeds from the operating system. |
| `parallel`      | Enables Rayon-based BLAKE3 and Argon2 work. Enables `std`, `blake3`, and `argon2`. |
| `serde`         | Serializes public types. |
| `serde-secrets` | Also serializes secret keys and shared secrets. Use it only at an explicit key-storage boundary. |
| `portable-only` | Uses portable Rust instead of accelerated backends, with the exceptions below. |
| `diag`          | Exposes capability and backend-selection introspection. Enables `std`. |

Benchmark, constant-time, zeroization, forced-kernel,
and component hooks need both `diag` and the repository-only `rscrypto_internal` compiler cfg.
No Cargo feature combination exposes them, including `--all-features`.
Applications must not use the internal cfg.
It has no compatibility guarantee.

With `std` and `blake3`, `Blake3::update_reader` hashes through EOF with bounded
input buffering. `Blake3Subtree::update_reader` also stops at the subtree's
remaining capacity, allowing callers to schedule independent readers and merge
their results through `hashes::expert::blake3_tree`. Both methods preserve
successfully read input on an I/O error and clear their owned input buffer.
Neither method requires `parallel`. On little-endian Linux AArch64, `parallel`
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

- `just check` repairs formatting and lints, then checks the host and the target catalog.
- `just ci-check` checks the host without repairs.
- Both lint the combined native and portable feature sets.
- `just ci-compat` checks each feature alone on the development compiler and on the MSRV.
  It also builds the bare-metal targets and runs the scalar
  and SIMD WebAssembly vectors in Wasmtime.

To check one feature selection by itself, use the `cargo check` command above.

[docs.rs](https://docs.rs/rscrypto) shows which items each feature exposes.
