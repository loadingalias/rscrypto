# Features

Select the smallest feature set that exposes the primitives you use.
[`Cargo.toml`](../Cargo.toml) is the complete, authoritative feature graph.

## Start here

The default feature is `std`; `std` enables `alloc`.
Disable defaults for `no_std`, then name every required primitive:

```toml
# no_std SHA-2
rscrypto = { version = "0.10", default-features = false, features = ["sha2"] }

# Full API with OS randomness
rscrypto = { version = "0.10", features = ["full", "getrandom"] }
```

Umbrella features trade build size for convenience:

| Feature         | Includes |
| --------------- | -------- |
| `checksums`     | CRC-16, CRC-24, CRC-32, and CRC-64 |
| `crypto-hashes` | SHA-2, SHA-3, BLAKE2, BLAKE3, and Ascon hash |
| `fast-hashes`   | XXH3 and RapidHash |
| `hashes`        | Cryptographic and fast hashes |
| `auth`          | MACs, KDFs, password hashing, signatures, and key exchange |
| `aead`          | Every AEAD implementation |
| `full`          | Checksums, hashes, authentication, and AEADs |

Prefer leaf features such as `sha2`, `blake3`, `aes-gcm`, `ed25519`, `p256-ecdh`, `ml-dsa`, or `ml-kem` in libraries and constrained builds.

`websocket-sha1` exposes only the compatibility digest for WebSocket handshakes.
It is excluded from every umbrella feature, including `full`; enable it explicitly.

## Capability features

| Feature         | Effect |
| --------------- | ------ |
| `alloc`         | Enables APIs that own dynamic memory, including `SecretVec` and `SecretString`. |
| `std`           | Enables runtime CPU detection and standard-library integrations; implies `alloc`. |
| `getrandom`     | Enables fallible helpers that obtain keys, nonces, salts, or seeds from the OS. |
| `parallel`      | Enables Rayon-backed BLAKE3 and Argon2 work; implies `std`, `blake3`, and `argon2`. |
| `serde`         | Serializes public types. |
| `serde-secrets` | Also serializes secret keys and shared secrets; use only at an explicit key-storage boundary. |
| `portable-only` | Makes runtime capability detection report no SIMD or ASM capabilities. |
| `diag`          | Exposes capability and backend-selection introspection; implies `std`. |

Benchmark, constant-time, zeroization, forced-kernel, and component hooks require both `diag` and the
repository-only `rscrypto_internal` compiler cfg. Ordinary Cargo feature combinations, including
`--all-features`, do not expose those operations. The internal cfg is unsupported for application dependencies
and carries no compatibility guarantee.

`getrandom` changes entropy acquisition, not algorithm availability.
APIs that accept caller-provided entropy remain available without it.

`ml-dsa` supports all three ML-DSA parameter sets without allocation or OS entropy.
See [ML-DSA](mldsa.md) for the API, memory costs, and open qualification gates.

`p256-ecdh` is a standalone leaf: it does not enable ECDSA, HMAC, `alloc`, or `std`.
See [`platforms.md`](platforms.md) for backend selection, [`constant-time.md`](constant-time.md) for timing claims,
and [`test-vector-coverage.md`](test-vector-coverage.md) for independent vectors.

`portable-only` affects dispatchers that consult `platform::caps()`.
It does not remove accelerated code from the binary
or override backends selected by compile-time `target_feature` settings.
See [`platforms.md`](platforms.md).

## Verify a selection

```bash
cargo check --no-default-features --features sha2
just plan
just check
```

`just check` and `just ci-check` lint the combined native and portable feature sets.
`just ci-compat` additionally checks each standalone feature on the development compiler
and the minimum supported Rust version.
It also builds bare-metal targets and executes scalar and SIMD WebAssembly vectors in Wasmtime.
Use the Cargo command above to check an isolated feature selection.

Use [docs.rs](https://docs.rs/rscrypto) to see which items each feature exposes.
