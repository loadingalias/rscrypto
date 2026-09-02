# Features

Select the smallest feature set that exposes the primitives you use.
`Cargo.toml` is the complete, authoritative feature graph.

## Start here

The default feature is `std`; `std` enables `alloc`. Disable defaults for
`no_std`, then name every required primitive:

```toml
# no_std SHA-2
rscrypto = { version = "0.9", default-features = false, features = ["sha2"] }

# Full API with OS randomness
rscrypto = { version = "0.9", features = ["full", "getrandom"] }
```

Umbrella features trade build size for convenience:

| Feature | Includes |
| --- | --- |
| `checksums` | CRC-16, CRC-24, CRC-32, and CRC-64 |
| `crypto-hashes` | SHA-2, SHA-3, BLAKE2, BLAKE3, and Ascon hash |
| `fast-hashes` | XXH3 and RapidHash |
| `hashes` | Cryptographic and fast hashes |
| `auth` | MACs, KDFs, password hashing, signatures, and key exchange |
| `aead` | Every AEAD implementation |
| `full` | Checksums, hashes, authentication, and AEADs |

Prefer leaf features such as `sha2`, `blake3`, `aes-gcm`, `ed25519`, or
`ml-kem` in libraries and constrained builds.

## Capability features

| Feature | Effect |
| --- | --- |
| `alloc` | Enables APIs that own dynamic memory. |
| `std` | Enables runtime CPU detection and standard-library integrations; implies `alloc`. |
| `getrandom` | Enables fallible helpers that obtain keys, nonces, salts, or seeds from the OS. |
| `parallel` | Enables Rayon-backed BLAKE3 and Argon2 work; implies `std`, `blake3`, and `argon2`. |
| `serde` | Serializes public types. |
| `serde-secrets` | Also serializes secret keys and shared secrets; use only at an explicit key-storage boundary. |
| `portable-only` | Makes runtime capability detection report no SIMD or ASM capabilities. |
| `diag` | Exposes unstable diagnostic and evidence hooks; do not use it as application API. |

`getrandom` changes entropy acquisition, not algorithm availability. APIs that
accept caller-provided entropy remain available without it.

`portable-only` affects dispatchers that consult `platform::caps()`. It does
not remove accelerated code from the binary or override backends selected by
compile-time `target_feature` settings. See [`platforms.md`](platforms.md).

## Verify a selection

```sh
cargo check --no-default-features --features sha2
just plan
just check
just feature-contracts compile
```

`just check` asks Cargo Rail for the affected feature groups and runs only the
compile profiles whose resolved Cargo feature graph includes those groups.
`just validate` also runs selected runtime capability profiles. The explicit
`just feature-contracts compile` command remains the full compile contract.

Use [docs.rs](https://docs.rs/rscrypto) to see which items each feature exposes.
