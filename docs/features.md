# Feature Flags

Every primitive has its own leaf feature so size-conscious builds compile only what they use. Umbrella features compose the leaves into common bundles, and the `full` feature turns everything on.

## Default

`default = ["std"]`, which implies `alloc`. With `default-features = false`, you get a strict `no_std` build and must opt in to leaf features explicitly.

## Quick Picks

```toml
# One algorithm, no_std.
rscrypto = { version = "0.3.1", default-features = false, features = ["sha2"] }

# RSA public-key import and verification, no_std + alloc.
rscrypto = { version = "0.3.1", default-features = false, features = ["rsa"] }

# RSA key generation, signing, encryption, and private-operation blinding.
rscrypto = { version = "0.3.1", default-features = false, features = ["rsa", "getrandom"] }

# Everything.
rscrypto = { version = "0.3.1", features = ["full", "getrandom"] }

# Everything, with parallel BLAKE3 / Argon2 lanes via Rayon.
rscrypto = { version = "0.3.1", features = ["full", "parallel", "getrandom"] }

# Audit-constrained: forces the portable backend at runtime.
rscrypto = { version = "0.3.1", features = ["full", "portable-only"] }
```

## Umbrella Features

| Feature | Pulls in |
|---|---|
| `full` | `checksums`, `hashes`, `auth`, `aead` |
| `checksums` | `crc16`, `crc24`, `crc32`, `crc64` |
| `hashes` | `crypto-hashes`, `fast-hashes` |
| `crypto-hashes` | `sha2`, `sha3`, `blake2b`, `blake2s`, `blake3`, `ascon-hash` |
| `fast-hashes` | `xxh3`, `rapidhash` |
| `auth` | `macs`, `kdfs`, `password-hashing`, `signatures`, `key-exchange` |
| `macs` | `hmac`, `kmac` |
| `kdfs` | `hkdf`, `pbkdf2` |
| `password-hashing` | `argon2`, `scrypt`, `phc-strings` |
| `signatures` | `ed25519`, `rsa` |
| `key-exchange` | `x25519` |
| `aead` | `aes-gcm`, `aes-gcm-siv`, `chacha20poly1305`, `xchacha20poly1305`, `aegis256`, `ascon-aead` |

## Algorithm Features

| Family | Leaf features (implied dependencies in `→`) |
|---|---|
| Checksums | `crc16`, `crc24`, `crc32`, `crc64` |
| Cryptographic hashes | `sha2`, `sha3`, `blake2b`, `blake2s`, `blake3`, `ascon-hash` |
| Fast hashes | `xxh3`, `rapidhash` |
| MACs | `hmac` → `sha2`; `kmac` → `sha3` |
| KDFs | `hkdf` → `hmac`; `pbkdf2` → `hmac` |
| Password hashing | `argon2` → `blake2b`, `alloc`; `scrypt` → `pbkdf2`, `alloc`; `phc-strings` → `alloc` |
| Signatures | `ed25519` → `sha2`; `rsa` → `alloc`, `sha2` |
| Key exchange | `x25519` |
| AEADs | `aes-gcm`, `aes-gcm-siv`, `chacha20poly1305`, `xchacha20poly1305`, `aegis256`, `ascon-aead` |

## Auxiliary Features

| Feature | Effect |
|---|---|
| `std` | Enables runtime CPU detection and `std` I/O adapters. Implies `alloc`. |
| `alloc` | Enables APIs that allocate (e.g. PHC string encoding, `Vec`-returning helpers). |
| `getrandom` | Enables `random()` / `try_random()` constructors via the `getrandom` crate, plus RSA OS-backed key generation, signing salt/blinding, OAEP encryption randomness, and private-operation blinding. |
| `serde` | Serde for non-secret byte wrappers (nonces, tags, public keys, signatures). |
| `serde-secrets` | Serde for secret-key and shared-secret bytes. Implies `serde`. Use only for controlled key-material storage, not logs or DTOs. |
| `parallel` | Rayon-backed BLAKE3 and Argon2 lane parallelism. Requires `std`, `blake3`, `argon2`. |
| `diag` | Diagnostic introspection of dispatch decisions and selected benchmark-only component hooks. Requires `std`; hidden diagnostic symbols are not stable application API. |
| `portable-only` | Hard-disables runtime SIMD invocation. See below. |

## `portable-only`

`portable-only` makes `platform::caps()` return the empty capability set, so every algorithm's three-tier dispatcher falls through to its portable backend. Intended for FIPS, DO-178C, ISO 26262, IEC 62443, and similar deployments where the running code path must be the constant-time portable implementation regardless of host capabilities.

This flag suppresses *invocation* of SIMD kernels but does **not** remove SIMD code from the binary. For binary-level exclusion, also restrict `target-feature` via `RUSTFLAGS`.

See [`compliance.md`](compliance.md) for framework-by-framework deployment posture.
