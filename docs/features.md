# Feature Flags

Every primitive has its own leaf feature so size-conscious builds compile only what they use. Umbrella features compose the leaves into common bundles, and the `full` feature turns everything on.

## Default

`default = ["std"]`, which implies `alloc`. With `default-features = false`, you get a strict `no_std` build and must opt in to leaf features explicitly.

## Quick Picks

```toml
# One algorithm, no_std.
rscrypto = { version = "0.4.0", default-features = false, features = ["sha2"] }

# RSA public-key import and verification, no_std + alloc.
rscrypto = { version = "0.4.0", default-features = false, features = ["rsa"] }

# RSA key generation, signing, encryption, and private-operation blinding.
rscrypto = { version = "0.4.0", default-features = false, features = ["rsa", "getrandom"] }

# ECDSA P-256/SHA-256 and P-384/SHA-384 signing and verification.
rscrypto = { version = "0.4.0", default-features = false, features = ["ecdsa"] }

# Everything.
rscrypto = { version = "0.4.0", features = ["full", "getrandom"] }

# Everything, with parallel BLAKE3 / Argon2 lanes via Rayon.
rscrypto = { version = "0.4.0", features = ["full", "parallel", "getrandom"] }

# Audit-constrained: makes runtime dispatch ignore host SIMD/ASM capabilities.
rscrypto = { version = "0.4.0", features = ["full", "portable-only"] }
```

## Complete Feature Index

### Core Features

| Feature | Pulls in | Use |
|---|---|---|
| `default` | `std` | Normal server, CLI, and app builds. |
| `std` | `alloc` | Runtime CPU detection and `std::io` adapters. |
| `alloc` | -- | Allocating APIs such as PHC string encoding and `Vec`-returning helpers. |

### Umbrella Features

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
| `signatures` | `ecdsa`, `ed25519`, `rsa` |
| `key-exchange` | `x25519` |
| `aead` | `aes-gcm`, `aes-gcm-siv`, `chacha20poly1305`, `xchacha20poly1305`, `aegis256`, `ascon-aead` |

### Algorithm Leaf Features

| Feature | Pulls in | Enables |
|---|---|---|
| `crc16` | -- | CRC-16/IBM and CRC-16/CCITT |
| `crc24` | -- | CRC-24/OpenPGP |
| `crc32` | -- | CRC-32/IEEE and CRC-32C |
| `crc64` | -- | CRC-64/XZ and CRC-64/NVMe |
| `sha2` | -- | SHA-224, SHA-256, SHA-384, SHA-512, SHA-512/256 |
| `sha3` | -- | SHA3-224/256/384/512, SHAKE128/256, cSHAKE256 |
| `blake2b` | -- | BLAKE2b variable output, BLAKE2b-256, BLAKE2b-512 |
| `blake2s` | -- | BLAKE2s-128, BLAKE2s-256 |
| `blake3` | -- | BLAKE3 hash, keyed hash, and XOF |
| `ascon-hash` | -- | Ascon-Hash256, Ascon-XOF128, Ascon-CXOF128 |
| `xxh3` | -- | XXH3-64 and XXH3-128 |
| `rapidhash` | -- | RapidHash-64 and RapidHash-128 |
| `hmac` | `sha2` | HMAC-SHA256/384/512 |
| `kmac` | `sha3` | KMAC256 |
| `hkdf` | `hmac` | HKDF-SHA256 and HKDF-SHA384 |
| `pbkdf2` | `hmac` | PBKDF2-HMAC-SHA256 and PBKDF2-HMAC-SHA512 |
| `phc-strings` | `alloc` | PHC string encode/decode support |
| `argon2` | `blake2b`, `alloc` | Argon2i, Argon2d, Argon2id |
| `scrypt` | `pbkdf2`, `alloc` | scrypt |
| `ecdsa-p256` | `hmac` | ECDSA P-256/SHA-256 signing and verification |
| `ecdsa-p384` | `hmac` | ECDSA P-384/SHA-384 signing and verification |
| `ecdsa` | `ecdsa-p256`, `ecdsa-p384` | ECDSA P-256/P-384 signing and verification |
| `ed25519` | `sha2` | Ed25519 signatures |
| `rsa` | `alloc`, `sha2` | RSA public/private keys, RSA signatures, OAEP, PKCS#1 v1.5, key generation |
| `x25519` | -- | X25519 key exchange |
| `aes-gcm` | -- | AES-128-GCM and AES-256-GCM |
| `aes-gcm-siv` | -- | AES-128-GCM-SIV and AES-256-GCM-SIV |
| `chacha20poly1305` | -- | ChaCha20-Poly1305 |
| `xchacha20poly1305` | -- | XChaCha20-Poly1305 |
| `aegis256` | -- | AEGIS-256 |
| `ascon-aead` | -- | Ascon-AEAD128 |

### Auxiliary Features

| Feature | Effect |
|---|---|
| `getrandom` | Enables `random()` / `try_random()` constructors via the `getrandom` crate, plus RSA key generation, signing salt/blinding, encryption randomness, and private-operation blinding. ECDSA key generation, caller-blinded signing, and no-std RSA encryption can use caller-supplied byte-filling closures; deterministic ECDSA signing does not use OS randomness. RSA key generation uses OS entropy to seed its key-generation HMAC_DRBG; no separate DRBG feature is required. |
| `serde` | Serde for non-secret byte wrappers (nonces, tags, public keys, signatures). |
| `serde-secrets` | Serde for secret-key and shared-secret bytes. Implies `serde`. Use only for controlled key-material storage, not logs or DTOs. |
| `parallel` | Rayon-backed BLAKE3 and Argon2 lane parallelism. Requires `std`, `blake3`, `argon2`. |
| `diag` | Diagnostic introspection of dispatch decisions and selected benchmark-only component hooks. Requires `std`; hidden diagnostic symbols are not stable application API. |
| `portable-only` | Makes runtime capability detection report no SIMD/ASM capabilities. See below. |

## `portable-only`

`portable-only` makes `platform::caps()` return the empty capability set. Dispatchers that select from runtime capabilities fall through to portable backends instead of invoking host SIMD/ASM kernels. Intended for FIPS, DO-178C, ISO 26262, IEC 62443, and similar deployments where the running code path must ignore host acceleration.

This flag does **not** change `platform::caps_static()`, remove SIMD code from the binary, or create a constant-time proof by itself. For binary-level exclusion, also restrict `target-feature` via `RUSTFLAGS`. For release evidence boundaries, use [`constant-time.md`](constant-time.md).

See [`compliance.md`](compliance.md) for framework-by-framework deployment posture.
