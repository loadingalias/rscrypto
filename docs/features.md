# Feature Flags

Each primitive family has a leaf feature so applications can select a narrow
surface. Umbrella features compose common bundles; `full` enables the complete
public primitive surface.

## Default

The default feature set is `["std"]`; `std` implies `alloc`. Set
`default-features = false` for a `no_std` build, then enable every required
algorithm feature explicitly.

## Quick picks

```toml
# One algorithm, no_std.
rscrypto = { version = "0.8.1", default-features = false, features = ["sha2"] }

# RSA public-key import and verification, no_std + alloc.
rscrypto = { version = "0.8.1", default-features = false, features = ["rsa"] }

# RSA key generation, signing, encryption, and private-operation blinding.
rscrypto = { version = "0.8.1", default-features = false, features = ["rsa", "getrandom"] }

# ECDSA P-256/SHA-256 and P-384/SHA-384 signing and verification.
rscrypto = { version = "0.8.1", default-features = false, features = ["ecdsa"] }

# FIPS 203 ML-KEM-512/768/1024 KEM APIs with caller-supplied randomness.
rscrypto = { version = "0.8.1", default-features = false, features = ["ml-kem"] }

# Everything.
rscrypto = { version = "0.8.1", features = ["full", "getrandom"] }

# Everything, with parallel BLAKE3 / Argon2 lanes via Rayon.
rscrypto = { version = "0.8.1", features = ["full", "parallel", "getrandom"] }

# Audit-constrained: makes runtime dispatch ignore host SIMD/ASM capabilities.
rscrypto = { version = "0.8.1", features = ["full", "portable-only"] }
```

## Complete feature index

### Core features

| Feature | Pulls in | Use |
|---|---|---|
| `default` | `std` | Normal server, CLI, and app builds. |
| `std` | `alloc` | Runtime CPU detection and `std::io` adapters. |
| `alloc` | -- | Allocating APIs such as PHC string encoding and `Vec`-returning digest, MAC, AEAD, and signature helpers. |

### Umbrella features

| Feature | Pulls in |
|---|---|
| `full` | `checksums`, `hashes`, `auth`, `aead` |
| `checksums` | `crc16`, `crc24`, `crc32`, `crc64` |
| `hashes` | `crypto-hashes`, `fast-hashes` |
| `crypto-hashes` | `sha2`, `sha3`, `blake2b`, `blake2s`, `blake3`, `ascon-hash` |
| `fast-hashes` | `xxh3`, `rapidhash` |
| `auth` | `macs`, `kdfs`, `password-hashing`, `signatures`, `key-exchange` |
| `macs` | `hmac`, `hmac-sha3`, `kmac`, `poly1305` |
| `kdfs` | `hkdf`, `pbkdf2` |
| `password-hashing` | `argon2`, `scrypt`, `phc-strings` |
| `signatures` | `ecdsa`, `ed25519`, `rsa` |
| `key-exchange` | `x25519`, `ml-kem` |
| `aead` | `aes-gcm`, `aes-gcm-siv`, `chacha20poly1305`, `xchacha20poly1305`, `aegis256`, `ascon-aead` |

### Algorithm leaf features

| Feature | Pulls in | Enables |
|---|---|---|
| `crc16` | -- | CRC-16/IBM and CRC-16/CCITT |
| `crc24` | -- | CRC-24/OpenPGP |
| `crc32` | -- | CRC-32/IEEE and CRC-32C |
| `crc64` | -- | CRC-64/XZ and CRC-64/NVMe |
| `sha2` | -- | SHA-224, SHA-256, SHA-384, SHA-512, SHA-512/256 |
| `sha3` | -- | SHA3-224/256/384/512, SHAKE128/256, cSHAKE128/256 |
| `blake2b` | -- | BLAKE2b variable output, BLAKE2b-256, BLAKE2b-512 |
| `blake2s` | -- | BLAKE2s-128, BLAKE2s-256 |
| `blake3` | -- | BLAKE3 hash, keyed hash, and XOF |
| `ascon-hash` | -- | Ascon-Hash256, Ascon-XOF128, Ascon-CXOF128 |
| `xxh3` | -- | XXH3-64 and XXH3-128 |
| `rapidhash` | -- | Portable RapidHash V3-64, streaming, and collection state |
| `hmac` | `sha2` | HMAC-SHA256/384/512 |
| `hmac-sha3` | `sha3` | HMAC-SHA3-224/256/384/512 |
| `kmac` | `sha3` | KMAC128 and KMAC256 |
| `hkdf` | `hmac` | HKDF-SHA256, HKDF-SHA384, and HKDF-SHA512 |
| `poly1305` | -- | Standalone Poly1305 one-time MAC |
| `pbkdf2` | `hmac` | PBKDF2-HMAC-SHA256 and PBKDF2-HMAC-SHA512 |
| `phc-strings` | `alloc` | Canonical password-record generation and bounded PHC verification |
| `argon2` | `blake2b`, `alloc` | Argon2i, Argon2d, Argon2id |
| `scrypt` | `pbkdf2`, `alloc` | scrypt |
| `ecdsa-p256` | `hmac` | ECDSA P-256/SHA-256 signing and verification |
| `ecdsa-p384` | `hmac` | ECDSA P-384/SHA-384 signing and verification |
| `ecdsa` | `ecdsa-p256`, `ecdsa-p384` | ECDSA P-256/P-384 signing and verification |
| `ed25519` | `sha2` | Ed25519 signatures |
| `rsa` | `alloc`, `sha2` | RSA public/private keys, RSA signatures, OAEP, PKCS#1 v1.5, key generation |
| `x25519` | -- | X25519 key exchange |
| `ml-kem` | `sha3` | ML-KEM-512, ML-KEM-768, and ML-KEM-1024 key encapsulation |
| `aes-gcm` | -- | AES-128-GCM and AES-256-GCM |
| `aes-gcm-siv` | -- | AES-128-GCM-SIV and AES-256-GCM-SIV |
| `chacha20poly1305` | -- | ChaCha20-Poly1305 |
| `xchacha20poly1305` | -- | XChaCha20-Poly1305 |
| `aegis256` | -- | AEGIS-256 |
| `ascon-aead` | -- | Ascon-AEAD128 |

### Auxiliary features

| Feature | Effect |
|---|---|
| `getrandom` | Adds OS-backed random generation; see below. |
| `serde` | Serde for non-secret byte wrappers (nonces, tags, public keys, signatures). |
| `serde-secrets` | Serde for secret-key and shared-secret bytes. Implies `serde`. Use only for controlled key-material storage, not logs or DTOs. |
| `parallel` | Rayon-backed BLAKE3 and Argon2 lane parallelism. Requires `std`, `blake3`, `argon2`. |
| `diag` | Diagnostic introspection of dispatch decisions and selected benchmark-only component hooks. Requires `std`; hidden diagnostic symbols are not stable application API. |
| `portable-only` | Makes runtime capability detection report no SIMD/ASM capabilities. See below. |

## `getrandom`

`getrandom` enables fallible OS-backed constructors such as `try_random()` and
`try_generate()`. It also enables `RapidRandomState::try_new()`, OS-salted
Argon2id and scrypt password-record generation, ML-KEM
`try_generate_keypair()` and `try_encapsulate()`, AEAD random sealing, and RSA
key generation, signing salt and blinding, encryption randomness, and
private-operation blinding.

`Argon2idPassword::hash_password_with` and
`ScryptPassword::hash_password_with` accept a fallible entropy-filling closure
without enabling `getrandom`; rscrypto still owns the fixed salt buffer and PHC
encoding. Other random-generation APIs retain equivalent byte-filling closures
for constrained integrations and deterministic tests. Deterministic ECDSA
signing does not use OS randomness. RSA key generation uses OS entropy to seed
its HMAC_DRBG; no separate DRBG feature is required.

## `portable-only`

`portable-only` makes `platform::caps()` return the empty capability set.
Dispatchers that consult runtime capabilities therefore fall through to
portable backends instead of invoking host SIMD/ASM kernels. Use it when a
deployment requires runtime dispatch to ignore host acceleration.

This flag does **not** change `platform::caps_static()`, override a backend
selected at compile time, remove accelerated code from the binary, or create a
constant-time proof. Restrict `target-feature` through `RUSTFLAGS` when the
binary must exclude compile-time accelerated paths. Use
[`constant-time.md`](constant-time.md) for release evidence boundaries.

See [`compliance.md`](compliance.md) for the FIPS-oriented deployment boundary.
