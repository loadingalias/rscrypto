# Full Type Inventory

Complete public type inventory for `rscrypto`.
The README keeps only the top-level API map; this file carries the detailed type list.

## Traits

| Trait | Purpose |
|-------|---------|
| `Checksum` | Stateful + one-shot checksums |
| `ChecksumCombine` | O(log n) parallel CRC combine |
| `Digest` | Fixed-output cryptographic hash |
| `Xof` | Variable-output extendable function |
| `Mac` | Keyed streaming MAC |
| `FastHash` | One-shot seeded non-crypto hash |
| `Aead` | Authenticated encryption |
| `ConstantTimeEq` | Constant-time byte equality |

## Checksums

Features: `checksums` or `crc16` / `crc24` / `crc32` / `crc64`.

| Type | Output | Standard |
|------|--------|----------|
| `Crc16Ccitt` / `Crc16Ibm` | `u16` | X.25/HDLC, ARC/IBM |
| `Crc24OpenPgp` | `u32` | RFC 4880 |
| `Crc32` / `Crc32C` | `u32` | Ethernet/gzip, iSCSI/ext4 |
| `Crc64` / `Crc64Nvme` | `u64` | XZ Utils, NVMe |

## Cryptographic Hashes

Features: `crypto-hashes` or `sha2` / `sha3` / `blake2b` / `blake2s` / `blake3` / `ascon-hash`.

| Type | Output | Standard |
|------|--------|----------|
| `Sha224` / `Sha256` / `Sha384` / `Sha512` / `Sha512_256` | 28-64B | FIPS 180-4 |
| `Sha3_224` / `Sha3_256` / `Sha3_384` / `Sha3_512` | 28-64B | FIPS 202 |
| `Shake128` / `Shake256` | XOF | FIPS 202 |
| `Cshake256` | XOF | SP 800-185 |
| `Blake2b256` / `Blake2b512` | 32B / 64B | RFC 7693 |
| `Blake2s128` / `Blake2s256` | 16B / 32B | RFC 7693 |
| `Blake3` | 32B / XOF | BLAKE3 spec |
| `AsconHash256` / `AsconXof` / `AsconCxof128` | 32B / XOF | Ascon v1.2 |

XOF readers: `Shake128XofReader`, `Shake256XofReader`, `Cshake256XofReader`, `Blake3XofReader`, `AsconXofReader`, `AsconCxof128Reader`.

## Fast Hashes

Features: `fast-hashes` or `xxh3` / `rapidhash`.

| Type | Output |
|------|--------|
| `Xxh3` / `Xxh3_128` | `u64` / `u128` |
| `RapidHash` / `RapidHash128` | `u64` / `u128` |

`BuildHasher` support requires `alloc`: `Xxh3BuildHasher`, `RapidBuildHasher`.

## MACs & KDFs

Features: `macs` / `kdfs` or `hmac` / `hkdf` / `pbkdf2` / `kmac`.

| Type | Tag/Output | Standard |
|------|------------|----------|
| `HmacSha256` / `HmacSha384` / `HmacSha512` | 32-64B | RFC 2104 |
| `Kmac256` | variable | SP 800-185 |
| `HkdfSha256` / `HkdfSha384` | 32-48B PRK | RFC 5869 |
| `Pbkdf2Sha256` / `Pbkdf2Sha512` | variable | RFC 2898 / SP 800-132 |

## Password Hashing

Features: `password-hashing` or `argon2` / `scrypt` / `phc-strings`.

| Type | Output | Standard |
|------|--------|----------|
| `Argon2d` / `Argon2i` / `Argon2id` | variable | RFC 9106 |
| `Argon2Params`, `Argon2VerifyPolicy`, `Argon2Version` | -- | RFC 9106 |
| `Scrypt`, `ScryptParams`, `ScryptVerifyPolicy` | variable | RFC 7914 |

PHC string-format encode/decode shared by both families: `auth::phc` (feature `phc-strings`).

## Signatures & Key Exchange

Features: `signatures` / `key-exchange` or `ed25519` / `x25519`.

| Type | Size | Standard |
|------|------|----------|
| `Ed25519SecretKey` / `Ed25519PublicKey` / `Ed25519Signature` | 32/32/64B | RFC 8032 |
| `Ed25519Keypair` | -- | RFC 8032 |
| `X25519SecretKey` / `X25519PublicKey` / `X25519SharedSecret` | 32B each | RFC 7748 |

## AEAD

Feature: `aead` or individual leaves.

| Cipher | Key | Nonce | Tag | Standard |
|--------|-----|-------|-----|----------|
| `Aes128Gcm` | `Aes128GcmKey` 16B | `Nonce96` 12B | `Aes128GcmTag` 16B | SP 800-38D |
| `Aes256Gcm` | `Aes256GcmKey` 32B | `Nonce96` 12B | `Aes256GcmTag` 16B | SP 800-38D |
| `Aes128GcmSiv` | `Aes128GcmSivKey` 16B | `Nonce96` 12B | `Aes128GcmSivTag` 16B | RFC 8452 |
| `Aes256GcmSiv` | `Aes256GcmSivKey` 32B | `Nonce96` 12B | `Aes256GcmSivTag` 16B | RFC 8452 |
| `ChaCha20Poly1305` | `ChaCha20Poly1305Key` 32B | `Nonce96` 12B | `ChaCha20Poly1305Tag` 16B | RFC 8439 |
| `XChaCha20Poly1305` | `XChaCha20Poly1305Key` 32B | `Nonce192` 24B | `XChaCha20Poly1305Tag` 16B | draft-irtf |
| `AsconAead128` | `AsconAead128Key` 16B | `Nonce128` 16B | `AsconAead128Tag` 16B | Ascon v1.2 |
| `Aegis256` | `Aegis256Key` 32B | `Nonce256` 32B | `Aegis256Tag` 16B | draft-irtf |

Nonce types: `Nonce96` (12B), `Nonce128` (16B), `Nonce192` (24B), `Nonce256` (32B).

## Error Types

| Error | When | Recovery |
|-------|------|----------|
| `VerificationError` | MAC/AEAD/signature check fails | Reject input (intentionally opaque) |
| `AeadBufferError` | Output buffer wrong size | Fix buffer length |
| `SealError` | AEAD combined encrypt failure | Buffer / nonce / counter |
| `OpenError` | AEAD combined decrypt failure | Buffer or verification |
| `NonceCounterSealError` | Deterministic-IV counter exhausted or buffer error | Rotate key / fix buffer |
| `HkdfOutputLengthError` | HKDF expand exceeds max | Request less output |
| `Pbkdf2Error` | PBKDF2 parameter validation | Adjust iterations / output length |
| `Argon2Error` | Argon2 parameter or input validation | Adjust params per RFC 9106 |
| `ScryptError` | scrypt parameter validation | Adjust N / r / p per RFC 7914 |
| `PhcError` | PHC string parse / encode | Fix encoded form |
| `X25519Error` | Low-order DH point | Reject peer key |
| `AsconCxofCustomizationError` | Customization > 256 bytes | Shorten string |
| `InvalidHexError` | Hex decode failure | Fix input |
| `platform::OverrideError` | Override after detection init | Set before first call |

## Utility

| Item | Purpose |
|------|---------|
| `ct::constant_time_eq` | Constant-time byte comparison |
| `ct::zeroize` | Volatile-write buffer wipe |
| `DisplaySecret` | Opt-in hex display for secret keys |
