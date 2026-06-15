# Public Type Inventory

Index of the public `rscrypto` types needed to choose imports, features, and
migration targets. The README keeps the top-level API map; this file carries
the root re-exports and documented module helper types.

Use rustdoc for exhaustive API details. This inventory excludes `diag_*`
functions, doc-hidden bench hooks, architecture feature constants, and private
impl types.

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

Prelude: `rscrypto::prelude` re-exports `Aead`, `Checksum`,
`ChecksumCombine`, `ConstantTimeEq`, `Digest`, `FastHash`, `Mac`,
`VerificationError`, and `Xof`.

## Checksums

Features: `checksums` or `crc16` / `crc24` / `crc32` / `crc64`.

| Type | Output | Standard |
|------|--------|----------|
| `Crc16Ccitt` / `Crc16Ibm` | `u16` | X.25/HDLC, ARC/IBM |
| `Crc24OpenPgp` | `u32` | RFC 4880 |
| `Crc32` / `Crc32C` | `u32` | Ethernet/gzip, iSCSI/ext4 |
| `Crc64` / `Crc64Nvme` | `u64` | XZ Utils, NVMe |

Aliases: `checksum::Crc32Ieee`, `checksum::Crc32Castagnoli`, and
`checksum::Crc64Xz`.

Module helpers: `checksum::config::{Crc16Config, Crc16Force, Crc24Config,
Crc24Force, Crc32Config, Crc32Force, Crc64Config, Crc64Force}`,
`checksum::buffered::{BufferedCrc16Ccitt, BufferedCrc16Ibm,
BufferedCrc24OpenPgp, BufferedCrc32, BufferedCrc32C, BufferedCrc64,
BufferedCrc64Nvme}`, and `checksum::io::{ChecksumReader, ChecksumWriter}`.

## Cryptographic Hashes

Features: `crypto-hashes` or `sha2` / `sha3` / `blake2b` / `blake2s` / `blake3` / `ascon-hash`.

| Type | Output | Standard |
|------|--------|----------|
| `Sha224` / `Sha256` / `Sha384` / `Sha512` / `Sha512_256` | 28-64B | FIPS 180-4 |
| `Sha3_224` / `Sha3_256` / `Sha3_384` / `Sha3_512` | 28-64B | FIPS 202 |
| `Shake128` / `Shake256` | XOF | FIPS 202 |
| `Cshake256` | XOF | SP 800-185 |
| `Blake2b`, `Blake2b256`, `Blake2b512`, `Blake2bParams` | 1-64B / 32B / 64B | RFC 7693 |
| `Blake2s128`, `Blake2s256`, `Blake2sParams` | 16B / 32B | RFC 7693 |
| `Blake3`, `Blake3KeyedHash` | 32B / XOF | BLAKE3 spec |
| `AsconHash256` / `AsconXof` / `AsconCxof128` | 32B / XOF | NIST SP 800-232 |

XOF readers: `Shake128XofReader`, `Shake256XofReader`,
`Cshake256XofReader`, `Blake3XofReader`, `AsconXofReader`, and
`AsconCxof128Reader`.

Aliases: `hashes::crypto::AsconXof128` and `hashes::crypto::AsconXof128Reader`.
`hashes::io::{DigestReader, DigestWriter}` provides `std::io` adapters.

## Fast Hashes

Features: `fast-hashes` or `xxh3` / `rapidhash`.

| Type | Output |
|------|--------|
| `Xxh3` / `Xxh3_128` | `u64` / `u128` |
| `RapidHash` / `RapidHash128` | `u64` / `u128` |
| `RapidHashFast64` / `RapidHashFast128` | `u64` / `u128` |

Aliases: `hashes::fast::Xxh3_64` and `hashes::fast::RapidHash64`.

`BuildHasher` support requires `alloc`: `Xxh3BuildHasher`, `Xxh3Hasher`,
`RapidBuildHasher`, and `RapidHasher`.

## MACs & KDFs

Features: `macs` / `kdfs` or `hmac` / `hkdf` / `pbkdf2` / `kmac`.

| Type | Tag/Output | Standard |
|------|------------|----------|
| `HmacSha256` / `HmacSha384` / `HmacSha512`; `HmacSha256Tag` / `HmacSha384Tag` / `HmacSha512Tag` | 32-64B | RFC 2104 |
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

Features: `signatures` / `key-exchange` or `ecdsa` / `ed25519` / `rsa` / `x25519`.

| Type | Size | Standard |
|------|------|----------|
| `EcdsaP256SecretKey` / `EcdsaP256PublicKey` / `EcdsaP256Signature` | secret 32B / SEC1 65B / raw 64B | FIPS 186-5 / SEC 1 |
| `EcdsaP384SecretKey` / `EcdsaP384PublicKey` / `EcdsaP384Signature` | secret 48B / SEC1 97B / raw 96B | FIPS 186-5 / SEC 1 |
| `EcdsaP256Keypair` / `EcdsaP384Keypair` | secret + public | FIPS 186-5 / SEC 1 |
| `Ed25519SecretKey` / `Ed25519PublicKey` / `Ed25519Signature` | 32/32/64B | RFC 8032 |
| `Ed25519Keypair` | -- | RFC 8032 |
| `RsaPublicKey`, `RsaPrivateKey`, `RsaPrivateKeyParts`, `RsaX509PublicKey`, `RsaPublicScratch`, `RsaPrivateScratch` | variable | RFC 8017 / RFC 4055 |
| `RsaSignatureProfile`, `RsaPssProfile`, `RsaPkcs1v15Profile`, `RsaOaepProfile`, `RsaPublicKeyPolicy`, `RsaKeyGenerationContract` | -- | RFC 8017 / RFC 4055 / FIPS 186-5 / protocol-specific profiles |
| `RsaPublicExponent`, `RsaPublicExponentPolicy`, `RsaTlsSignatureSchemes`, `RsaX509PublicKeyAlgorithm` | -- | RSA policy / protocol mapping |
| `X25519SecretKey` / `X25519PublicKey` / `X25519SharedSecret` | 32B each | RFC 7748 |

ECDSA supports P-256/SHA-256 and P-384/SHA-384 signing and verification, raw `r || s` and DER signature import, SEC1/SPKI public keys, deterministic signing, keypair wrappers, and caller-blinded signing APIs for CT-claimed private-key scalar work.
RSA public-key verification, import, and caller-filled public encryption require
`rsa` (`alloc`, `sha2`). OS-backed private operations, key generation, and
randomized encryption wrappers require `getrandom`. Key generation seeds a
key-generation HMAC_DRBG from OS entropy; deterministic caller-supplied
salt/blinding APIs remain available for constrained private-operation
integrations that own their entropy boundary.

## AEAD

Feature: `aead` or individual leaves.

| Cipher | Key | Nonce | Tag | Standard |
|--------|-----|-------|-----|----------|
| `Aes128Gcm` | `Aes128GcmKey` 16B | `Nonce96` 12B | `Aes128GcmTag` 16B | SP 800-38D |
| `Aes256Gcm` | `Aes256GcmKey` 32B | `Nonce96` 12B | `Aes256GcmTag` 16B | SP 800-38D |
| `Aes128GcmSiv` | `Aes128GcmSivKey` 16B | `Nonce96` 12B | `Aes128GcmSivTag` 16B | RFC 8452 |
| `Aes256GcmSiv` | `Aes256GcmSivKey` 32B | `Nonce96` 12B | `Aes256GcmSivTag` 16B | RFC 8452 |
| `ChaCha20Poly1305` | `ChaCha20Poly1305Key` 32B | `Nonce96` 12B | `ChaCha20Poly1305Tag` 16B | RFC 8439 |
| `XChaCha20Poly1305` | `XChaCha20Poly1305Key` 32B | `Nonce192` 24B | `XChaCha20Poly1305Tag` 16B | draft-irtf-cfrg-xchacha |
| `AsconAead128` | `AsconAead128Key` 16B | `Nonce128` 16B | `AsconAead128Tag` 16B | NIST SP 800-232 |
| `Aegis256` | `Aegis256Key` 32B | `Nonce256` 32B | `Aegis256Tag` 16B | draft-irtf-cfrg-aegis-aead |

Nonce types: `Nonce96` (12B), `Nonce128` (16B), `Nonce192` (24B), `Nonce256` (32B).

AEAD support types: `SealError`, `OpenError`, `AeadBufferError`,
`NonceCounter`, `NonceCounterExhausted`, and `NonceCounterSealError`.

## Error Types

| Error | When | Recovery |
|-------|------|----------|
| `VerificationError` | MAC/AEAD/signature check fails | Reject input without revealing failure detail |
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
| `RsaKeyError` | RSA DER or component validation failure | Reject key / tighten import policy |
| `RsaPublicOpError` | RSA public operation input shape/range failure | Fix representative length or reject input |
| `RsaPrivateOpError` | RSA private operation, padding, entropy, or fault-check failure | Reject input; do not expose reason to peer |
| `RsaEncryptionError` | RSA public encryption shape, padding, or entropy failure | Fix input / entropy source |
| `RsaKeyGenerationError` | RSA key generation policy or entropy failure | Adjust key size/policy or entropy source |
| `RsaProtocolAlgorithmError` | Unsupported/confused JWT/COSE/TLS/X.509 RSA selector | Reject algorithm mapping |
| `AsconCxofCustomizationError` | Customization > 256 bytes | Shorten string |
| `InvalidHexError` | Hex decode failure | Fix input |
| `platform::OverrideError` | Override after detection init | Set before first call |

## Platform And Dispatch

| Item | Purpose |
|------|---------|
| `platform::Caps` | 256-bit CPU capability set |
| `platform::Arch` | Detected architecture family |
| `platform::Detected` | Architecture plus capability set |
| `platform::Description` | Zero-allocation display wrapper for detected platform facts |
| `platform::DispatchInfo` | Shared dispatch metadata used by introspection modules |
| `platform::KernelIntrospect` | Trait for algorithms that can report selected kernels by input length |
| `platform::OverrideError` | Detection override failure |

## Utility

| Item | Purpose |
|------|---------|
| `ct::constant_time_eq` | Constant-time byte comparison |
| `ct::zeroize` | Volatile-write buffer wipe |
| `DisplaySecret` | Opt-in hex display for secret keys |
| `SecretBytes<N>` | Fixed-size secret byte buffer that zeroizes on drop |
