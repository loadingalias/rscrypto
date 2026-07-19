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
| `Kem` | Key encapsulation mechanism profile |
| `FastHash` | One-shot seeded non-crypto hash |
| `Aead` | Authenticated encryption |

Prelude: `rscrypto::prelude` re-exports `Aead`, `Checksum`,
`ChecksumCombine`, `Digest`, `FastHash`, `Kem`, `Mac`,
`VerificationError`, and `Xof`.

## Native API Conventions

- Prefer caller-provided output buffers and scratch buffers when both forms exist.
- Use `alloc` helpers such as `*_to_vec` only when an owned allocation is the right boundary.
- Enable `getrandom` for OS-backed one-liners. Password-record salts are intentionally OS-owned;
  other randomized APIs expose caller-supplied entropy where deterministic or constrained use needs it.
- Bind RSA generic signing and verification through `RsaPrivateKey::signer(profile)` and
  `RsaPublicKey::verifier(profile)` so the padding and hash policy are explicit.
- Bind JWT/JWS verification through `RsaPublicKey::jwt_verifier(algorithm)`. The verifier owns one typed
  `RsaJwtAlgorithm`; peer-controlled `alg` metadata can match that policy but cannot select it.

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
| `Cshake128` / `Cshake256` | XOF | SP 800-185 |
| `Blake2b`, `Blake2b256`, `Blake2b512`, `Blake2bParams` | 1-64B / 32B / 64B | RFC 7693 |
| `Blake2s128`, `Blake2s256`, `Blake2sParams` | 16B / 32B | RFC 7693 |
| `Blake3`, `Blake3KeyedHash` | 32B / XOF | BLAKE3 spec |
| `AsconHash256` / `AsconXof` / `AsconCxof128` | 32B / XOF | NIST SP 800-232 |

XOF readers: `Shake128XofReader`, `Shake256XofReader`,
`Cshake128XofReader`, `Cshake256XofReader`, `Blake3XofReader`, `AsconXofReader`, and
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
| `Xxh3Hasher` / `Xxh3_128Hasher` | streaming `u64` / `u128` |
| `RapidStreamHasher` | streaming `u64` |
| `RapidHasher` | collection-key `u64` |

Aliases: `hashes::fast::Xxh3_64` and `hashes::fast::RapidHash64`.

All fast hashers use bounded inline state and do not allocate.
`RapidStreamHasher` preserves concatenated-stream output; `RapidBuildHasher`
produces `RapidHasher` for collections. Both builders work in pure `no_std`.

## MACs & KDFs

Features: `macs` / `kdfs` or `hmac` / `hmac-sha3` / `hkdf` / `pbkdf2` / `kmac` / `poly1305`.

| Type | Tag/Output | Standard |
|------|------------|----------|
| `HmacSha256` / `HmacSha384` / `HmacSha512`; `HmacSha256Tag` / `HmacSha384Tag` / `HmacSha512Tag` | 32-64B | RFC 2104 |
| `HmacSha3_224` / `HmacSha3_256` / `HmacSha3_384` / `HmacSha3_512`; matching `HmacSha3_*Tag` types | 28-64B | RFC 2104 over FIPS 202 SHA-3 |
| `Kmac128` / `Kmac256` | variable | SP 800-185 |
| `Poly1305`, `Poly1305OneTimeKey`, `Poly1305Tag` | 16B tag | RFC 8439 |
| `HkdfSha256` / `HkdfSha384` / `HkdfSha512` | 32-64B PRK | RFC 5869 |
| `Pbkdf2Sha256` / `Pbkdf2Sha512` | variable | RFC 2898 / SP 800-132 |

## Password Hashing

Features: `password-hashing` or `argon2` / `scrypt` / `phc-strings`.

| Type | Output | Standard |
|------|--------|----------|
| `Argon2d` / `Argon2i` / `Argon2id` | variable | RFC 9106 |
| `Argon2Params`, `Argon2Context` | -- | RFC 9106 raw-KDF configuration |
| `Argon2idPassword`, `Argon2VerificationLimits` | 32B verifier | Bounded canonical Argon2id PHC records |
| `Scrypt`, `ScryptParams` | variable | RFC 7914 raw KDF |
| `ScryptPassword`, `ScryptVerificationLimits` | 32B verifier | Bounded canonical scrypt PHC records |
| `PasswordStatus` | -- | Current-profile / rehash decision |

Password-record operations require `phc-strings`; OS-salted generation also requires `getrandom`.
PHC parsing and encoding are intentionally internal so attacker-controlled costs cannot bypass the
algorithm-specific verification limits.

## Signatures & Key Exchange

Features: `signatures` / `key-exchange` or `ecdsa` / `ed25519` / `rsa` / `x25519` / `ml-kem`.

| Type | Size | Standard |
|------|------|----------|
| `EcdsaP256SecretKey` / `EcdsaP256PublicKey` / `EcdsaP256Signature` | secret 32B / SEC1 65B / raw 64B | FIPS 186-5 / SEC 1 |
| `EcdsaP384SecretKey` / `EcdsaP384PublicKey` / `EcdsaP384Signature` | secret 48B / SEC1 97B / raw 96B | FIPS 186-5 / SEC 1 |
| `EcdsaP256Keypair` / `EcdsaP384Keypair` | secret + public | FIPS 186-5 / SEC 1 |
| `Ed25519SecretKey` / `Ed25519PublicKey` / `Ed25519Signature` | 32/32/64B | RFC 8032 |
| `Ed25519Keypair` | -- | RFC 8032 |
| `RsaPublicKey`, `RsaPrivateKey`, `RsaPrivateKeyParts`, `RsaX509PublicKey`, `RsaPublicScratch`, `RsaPrivateScratch` | variable | RFC 8017 / RFC 4055 |
| `RsaSignatureSigner`, `RsaSignatureVerifier` | profile-bound wrappers | RFC 8017 / RFC 4055 |
| `RsaJwtAlgorithm`, `RsaJwtVerifier` | verifier-owned JWT/JWS policy | RFC 7515 / RFC 8725 |
| `RsaSignatureProfile`, `RsaPssProfile`, `RsaPkcs1v15Profile`, `RsaOaepProfile`, `RsaPublicKeyPolicy`, `RsaKeyGenerationContract` | -- | RFC 8017 / RFC 4055 / FIPS 186-5 / protocol-specific profiles |
| `RsaPublicExponent`, `RsaPublicExponentPolicy`, `RsaTlsSignatureSchemes`, `RsaX509PublicKeyAlgorithm` | -- | RSA policy / protocol mapping |
| `X25519SecretKey` / `X25519PublicKey` / `X25519SharedSecret` | 32B each | RFC 7748 |
| `MlKem512` / `MlKem768` / `MlKem1024` | profile types | FIPS 203 |
| `MlKem512EncapsulationKey` / `MlKem512DecapsulationKey` / `MlKem512Ciphertext` / `MlKem512SharedSecret` | 800B / 1632B / 768B / 32B | FIPS 203 ML-KEM-512 |
| `MlKem768EncapsulationKey` / `MlKem768DecapsulationKey` / `MlKem768Ciphertext` / `MlKem768SharedSecret` | 1184B / 2400B / 1088B / 32B | FIPS 203 ML-KEM-768 |
| `MlKem1024EncapsulationKey` / `MlKem1024DecapsulationKey` / `MlKem1024Ciphertext` / `MlKem1024SharedSecret` | 1568B / 3168B / 1568B / 32B | FIPS 203 ML-KEM-1024 |
| `MlKem512PreparedEncapsulationKey` / `MlKem512PreparedDecapsulationKey` | 800B / 1632B | Validated reusable ML-KEM-512 state |
| `MlKem768PreparedEncapsulationKey` / `MlKem768PreparedDecapsulationKey` | 1184B / 2400B | Validated reusable ML-KEM-768 state |
| `MlKem1024PreparedEncapsulationKey` / `MlKem1024PreparedDecapsulationKey` | 1568B / 3168B | Validated reusable ML-KEM-1024 state |

ECDSA supports P-256/SHA-256 and P-384/SHA-384 signing and verification, raw
`r || s` and DER signature import, SEC1/SPKI public keys, deterministic signing,
bounded-retry `try_generate_with` / `try_generate` key generation, keypair
wrappers, and caller-blinded signing APIs for CT-claimed private-key scalar
work.
RSA public-key verification, import, and caller-filled public encryption require
`rsa` (`alloc`, `sha2`). OS-backed private operations, key generation, and
randomized encryption wrappers require `getrandom`. Key generation seeds a
key-generation HMAC_DRBG from OS entropy; deterministic caller-supplied
salt/blinding APIs remain available for constrained private-operation
integrations that own their entropy boundary.

ML-KEM supports key generation, encapsulation, decapsulation, validated prepared
encapsulation keys, and validated prepared decapsulation keys. The core API
takes caller-supplied random-fill closures for key generation and encapsulation,
so `ml-kem` does not require `getrandom`; `try_generate_keypair` and
`try_encapsulate` are available when `getrandom` is enabled. Each profile
exposes FIPS 203 size, randomness, security-category, and required-RBG-strength
constants.

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
Caller-buffer and detached APIs are allocation-free. With `alloc`, every AEAD
also has `encrypt_to_vec` and `decrypt_to_vec`; with `alloc` + `getrandom`, it
also has `seal_random_to_vec`.

## Error Types

| Error | When | Recovery |
|-------|------|----------|
| `VerificationError` | MAC/AEAD/signature/password verification fails | Reject input without revealing failure detail |
| `EcdsaKeyGenerationError` | ECDSA random source failure or bounded scalar rejection exhaustion | Fix entropy source; investigate deterministic fillers |
| `AeadBufferError` | Output buffer wrong size | Fix buffer length |
| `SealError` | AEAD combined encrypt failure | Buffer / nonce / counter |
| `OpenError` | AEAD combined decrypt failure | Buffer or verification |
| `NonceCounterSealError` | Deterministic-IV counter exhausted or buffer error | Rotate key / fix buffer |
| `HkdfOutputLengthError` | HKDF expand exceeds max | Request less output |
| `Pbkdf2Error` | PBKDF2 parameter validation | Adjust iterations / output length |
| `Argon2Error` | Argon2 configuration, input, entropy, or resource failure | Fix the profile/input or restore resources |
| `ScryptError` | scrypt configuration, entropy, or resource failure | Fix N/r/p or restore resources |
| `X25519Error` | Low-order DH point | Reject peer key |
| `MlKemError` | ML-KEM random source, key, or ciphertext validation failure | Reject input or fix entropy source |
| `RsaKeyError` | RSA DER or component validation failure | Reject key / tighten import policy |
| `RsaPublicOpError` | RSA public operation input shape/range failure | Fix representative length or reject input |
| `RsaPrivateOpError` | RSA private operation, padding, entropy, or fault-check failure | Reject input; do not expose reason to peer |
| `RsaEncryptionError` | RSA public encryption shape, padding, or entropy failure | Fix input / entropy source |
| `RsaKeyGenerationError` | RSA key generation policy or entropy failure | Adjust key size/policy or entropy source |
| `RsaProtocolAlgorithmError` | Unsupported/confused COSE/TLS/X.509 RSA selector | Reject algorithm mapping |
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
| `ct::zeroize` | Volatile-write buffer wipe |
| `DisplaySecret` | Opt-in hex display for secret keys |
| `SecretBytes<N>` | Fixed-size secret byte buffer that zeroizes on drop |
| `SecretVec` | Variable-length secret allocation that zeroizes on drop; ordinary extraction requires `into_unprotected_vec()` |

Generic secret wrappers deliberately do not implement equality. Fixed-size
keys, shared secrets, authentication tags, and keyed outputs compare only
through their concrete semantic owner types.

Secret keys, shared secrets, keypairs, and AEAD cipher contexts do not implement
`Clone`. Where duplication is necessary, call the concrete type's
`duplicate_secret()` method so the additional secret lifetime is visible at the
call site. RSA private-key DER exports return `SecretVec`; borrow the encoded
bytes for parsers and writers whenever possible.
