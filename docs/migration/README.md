# Migration Guides

Switch crate-by-crate without hunting through the full API docs.

Each guide shows the smallest useful diff: dependency change, import change,
and call-site change. Code blocks are validated against the latest stable
release of each upstream crate via the equivalence harness at
`/tmp/rscrypto-migration-validate/` (104 byte-equivalent tests at the time of
writing). Every crate in `rscrypto`'s benchmark dev-dependencies has a guide.

If you are just evaluating `rscrypto`, start with the crate you already use.
The guide will tell you whether the migration is a one-line replacement, a small
API shape change, or not a good fit.

## Conventions

- One file per source crate, lowercase, matching the crates.io name.
- Multi-crate organisations live in a directory (e.g. `RustCrypto/sha2.md`).
- Every `// After` block compiles and produces byte-identical output to the `// Before` block it replaces, asserted in the validation harness.

## Checksums

| From | To | Status |
|---|---|---|
| [`crc`](crc.md) | `Crc16Ccitt`, `Crc16Ibm`, `Crc24OpenPgp`, `Crc32`, `Crc32C`, `Crc64`, `Crc64Nvme` | Verified against `crc 3.4.0` |
| [`crc-fast`](crc-fast.md) | `Crc32`, `Crc32C`, `Crc64`, `Crc64Nvme` (catalogue subset) | Verified against `crc-fast 1.10.0` |
| [`crc32fast`](crc32fast.md) | `Crc32` | Verified against `crc32fast 1.5.0` |
| [`crc32c`](crc32c.md) | `Crc32C` | Verified against `crc32c 0.6.8` |
| [`crc64fast`](crc64fast.md) | `Crc64`, `Crc64Nvme` (covers `crc64fast-nvme` aside) | Verified against `crc64fast 1.1.0` and `crc64fast-nvme 1.2.1` |

## Hashes

| From | To | Status |
|---|---|---|
| [`blake3`](blake3.md) | `Blake3` | Verified against `blake3 1.8.5` |
| [`sha2`](RustCrypto/sha2.md) (RustCrypto) | `Sha224`, `Sha256`, `Sha384`, `Sha512`, `Sha512_256` | Verified against `sha2 0.11.0` |
| [`sha3`](RustCrypto/sha3.md) (RustCrypto) | `Sha3_224`, `Sha3_256`, `Sha3_384`, `Sha3_512`, `Shake128`, `Shake256`, `Cshake256` | Verified against `sha3 0.11.0` |
| [`blake2`](RustCrypto/blake2.md) (RustCrypto) | `Blake2b256`, `Blake2b512`, `Blake2s128`, `Blake2s256` | Verified against `blake2 0.10.6` (max stable) |
| [`ascon-hash`](RustCrypto/ascon-hash.md) (RustCrypto) | `AsconHash256`, `AsconXof`, `AsconCxof128` | Verified against `ascon-hash 0.4.0` |
| [`xxhash-rust`](xxhash-rust.md) | `Xxh3`, `Xxh3_128`, `Xxh3Hasher`, `Xxh3BuildHasher` | Verified against `xxhash-rust 0.8.15` |
| [`twox-hash`](twox-hash.md) | `Xxh3`, `Xxh3_128`, `Xxh3Hasher`, `Xxh3BuildHasher` | Verified against `twox-hash 2.1.2` |
| [`rapidhash`](rapidhash.md) | `RapidHash`, `RapidHash128`, `RapidHasher`, `RapidBuildHasher` | Verified against `rapidhash 4.4.1` |

## Auth (MAC + KDF)

| From | To | Status |
|---|---|---|
| [`hmac`](RustCrypto/hmac.md) (RustCrypto) | `HmacSha256`, `HmacSha384`, `HmacSha512` | Verified against `hmac 0.13.0` |
| [`hkdf`](RustCrypto/hkdf.md) (RustCrypto) | `HkdfSha256`, `HkdfSha384` | Verified against `hkdf 0.13.0` |
| [`pbkdf2`](RustCrypto/pbkdf2.md) (RustCrypto) | `Pbkdf2Sha256`, `Pbkdf2Sha512` | Verified against `pbkdf2 0.13.0` |
| [`sha3-kmac`](sha3-kmac.md) | `Kmac256` | Verified against `sha3-kmac 0.3.0` |
| [`tiny-keccak`](tiny-keccak.md) | `Kmac256`, `Cshake256` | Verified against `tiny-keccak 2.0.2` |

## AEAD

| From | To | Status |
|---|---|---|
| [`aes-gcm`](RustCrypto/aes-gcm.md) (RustCrypto) | `Aes128Gcm`, `Aes256Gcm` | Verified against `aes-gcm 0.10.3` |
| [`aes-gcm-siv`](RustCrypto/aes-gcm-siv.md) (RustCrypto) | `Aes128GcmSiv`, `Aes256GcmSiv` | Verified against `aes-gcm-siv 0.11.1` |
| [`chacha20poly1305`](RustCrypto/chacha20poly1305.md) (RustCrypto) | `ChaCha20Poly1305`, `XChaCha20Poly1305` | Verified against `chacha20poly1305 0.10.1` |
| [`ascon-aead`](RustCrypto/ascon-aead.md) (RustCrypto) | `AsconAead128` | Verified against `ascon-aead 0.5.2` |
| [`aegis`](aegis.md) | `Aegis256` | Verified against `aegis 0.9.8` |

## Signatures + Key Exchange

| From | To | Status |
|---|---|---|
| [`ed25519-dalek`](RustCrypto/ed25519-dalek.md) | `Ed25519SecretKey`, `Ed25519PublicKey`, `Ed25519Signature`, `Ed25519Keypair` | Verified against `ed25519-dalek 2.2.0` |
| [`x25519-dalek`](RustCrypto/x25519-dalek.md) | `X25519SecretKey`, `X25519PublicKey`, `X25519SharedSecret` | Verified against `x25519-dalek 2.0.1` |

## Password Hashing

| From | To | Status |
|---|---|---|
| [`argon2`](RustCrypto/argon2.md) (RustCrypto) | `Argon2id`, `Argon2i`, `Argon2d`, `Argon2Params`, `Argon2VerifyPolicy` | Verified against `argon2 0.5.3` |
| [`scrypt`](RustCrypto/scrypt.md) (RustCrypto) | `Scrypt`, `ScryptParams`, `ScryptVerifyPolicy` | Verified against `scrypt 0.12.0` |

## Status

28 migration guides shipped across the checksum, hash, auth, AEAD, signature/KEX, and password-hashing lanes. Every crate in `rscrypto`'s benchmark dev-dependencies has a guide. If your upstream crate is not yet listed, open an issue with the crate name and the API surface you depend on.
