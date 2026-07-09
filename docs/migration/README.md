# Migration Guides

Switch crate-by-crate without hunting through the full API docs. This index
covers 35 migration guides.

Each guide shows the smallest useful diff: dep change, import change,
and call site change.

If you are evaluating `rscrypto`, start with the crate you already use.
The guide will tell you whether the migration is a one-line replacement, a small
API shape change, or not a good fit.

## Checksums

| From | To | Status |
|---|---|---|
| [`crc`](crc.md) | `Crc16Ccitt`, `Crc16Ibm`, `Crc24OpenPgp`, `Crc32`, `Crc32C`, `Crc64`, `Crc64Nvme` | Verified against `crc 3.4.0` |
| [`crc-fast`](crc-fast.md) | `Crc32`, `Crc32C`, `Crc64`, `Crc64Nvme` (catalogue subset) | Verified against `crc-fast 1.10.0` |
| [`crc32fast`](crc32fast.md) | `Crc32` | Output covered by CRC-32 oracle/property tests |
| [`crc32c`](crc32c.md) | `Crc32C` | Output covered by CRC-32C oracle/property tests |
| [`crc64fast`](crc64fast.md) | `Crc64`, `Crc64Nvme` (covers `crc64fast-nvme` aside) | Verified against `crc64fast 1.1.0`; NVMe oracle coverage uses `crc-fast 1.10.0` |

## Hashes

| From | To | Status |
|---|---|---|
| [`blake3`](blake3.md) | `Blake3` | Verified against `blake3 1.8.5` |
| [`sha2`](RustCrypto/sha2.md) (RustCrypto) | `Sha224`, `Sha256`, `Sha384`, `Sha512`, `Sha512_256` | Verified against `sha2 0.11.0` |
| [`sha3`](RustCrypto/sha3.md) (RustCrypto) | `Sha3_224`, `Sha3_256`, `Sha3_384`, `Sha3_512`, `Shake128`, `Shake256`, `Cshake128`, `Cshake256` | Verified against `sha3 0.12.0` |
| [`blake2`](RustCrypto/blake2.md) (RustCrypto) | `Blake2b256`, `Blake2b512`, `Blake2s128`, `Blake2s256` | Verified against `blake2 0.11.0-rc.6` |
| [`ascon-hash`](RustCrypto/ascon-hash.md) (RustCrypto) | `AsconHash256`, `AsconXof`, `AsconCxof128` | Verified against `ascon-hash 0.4.0` |
| [`xxhash-rust`](xxhash-rust.md) | `Xxh3`, `Xxh3_128`, `Xxh3Hasher`, `Xxh3BuildHasher` | Verified against `xxhash-rust 0.8.15` |
| [`twox-hash`](twox-hash.md) | `Xxh3`, `Xxh3_128`, `Xxh3Hasher`, `Xxh3BuildHasher` | API migration guidance; XXH3 output covered by `xxhash-rust` oracle tests |
| [`rapidhash`](rapidhash.md) | `RapidHash`, `RapidHash128`, `RapidHasher`, `RapidBuildHasher` | Verified against `rapidhash 4.4.1` |

## Auth (MAC + KDF)

| From | To | Status |
|---|---|---|
| [`hmac`](RustCrypto/hmac.md) (RustCrypto) | `HmacSha256`, `HmacSha384`, `HmacSha512`, `HmacSha3_224`, `HmacSha3_256`, `HmacSha3_384`, `HmacSha3_512` | Verified against `hmac 0.13.0` and RustCrypto SHA-3 digests |
| [`hkdf`](RustCrypto/hkdf.md) (RustCrypto) | `HkdfSha256`, `HkdfSha384`, `HkdfSha512` | Verified against `hkdf 0.13.0` |
| [`pbkdf2`](RustCrypto/pbkdf2.md) (RustCrypto) | `Pbkdf2Sha256`, `Pbkdf2Sha512` | Verified against `pbkdf2 0.13.0` |
| [`sha3-kmac`](sha3-kmac.md) | `Kmac128`, `Kmac256` | KMAC128/256 covered by NIST and `tiny-keccak`; KMAC256 also has Wycheproof coverage |
| [`tiny-keccak`](tiny-keccak.md) | `Kmac128`, `Kmac256`, `Cshake128`, `Cshake256` | Verified against `tiny-keccak 2.0.2` |

## AEAD

| From | To | Status |
|---|---|---|
| [`aes-gcm`](RustCrypto/aes-gcm.md) (RustCrypto) | `Aes128Gcm`, `Aes256Gcm` | Verified against `aes-gcm 0.10.3` |
| [`aes-gcm-siv`](RustCrypto/aes-gcm-siv.md) (RustCrypto) | `Aes128GcmSiv`, `Aes256GcmSiv` | Verified against `aes-gcm-siv 0.11.1` |
| [`chacha20poly1305`](RustCrypto/chacha20poly1305.md) (RustCrypto) | `ChaCha20Poly1305`, `XChaCha20Poly1305` | Verified against `chacha20poly1305 0.10.1` |
| [`ascon-aead`](RustCrypto/ascon-aead.md) (RustCrypto) | `AsconAead128` | Verified against `ascon-aead 0.5.2` |
| [`aegis`](aegis.md) | `Aegis256` | Verified against `aegis 0.9.12` |

## Signatures + Key Exchange

| From | To | Status |
|---|---|---|
| [`p256`](RustCrypto/p256.md) / [`p384`](RustCrypto/p384.md) (RustCrypto) | `EcdsaP256SecretKey`, `EcdsaP384SecretKey`, `EcdsaP256PublicKey`, `EcdsaP384PublicKey`, raw/DER signatures | Signing and verification tested against RustCrypto `p256 0.13.2` / `p384 0.13.1` |
| [`ed25519-dalek`](RustCrypto/ed25519-dalek.md) | `Ed25519SecretKey`, `Ed25519PublicKey`, `Ed25519Signature`, `Ed25519Keypair` | Verified against `ed25519-dalek 2.2.0` |
| [`rsa`](RustCrypto/rsa.md) (RustCrypto) | `RsaPublicKey`, `RsaPrivateKey`, RSA-PSS, RSASSA-PKCS1-v1_5, OAEP | Partial; verified through CAVP, Wycheproof, and RustCrypto/ring/OpenSSL oracles |
| [`x25519-dalek`](RustCrypto/x25519-dalek.md) | `X25519SecretKey`, `X25519PublicKey`, `X25519SharedSecret` | Verified against `x25519-dalek 2.0.1` |

## Password Hashing

| From | To | Status |
|---|---|---|
| [`argon2`](RustCrypto/argon2.md) (RustCrypto) | `Argon2id`, `Argon2i`, `Argon2d`, `Argon2Params`, `Argon2VerifyPolicy` | Verified against `argon2 0.6.0-rc.8` |
| [`scrypt`](RustCrypto/scrypt.md) (RustCrypto) | `Scrypt`, `ScryptParams`, `ScryptVerifyPolicy` | Verified against `scrypt 0.12.0` |

## Stack Migrations

| From | To | Status |
|---|---|---|
| [`aws-lc-rs`](aws-lc-rs.md) | AEAD, SHA-2, HMAC, HKDF, PBKDF2, ECDSA, Ed25519, X25519, RSA verify | Partial; shape-compatible surfaces only |
| [`aws-lc-sys`](aws-lc-sys.md) | none directly | Not a direct migration; replace the safe wrapper API instead |
| [`dryoc`](dryoc.md) | Ed25519, X25519, BLAKE2, Argon2id/Argon2i-adjacent surfaces | Partial; libsodium-style APIs are not one-to-one |
| [`ring`](ring.md) | AEAD, SHA-2, HMAC, HKDF, PBKDF2, ECDSA, Ed25519, RSA verify | Partial; `ring` is protocol-shaped in several areas |
| [`openssl`](openssl.md) | selected hash, MAC, AEAD, RSA operations | Partial; rscrypto does not replace TLS, PKI, engines, or providers |
