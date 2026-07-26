# Examples

Runnable demonstrations of the public `rscrypto` API. Each example is small enough to read end-to-end and feature-gated to the smallest set that compiles.

## Getting Started

If you've never used `rscrypto`, run `basic` first. It tours the main API patterns across the primitive families. After that, pick the example that matches what you're trying to do.

If you are replacing an existing crate, start with the matching guide in
[`docs/migration/`](../docs/migration/) and then use these examples to confirm
the new API shape.

## All examples

### `basic`: API tour across every family

```bash
cargo run --example basic --features full,getrandom
```

Walks through checksums (`Crc32C`), digests (`Sha256`, `Blake3`), MACs
(`HmacSha256`), KDFs (`HkdfSha256`), XOFs (`Shake256`, `Blake3`), fast hashes
(`Xxh3`, `RapidHash64`), AEAD (`ChaCha20Poly1305` with a fresh random nonce),
hex formatting, secret-key `Debug` masking, byte-array round trips, and
`std::io::{Read, Write}` adapters. The checksum and hash sections compare
one-shot and streaming output.

### `password_hashing`: generated and bounded password records

```bash
cargo run --example password_hashing --features password-hashing,getrandom
```

Generates Argon2id and scrypt password records with fresh OS-random salts, then verifies them through
the bounded `Argon2idPassword` and `ScryptPassword` APIs. Encoded costs outside each verifier's finite
resource envelope are rejected before base64 decoding, allocation, or KDF work. The example prints
both canonical PHC records so you can inspect their format.

### `aead_seal_open`: ChaCha20-Poly1305 seal/open

```bash
cargo run --example aead_seal_open --features chacha20poly1305,getrandom
```

Generates a ChaCha20-Poly1305 key, seals a short payload with associated data
and a fresh random nonce, then opens it back to the original plaintext. This is
the smallest authenticated-encryption example.

### `signatures`: Ed25519 and ECDSA P-256 signing

```bash
cargo run --example signatures --features ed25519,ecdsa-p256,getrandom
```

Generates Ed25519 and ECDSA P-256 keypairs, signs one message with each, and
verifies both signatures through the public-key API. Both example signing paths
are deterministic; the key generation draws from the operating system.

### `rsa_pss_verify`: RSA-PSS fixture verification

```bash
cargo run --example rsa_pss_verify --features rsa
```

Loads a checked-in RSA-3072 SubjectPublicKeyInfo fixture and verifies a PSS/SHA-256 signature over a fixed message. This keeps the example deterministic and focused on the verification path used by certificate, package, and protocol integrations.

### `mlkem_encapsulation`: ML-KEM-768 key encapsulation

```bash
cargo run --example mlkem_encapsulation --features ml-kem,getrandom
```

Generates an ML-KEM-768 keypair, encapsulates a shared secret to the public key,
decapsulates with the private key, and asserts both sides derived the same
bytes. The example does not define a hybrid key-establishment protocol.

### `parallel`: CRC chunk combining for large inputs

```bash
cargo run --example parallel --features checksums
```

Shows how `rscrypto` combines CRC states: given `crc(A)` and `crc(B)`, you can compute `crc(A || B)` without ever holding both chunks together. The example checks a two-way split, a multi-part loop, and scoped-thread chunk processing against sequential references. Uses `Crc32` and `Crc64` (XZ polynomial). The pattern applies to any `Checksum` type that implements `ChecksumCombine`.

### `introspect`: Runtime dispatch inspection

```bash
cargo run --example introspect --features checksums,hashes,aead,diag
```

Prints detected CPU capabilities and the kernel selected for representative
checksums, hashes, fast hashes, and AEADs at specific buffer sizes. Use it to
inspect dispatch on a target or investigate a performance result. The diagnostic
surface requires the opt-in `diag` feature.

## Pattern reference

| To do this | See |
|---|---|
| Hash data (one-shot or streaming) | `basic` (digest section) |
| Compute and verify a MAC | `basic` (auth section) |
| Encrypt and decrypt with AEAD | `aead_seal_open`, `basic` (AEAD section) |
| Sign and verify messages | `signatures`, `rsa_pss_verify` |
| Encapsulate and decapsulate a KEM shared secret | `mlkem_encapsulation` |
| Hash a password and verify safely | `password_hashing` |
| Process a large file in parallel | `parallel` |
| Inspect runtime backend selection | `introspect` |
| Stream a digest through `std::io::Read` | `basic` (I/O adapters section) |

## Beyond examples

- Full API reference: [`docs.rs/rscrypto`](https://docs.rs/rscrypto)
- Algorithm inventory and feature flags: [`README.md`](../README.md)
- Migration guides from RustCrypto, `blake3`, CRC crates, AEADs, signatures, and password hashing: [`docs/migration/`](../docs/migration/)
- Security posture and constant-time boundaries: [`README.md#security`](../README.md#security), [`docs/constant-time.md`](../docs/constant-time.md)
