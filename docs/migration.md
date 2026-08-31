# Migrating to rscrypto

Migrate one primitive at a time. `rscrypto` is not API-compatible with other
cryptography crates, and replacing a toolkit does not replace its protocol,
certificate, key-storage, or compliance behavior.

## Choose features

Disable default features for `no_std`, then enable only the primitives you use:

```toml
rscrypto = { version = "0.9", default-features = false, features = ["sha2"] }
```

Common source crates map as follows:

| Source | `rscrypto` feature |
| --- | --- |
| `aes-gcm`, `aes-gcm-siv`, `chacha20poly1305`, `ascon-aead`, `aegis` | `aes-gcm`, `aes-gcm-siv`, `chacha20poly1305`, `xchacha20poly1305`, `ascon-aead`, or `aegis256` |
| `sha2`, `sha3`, `blake2`, `blake3`, `tiny-keccak`, `sha3-kmac` | `sha2`, `sha3`, `blake2b`, `blake2s`, `blake3`, or `kmac` |
| `hmac`, `hkdf`, `pbkdf2`, `argon2`, `scrypt` | `hmac`, `hkdf`, `pbkdf2`, `argon2`, `scrypt`, and `phc-strings` as needed |
| `p256`, `p384`, `ed25519-dalek`, `x25519-dalek`, `rsa` | `ecdsa-p256`, `ecdsa-p384`, `ed25519`, `x25519`, or `rsa` |
| `crc`, `crc-fast`, `crc32fast`, `crc32c`, `crc64fast` | `crc16`, `crc24`, `crc32`, or `crc64` |
| `xxhash-rust`, `twox-hash`, `rapidhash` | `xxh3` or `rapidhash` |

`aws-lc-rs`, `ring`, `dryoc`, and `openssl` are broader toolkits. Map each
primitive separately and keep their protocol or certificate work outside
`rscrypto`. `aws-lc-sys` has no direct replacement because `rscrypto` exposes
Rust APIs, not AWS-LC symbols.

## Review these API boundaries

- Digests return fixed Rust arrays and `finalize` borrows the hasher. Reset or
  create a new hasher before processing another message.
- Keys, tags, signatures, ciphertexts, and shared secrets use distinct types.
  Convert at the input boundary instead of carrying generic byte buffers.
- Combined AEAD output includes the authentication tag; detached APIs return it
  separately. Opening requires the same nonce and associated data. Failed
  in-place opens clear unauthenticated plaintext.
- Random key and nonce helpers require `getrandom`. Without it, supply entropy
  explicitly and preserve uniqueness requirements.
- X25519 rejects an all-zero shared secret. Feed successful output into a KDF
  that binds the protocol transcript; do not use the raw secret as a key.
- Password helpers validate and emit bounded PHC strings. Set an application
  policy for parameters, accepted algorithms, and rehashing.
- Caller-controlled nonce operations and other sharp tools live under
  `expert`; ordinary callers should use the root API.

## Verify the migration

Run the example closest to the operation, then test old and new implementations
against the same vectors before removing the old dependency:

```sh
just test-examples
just test --all
```

See [`../examples/README.md`](../examples/README.md) for runnable workflows,
[`features.md`](features.md) for build selection, and
[docs.rs](https://docs.rs/rscrypto) for exact types and methods.
