# Examples

Runnable demonstrations of the public `rscrypto` API. Each example is small enough to read end-to-end and feature-gated to the smallest set that compiles.

## Where to start

If you've never used `rscrypto`, run `basic` first — it tours every primitive family the crate exposes. After that, pick the example that matches what you're trying to do.

If you are replacing an existing crate, start with the matching guide in
[`docs/migration/`](../docs/migration/) and then use these examples to confirm
the new API shape.

## All examples

### `basic` — API tour across every family

```bash
cargo run --example basic --features full
```

Walks through checksums (`Crc32C`), digests (`Sha256`, `Blake3`), MACs (`HmacSha256`), KDFs (`HkdfSha256`), XOFs (`Shake256`, `Blake3`), fast hashes (`Xxh3`, `RapidHash`), AEAD (`ChaCha20Poly1305`), hex formatting, secret-key Debug masking, byte-array round-trips through `from_bytes` / `to_bytes` / `as_bytes`, and the `std::io::{Read, Write}` adapters for streaming digests and checksums. Every section asserts that one-shot equals streaming — that's the API contract every primitive in `rscrypto` follows.

### `password_hashing` — Argon2id and scrypt with bounded-policy verify

```bash
cargo run --example password_hashing --features password-hashing,getrandom
```

Hashes a password with both Argon2id and scrypt, encodes the result as a PHC string, then verifies it through `verify_string_with_policy`. Bounded-policy verification is the safe default whenever the encoded hash can come from untrusted storage — the policy caps the cost parameters the verifier will accept, blocking maliciously expensive PHC strings. Prints both PHC encodings to stdout so you can inspect the format.

### `parallel` — CRC chunk combining for large inputs

```bash
cargo run --example parallel --features checksums
```

Demonstrates that `rscrypto`'s CRC implementations are mathematically combinable: given `crc(A)` and `crc(B)`, you can compute `crc(A || B)` without ever holding both chunks together. The example computes a 4 MiB CRC four ways — sequentially, two-way split via `combine`, multi-threaded with `std::thread::scope`, and a sequential multi-part loop — and asserts every result matches. Uses `Crc32` and `Crc64` (XZ polynomial). The pattern applies to any `Checksum` type that implements `ChecksumCombine`.

### `introspect` — Runtime dispatch inspection

```bash
cargo run --example introspect --features checksums,hashes,aead,diag
```

Prints the platform's detected CPU capabilities and reports which kernel the dispatcher selected for each algorithm at representative buffer sizes (64 B, 256 B, 1 KiB, 4 KiB, 64 KiB, 1 MiB). Use this when you want to confirm hardware acceleration kicked in on a new platform, or when you're investigating a performance surprise. Covers every CRC, every cryptographic and fast hash, and every AEAD backend. Requires the `diag` feature — introspection is opt-in to keep the default binary small.

## Pattern reference

| To do this | See |
|---|---|
| Hash data (one-shot or streaming) | `basic` (digest section) |
| Compute and verify a MAC | `basic` (auth section) |
| Encrypt and decrypt with AEAD | `basic` (AEAD section) |
| Hash a password and verify safely | `password_hashing` |
| Process a large file in parallel | `parallel` |
| Confirm hardware acceleration is active | `introspect` |
| Stream a digest through `std::io::Read` | `basic` (I/O adapters section) |

## Beyond examples

- Full API reference: [`docs.rs/rscrypto`](https://docs.rs/rscrypto)
- Algorithm inventory and feature flags: [`README.md`](../README.md)
- Migration guides from RustCrypto, `blake3`, CRC crates, AEADs, signatures, and password hashing: [`docs/migration/`](../docs/migration/)
- Security guidance (nonce lifecycle, PHC verification limits, fallback notes): [`docs/security.md`](../docs/security.md)
- Architecture (modules, dispatch model, internals): [`docs/architecture.md`](../docs/architecture.md)
