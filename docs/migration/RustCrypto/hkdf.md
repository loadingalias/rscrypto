# Migration: `hkdf` (RustCrypto) → `rscrypto`

> Replace `Hkdf::<Sha256>::new(Some(salt), ikm)` with `HkdfSha256::new(salt, ikm)`. The `Option<&[u8]>` salt becomes a plain `&[u8]` (empty slice == "no salt"); fused one-shot `HkdfSha256::derive_array::<N>(...)` collapses extract+expand.

Verified against `hkdf = "0.13.0"` and the `rscrypto` 0.5.0 line.
Evidence: `tests/hkdf_sha256_vectors.rs`, `tests/hkdf_sha384_vectors.rs`, the HKDF proptests, and `tests/hkdf_wycheproof.rs`.

## TL;DR

| | Before (`hkdf` 0.13.x) | After (`rscrypto` 0.5.0) |
|---|---|---|
| Cargo dep | `hkdf = "0.13"` + `sha2 = "0.11"` | `rscrypto = { version = "0.5.0", features = ["hkdf"] }` |
| Import | `use hkdf::Hkdf; use sha2::Sha256;` | `use rscrypto::HkdfSha256;` |
| Call | `Hkdf::<Sha256>::new(Some(salt), ikm).expand(info, &mut okm)?` | `HkdfSha256::new(salt, ikm).expand(info, &mut okm)?` |

## Cargo.toml

```toml
# Before
[dependencies]
hkdf = "0.13"
sha2 = "0.11"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.5.0", features = ["hkdf"] }
```

The `hkdf` feature implies `hmac` which implies `sha2`.

## Algorithm map

| `hkdf` instantiation | rscrypto type | Block size |
|---|---|---|
| `Hkdf<Sha256>` | `HkdfSha256` | 32 bytes |
| `Hkdf<Sha384>` | `HkdfSha384` | 48 bytes |
| `Hkdf<Sha512>` | not currently mapped: file an issue | 64 bytes |

## API patterns

### Extract → expand

```rust
// Before
use hkdf::Hkdf;
use sha2::Sha256;
let hk = Hkdf::<Sha256>::new(Some(b"salt"), b"input-key-material");
let mut okm = [0u8; 42];
hk.expand(b"context", &mut okm).expect("okm length must be <= 255 * HashLen");
```

```rust
// After
use rscrypto::HkdfSha256;
let hk = HkdfSha256::new(b"salt", b"input-key-material");
let mut okm = [0u8; 42];
hk.expand(b"context", &mut okm)?;     // HkdfOutputLengthError on too-long output
```

The salt argument is `&[u8]` directly (not `Option<&[u8]>`). Pass `b""` for "no salt": internally treated as a zero-byte salt of the hash's output length, matching `hkdf::Hkdf::new(None, ikm)`.

### Fused one-shot (extract + expand in one call)

```rust
// Before
use hkdf::Hkdf;
use sha2::Sha256;
let hk = Hkdf::<Sha256>::new(Some(b"salt"), b"ikm");
let mut okm = [0u8; 32];
hk.expand(b"info", &mut okm)?;
```

```rust
// After
use rscrypto::HkdfSha256;
let okm: [u8; 32] = HkdfSha256::derive_array(b"salt", b"ikm", b"info")?;
```

For runtime-variable length, use `HkdfSha256::derive(salt, ikm, info, &mut okm)?`.

### Multiple expansions from the same `(salt, ikm)`

```rust
// Before
use hkdf::Hkdf;
use sha2::Sha256;
let hk = Hkdf::<Sha256>::new(Some(b"salt"), b"ikm");
let mut k_enc = [0u8; 32];
let mut k_mac = [0u8; 32];
hk.expand(b"encryption", &mut k_enc)?;
hk.expand(b"mac", &mut k_mac)?;
```

```rust
// After
use rscrypto::HkdfSha256;
let hk = HkdfSha256::new(b"salt", b"ikm");
let mut k_enc = [0u8; 32];
let mut k_mac = [0u8; 32];
hk.expand(b"encryption", &mut k_enc)?;
hk.expand(b"mac", &mut k_mac)?;
```

Both crates let you reuse the extracted PRK for multiple expansions with different `info` strings: no rebuild needed.

## Notes

- **`Option<&[u8]>` salt → plain `&[u8]`.** RFC 5869 §2.2 says an empty salt is treated as a string of `HashLen` zero bytes. RustCrypto encodes that as `None`; rscrypto encodes it as `b""` (or any zero-length slice). The output is byte-identical for all three.
- **No `Hkdf::from_prk(prk)` constructor (yet).** RustCrypto exposes a way to skip the extract step when you already have a PRK from another source (e.g., a TLS exporter). rscrypto currently always extracts. If you need the "skip extract" path, file an issue.
- **No `verify` method.** HKDF's output is intended to be used as key material, not compared. Both crates omit a `verify` helper; if you need to check that two parties derived the same key, route the comparison through HMAC over the derived key.
- **No reset.** Both crates require constructing a fresh `Hkdf` for each `(salt, ikm)` pair. The `expand` step is stateless once the PRK is computed.
- **Output length cap.** RFC 5869 limits OKM length to `255 × HashLen` (8160 bytes for SHA-256). Both crates return an error on overflow: `hkdf::InvalidLength` upstream, `rscrypto::HkdfOutputLengthError` here. Same `Result` shape; rename the type.
- **`no_std`.** Both crates work in `no_std`. rscrypto adds no `alloc`-only helpers for HKDF: every variant is fixed-array or user-supplied buffer.
