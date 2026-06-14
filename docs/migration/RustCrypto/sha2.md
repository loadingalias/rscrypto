# Migration: `sha2` (RustCrypto) → `rscrypto`

> Same algorithms (FIPS 180-4), same `Sha224` / `Sha256` / `Sha384` / `Sha512` / `Sha512_256` names, same `new()` / `update()` / `finalize()` shape. The output type changes from `Output<D>` (a `GenericArray`) to `[u8; N]`, and `finalize()` now borrows instead of consuming.

Verified against `sha2 = "0.11.0"` and the `rscrypto` 0.5.0 line.
Evidence: `tests/sha2_official_vectors.rs`, `tests/sha256_differential.rs`, and `tests/sha512_differential.rs`.

## TL;DR

| | Before (`sha2` 0.11.x) | After (`rscrypto` 0.5.0) |
|---|---|---|
| Cargo dep | `sha2 = "0.11"` | `rscrypto = { version = "0.5.0", features = ["sha2"] }` |
| Import | `use sha2::{Sha256, Digest};` | `use rscrypto::{Sha256, Digest};` |
| Call | `Sha256::digest(data)` | `Sha256::digest(data)` |

## Cargo.toml

```toml
# Before
[dependencies]
sha2 = "0.11"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.5.0", features = ["sha2"] }
```

## Algorithm map

| `sha2` type | rscrypto type | Output |
|---|---|---|
| `sha2::Sha224` | `rscrypto::Sha224` | `[u8; 28]` |
| `sha2::Sha256` | `rscrypto::Sha256` | `[u8; 32]` |
| `sha2::Sha384` | `rscrypto::Sha384` | `[u8; 48]` |
| `sha2::Sha512` | `rscrypto::Sha512` | `[u8; 64]` |
| `sha2::Sha512_256` | `rscrypto::Sha512_256` | `[u8; 32]` |

`sha2::Sha512_224` is not currently mapped; if you need it, file an issue.

## API patterns

### One-shot

```rust
// Before
use sha2::{Digest, Sha256};
let out = Sha256::digest(b"123456789");        // Output<Sha256> (GenericArray<u8, U32>)
let bytes: &[u8] = out.as_slice();
```

```rust
// After
use rscrypto::{Digest, Sha256};
let out: [u8; 32] = Sha256::digest(b"123456789");
```

Drop `.as_slice()` / `.as_ref()` calls; the array is the value.

### Streaming

```rust
// Before
use sha2::{Digest, Sha256};
let mut hasher = Sha256::new();
hasher.update(b"foo");
hasher.update(b"bar");
let out = hasher.finalize();                   // consumes hasher
```

```rust
// After
use rscrypto::{Digest, Sha256};
let mut hasher = Sha256::new();
hasher.update(b"foo");
hasher.update(b"bar");
let out = hasher.finalize();                   // borrows &self
```

`finalize()` borrows in rscrypto. You can call it repeatedly to read a running hash, then keep feeding bytes. Use `hasher.reset()` to start over without reallocating.

### Reset

```rust
// Before
use sha2::{Digest, Sha256};
let mut hasher = Sha256::new();
hasher.update(b"throwaway");
hasher.reset();
hasher.update(b"123456789");
let out = hasher.finalize();
```

```rust
// After
use rscrypto::{Digest, Sha256};
let mut hasher = Sha256::new();
hasher.update(b"throwaway");
hasher.reset();
hasher.update(b"123456789");
let out = hasher.finalize();
```

## Notes

- **Output type widening.** `sha2::Output<Sha256>` is `GenericArray<u8, U32>`; rscrypto returns `[u8; 32]`. If you stored hashes in a `Vec<Output<Sha256>>` or compared via `Output::default()`, switch to `[u8; 32]` and `[0u8; 32]` respectively.
- **`finalize` consumes vs. borrows.** RustCrypto's `Digest::finalize(self)` consumes; `rscrypto::Digest::finalize(&self)` borrows. Drop any `.clone()` you added to keep ownership.
- **No `Mac` trait split.** RustCrypto separates `Digest` and `Mac` (for HMAC, etc.). rscrypto's `Digest` trait covers raw hashing only; HMAC / HKDF / KMAC live in their own modules under `rscrypto::auth`. See the [`hmac` migration guide](hmac.md).
- **`digest::Update` etc.** RustCrypto exposes finer-grained traits (`Update`, `FixedOutput`, `Reset`); rscrypto consolidates them in `Digest`. Replace any code generic over `digest::Update` with the equivalent constraint over `rscrypto::Digest`.
- **`generic-array` is gone.** rscrypto returns plain `[u8; N]` and uses no `generic-array` types in its public surface. If a downstream crate of yours had a `where Out: GenericArray<u8, _>` bound, swap to `where Out: AsRef<[u8]>` or accept `[u8; N]` directly.
- **`no_std`.** Both crates support `no_std`. rscrypto's runtime SIMD detection requires `std`; in `no_std` builds it falls back to compile-time `target_feature` selection with a portable backend always present.
