# Migration: `blake3` → `rscrypto`

> Same algorithm, same outputs. Replace `blake3::Hasher` with `rscrypto::Blake3`.
> Unkeyed hashes return `[u8; 32]`; keyed hashes return `Blake3KeyedHash` so
> equality stays constant-time for authenticator use.

Verified against `blake3 = "1.8.5"` and the `rscrypto` 0.5.0 line.
Evidence: `tests/blake3_official_vectors.rs` and `tests/blake3_differential.rs`.

## TL;DR

| | Before (`blake3` 1.x) | After (`rscrypto` 0.5.0) |
|---|---|---|
| Cargo dep | `blake3 = "1.8"` | `rscrypto = { version = "0.5.0", features = ["blake3"] }` |
| Import | `use blake3::Hasher;` | `use rscrypto::{Blake3, prelude::*};` |
| Call | `blake3::hash(data).as_bytes()` | `&Blake3::digest(data)` |

## Cargo.toml

```toml
# Before
[dependencies]
blake3 = "1.8"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.5.0", features = ["blake3"] }
```

## API patterns

### One-shot

```rust
// Before
let hash = blake3::hash(b"123456789");
let bytes: &[u8; 32] = hash.as_bytes();
```

```rust
// After
use rscrypto::{Blake3, prelude::*};
let bytes: [u8; 32] = Blake3::digest(b"123456789");
```

`blake3::hash` returns a `blake3::Hash` newtype (you call `.as_bytes()` to peek inside). `Blake3::digest` returns the `[u8; 32]` directly — the `Digest` trait must be in scope.

### Streaming

```rust
// Before
use blake3::Hasher;
let mut hasher = Hasher::new();
hasher.update(b"foo");
hasher.update(b"bar");
let hash = hasher.finalize();              // blake3::Hash, doesn't consume
```

```rust
// After
use rscrypto::{Blake3, prelude::*};
let mut hasher = Blake3::new();
hasher.update(b"foo");
hasher.update(b"bar");
let hash = hasher.finalize();              // [u8; 32], doesn't consume
```

### Keyed mode (MAC)

```rust
// Before
let key: [u8; 32] = [0x42; 32];
let tag = blake3::keyed_hash(&key, b"message");
let bytes = tag.as_bytes();
```

```rust
// After
use rscrypto::Blake3;
let key: [u8; 32] = [0x42; 32];
let tag = Blake3::keyed_digest(&key, b"message");
assert!(Blake3::verify_keyed(&key, b"message", &tag).is_ok());
```

`tag` is a `Blake3KeyedHash`, not a raw array. Its `==` is constant-time.
Streaming form: `Blake3::new_keyed(&key)` (matches `blake3::Hasher::new_keyed`).

### Key derivation (KDF)

```rust
// Before
let derived = blake3::derive_key("rscrypto-migration-2026", b"input-key-material");
// derived: [u8; 32]
```

```rust
// After
use rscrypto::Blake3;
let derived = Blake3::derive_key("rscrypto-migration-2026", b"input-key-material");
// derived: [u8; 32]
```

Streaming form: `Blake3::new_derive_key(ctx)`.

### XOF (extendable output)

```rust
// Before
use blake3::Hasher;
let mut hasher = Hasher::new();
hasher.update(b"data");
let mut reader = hasher.finalize_xof();
let mut out = [0u8; 96];
reader.fill(&mut out);
```

```rust
// After
use rscrypto::{Blake3, prelude::*};
let mut hasher = Blake3::new();
hasher.update(b"data");
let mut reader = hasher.finalize_xof();
let mut out = [0u8; 96];
reader.squeeze(&mut out);
```

Renames: `reader.fill(&mut out)` → `reader.squeeze(&mut out)`. The one-shot form `Blake3::xof(data)` returns the reader directly.

## Notes

- **Unkeyed arrays vs. keyed tags.** `Blake3::digest` returns `[u8; 32]`
  because unkeyed hashes are usually content IDs, cache keys, or integrity
  fingerprints; ordinary array equality is fine there. For authenticator use,
  `Blake3::keyed_digest` returns `Blake3KeyedHash` and `Blake3::verify_keyed`
  verifies it in constant time.
- **`finalize` borrows in both crates.** `blake3::Hasher::finalize(&self)` and `Blake3::finalize(&self)` are both idempotent — call repeatedly without rebuilding.
- **Parallel hashing.** `blake3` ships a `rayon` feature for multi-threaded chunk hashing. rscrypto exposes the same via the `parallel` umbrella feature (`features = ["blake3", "parallel"]`); the public API is unchanged.
- **`std::io::Write`.** `blake3::Hasher: io::Write`. `rscrypto::Blake3`
  implements `io::Write` when `std` is enabled, and still exposes
  `Blake3::writer(W)` for transparent pass-through writes to an inner writer.
- **`mmap` helpers.** `blake3` ships `Hasher::update_mmap(path)` etc. rscrypto leaves memory-mapping to the caller — wrap with `Blake3::reader(File::open(path)?)` and `io::copy` if you need the same shape.
- **`no_std`.** Both crates support `no_std` with portable fallbacks. rscrypto runtime-detects SIMD when `std` is enabled; force the portable kernel via the `portable-only` feature for FIPS / DO-178C / IEC 62443 lanes.
