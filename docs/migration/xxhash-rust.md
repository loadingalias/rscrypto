# Migration: `xxhash-rust` → `rscrypto`

> Replace `xxhash_rust::xxh3::xxh3_64` with `rscrypto::Xxh3::hash` (and `xxh3_128` with `Xxh3_128::hash`). One-shot bit-equivalent. Streaming routes through `Xxh3Hasher` (a `core::hash::Hasher`).

Verified against `xxhash-rust = "0.8.15"` (with `xxh3` feature) and the `rscrypto` 0.3.1 line.
Evidence: `tests/xxh3_differential.rs`.

## TL;DR

| | Before (`xxhash-rust` 0.8.x) | After (`rscrypto` 0.3.1) |
|---|---|---|
| Cargo dep | `xxhash-rust = { version = "0.8", features = ["xxh3"] }` | `rscrypto = { version = "0.3.1", features = ["xxh3"] }` |
| Import | `use xxhash_rust::xxh3::xxh3_64;` | `use rscrypto::{FastHash, Xxh3};` |
| Call | `xxh3_64(data)` | `Xxh3::hash(data)` |

## Cargo.toml

```toml
# Before
[dependencies]
xxhash-rust = { version = "0.8", features = ["xxh3"] }
```

```toml
# After
[dependencies]
rscrypto = { version = "0.3.1", features = ["xxh3"] }
```

## Algorithm map

| `xxhash-rust` function | rscrypto type | Output |
|---|---|---|
| `xxh3::xxh3_64` | `Xxh3` (alias for `Xxh3_64`) | `u64` |
| `xxh3::xxh3_64_with_seed` | `Xxh3::hash_with_seed` | `u64` |
| `xxh3::xxh3_128` | `Xxh3_128` | `u128` |
| `xxh3::xxh3_128_with_seed` | `Xxh3_128::hash_with_seed` | `u128` |

## API patterns

### One-shot 64-bit

```rust
// Before
use xxhash_rust::xxh3::xxh3_64;
let h: u64 = xxh3_64(b"123456789");
```

```rust
// After
use rscrypto::{FastHash, Xxh3};
let h: u64 = Xxh3::hash(b"123456789");
```

The `FastHash` trait must be in scope to call `::hash`.

### One-shot 64-bit, seeded

```rust
// Before
use xxhash_rust::xxh3::xxh3_64_with_seed;
let h = xxh3_64_with_seed(b"123456789", 0xDEADBEEF);   // (data, seed)
```

```rust
// After
use rscrypto::{FastHash, Xxh3};
let h = Xxh3::hash_with_seed(0xDEADBEEF, b"123456789"); // (seed, data)
```

**Argument order is swapped.** `xxhash-rust` puts `data` first, `seed` second; rscrypto puts `seed` first, `data` second (matches the `FastHash` trait signature across all rscrypto fast hashes).

### One-shot 128-bit

```rust
// Before
use xxhash_rust::xxh3::xxh3_128;
let h: u128 = xxh3_128(b"123456789");
```

```rust
// After
use rscrypto::{FastHash, Xxh3_128};
let h: u128 = Xxh3_128::hash(b"123456789");
```

### Streaming (via `core::hash::Hasher`)

```rust
// Before
use xxhash_rust::xxh3::Xxh3;
let mut hasher = Xxh3::new();
hasher.update(b"foo");
hasher.update(b"bar");
let h: u64 = hasher.digest();
```

```rust
// After
use rscrypto::Xxh3Hasher;
use core::hash::Hasher;
let mut hasher = Xxh3Hasher::default();
hasher.write(b"foo");
hasher.write(b"bar");
let h: u64 = hasher.finish();
```

Renames at the streaming layer:

| `xxhash-rust` | rscrypto | Notes |
|---|---|---|
| `Xxh3::new()` | `Xxh3Hasher::default()` | both unseeded |
| `Xxh3::with_seed(seed)` | `Xxh3Hasher::with_seed(seed)` | seeded ctor |
| `.update(&[u8])` | `.write(&[u8])` | matches `core::hash::Hasher` |
| `.digest() -> u64` | `.finish() -> u64` | matches `core::hash::Hasher` |

`Xxh3Hasher` implements `core::hash::Hasher` directly. Drop the `xxhash-rust` crate from your `Hasher` / `BuildHasher` bounds and use `Xxh3Hasher` / `Xxh3BuildHasher` (e.g., `HashMap<K, V, Xxh3BuildHasher>`).

## Notes

- **Streaming = `Hasher` trait.** `xxhash-rust`'s streaming type rolls its own `update` / `digest` method names; rscrypto routes streaming through the standard `core::hash::Hasher` so it slots into `BuildHasher`-based collections without an adapter.
- **Argument order swap** for `hash_with_seed`. Re-read every call site that previously used `xxh3_64_with_seed`.
- **Streaming requires `alloc`.** `Xxh3Hasher` and `Xxh3BuildHasher` are gated on `alloc`. The one-shot `Xxh3::hash` / `Xxh3::hash_with_seed` is fully `no_std`.
- **128-bit streaming.** rscrypto exposes `Xxh3_128` for one-shot 128-bit; for 128-bit streaming you currently need to feed your data into the 128-bit one-shot rather than incrementally — file an issue if 128-bit streaming is a hard requirement.
- **`xxh32` / `xxh64` legacy.** rscrypto ships only XXH3 (the modern variant). If you depend on the legacy XXH32 or XXH64 algorithms, keep `xxhash-rust` for those.
- **No SIMD-acceleration trade-off.** Both crates ship SIMD backends; rscrypto runtime-dispatches with the same three-tier model used elsewhere (`std` enables runtime detection; the portable kernel is always present).
