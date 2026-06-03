# Migration: `twox-hash` → `rscrypto`

> Replace `twox_hash::XxHash3_64::oneshot` with `rscrypto::Xxh3::hash`. Streaming flows through `core::hash::Hasher` in both crates, so the only changes are the type name and the import path.

Verified against `twox-hash = "2.1.2"` and the `rscrypto` 0.3.1 line.

## TL;DR

| | Before (`twox-hash` 2.x) | After (`rscrypto` 0.3.1) |
|---|---|---|
| Cargo dep | `twox-hash = "2.1"` | `rscrypto = { version = "0.3.1", features = ["xxh3"] }` |
| Import | `use twox_hash::XxHash3_64;` | `use rscrypto::{FastHash, Xxh3};` |
| Call | `XxHash3_64::oneshot(data)` | `Xxh3::hash(data)` |

## Cargo.toml

```toml
# Before
[dependencies]
twox-hash = "2.1"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.3.1", features = ["xxh3"] }
```

## Algorithm map

| `twox-hash` type | rscrypto type | Output |
|---|---|---|
| `XxHash3_64` | `Xxh3` (alias for `Xxh3_64`) | `u64` |
| `XxHash3_128` | `Xxh3_128` | `u128` |
| `XxHash64` (legacy XXH64) | not mapped — keep `twox-hash` | `u64` |
| `XxHash32` (legacy XXH32) | not mapped — keep `twox-hash` | `u32` |

## API patterns

### One-shot 64-bit

```rust
// Before
use twox_hash::XxHash3_64;
let h: u64 = XxHash3_64::oneshot(b"123456789");
```

```rust
// After
use rscrypto::{FastHash, Xxh3};
let h: u64 = Xxh3::hash(b"123456789");
```

### One-shot 64-bit, seeded

```rust
// Before
use twox_hash::XxHash3_64;
let h = XxHash3_64::oneshot_with_seed(0xDEADBEEF, b"123456789");
```

```rust
// After
use rscrypto::{FastHash, Xxh3};
let h = Xxh3::hash_with_seed(0xDEADBEEF, b"123456789");
```

Argument order matches — `(seed, data)` in both crates.

### One-shot 128-bit

```rust
// Before
use twox_hash::XxHash3_128;
let h: u128 = XxHash3_128::oneshot(b"123456789");
```

```rust
// After
use rscrypto::{FastHash, Xxh3_128};
let h: u128 = Xxh3_128::hash(b"123456789");
```

### Streaming (via `core::hash::Hasher`)

```rust
// Before
use twox_hash::XxHash3_64;
use core::hash::Hasher;
let mut hasher = XxHash3_64::with_seed(0);
hasher.write(b"foo");
hasher.write(b"bar");
let h = hasher.finish();
```

```rust
// After
use rscrypto::Xxh3Hasher;
use core::hash::Hasher;
let mut hasher = Xxh3Hasher::default();
hasher.write(b"foo");
hasher.write(b"bar");
let h = hasher.finish();
```

Both implement `core::hash::Hasher` — same `write` / `finish` shape. Rename the type, keep the trait imports.

## Notes

- **Drop-in for `BuildHasher`.** `twox-hash` ships `xxhash3_64::RandomState`, `xxhash3_64::FixedState`, etc. for `HashMap` / `HashSet` use. rscrypto's `Xxh3BuildHasher` is the equivalent (`HashMap<K, V, Xxh3BuildHasher>`).
- **Legacy XXH32 / XXH64.** rscrypto does not ship the legacy variants. Keep `twox-hash` as a sibling dependency for those, or open a feature request.
- **Streaming requires `alloc`.** `Xxh3Hasher` and `Xxh3BuildHasher` are gated on `alloc`. The one-shot `Xxh3::hash` / `Xxh3::hash_with_seed` is fully `no_std`.
- **`twox-hash` 1.x → 2.x.** If you are still on `twox-hash` 1.x, the API was substantially reworked in 2.x. Migrate to 2.x first or jump straight to rscrypto using the 1.x shape (`Xxh3Hasher` for streaming, `Xxh3::hash` for one-shot) — the patterns above are the destination either way.
