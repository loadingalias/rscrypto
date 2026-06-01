# Migration: `crc32fast` → `rscrypto`

> Drop-in replacement for `crc32fast::Hasher`. Same algorithm (CRC-32/IEEE), wider hardware coverage, identical method names — `update` / `finalize` / `reset` carry over verbatim.

Verified against `crc32fast = "1.5.0"` and the `rscrypto` 0.1 line.

## TL;DR

| | Before (`crc32fast` 1.x) | After (`rscrypto` 0.1) |
|---|---|---|
| Cargo dep | `crc32fast = "1.5"` | `rscrypto = { version = "0.1", features = ["crc32"] }` |
| Import | `use crc32fast::Hasher;` | `use rscrypto::checksum::{Checksum, Crc32};` |
| Call | `crc32fast::hash(data)` | `Crc32::checksum(data)` |

## Cargo.toml

```toml
# Before
[dependencies]
crc32fast = "1.5"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.1", features = ["crc32"] }
```

`features = ["crc32"]` enables both CRC-32/IEEE (`Crc32`) and CRC-32C/Castagnoli (`Crc32C`) at no extra build cost.

## API patterns

### One-shot

```rust
// Before
let value = crc32fast::hash(b"123456789");
```

```rust
// After
use rscrypto::checksum::{Checksum, Crc32};
let value = Crc32::checksum(b"123456789");
```

The `Checksum` trait must be in scope to call `::checksum`.

### Streaming

```rust
// Before
use crc32fast::Hasher;
let mut hasher = Hasher::new();
hasher.update(b"foo");
hasher.update(b"bar");
let value = hasher.finalize();        // consumes self
```

```rust
// After
use rscrypto::checksum::{Checksum, Crc32};
let mut hasher = Crc32::new();
hasher.update(b"foo");
hasher.update(b"bar");
let value = hasher.finalize();        // borrows &self
```

### Reset

```rust
// Before
let mut hasher = crc32fast::Hasher::new();
hasher.update(b"throwaway");
hasher.reset();
hasher.update(b"123456789");
let value = hasher.finalize();
```

```rust
// After
use rscrypto::checksum::{Checksum, Crc32};
let mut hasher = Crc32::new();
hasher.update(b"throwaway");
hasher.reset();
hasher.update(b"123456789");
let value = hasher.finalize();
```

### Combine (parallel chunks)

```rust
// Before
use crc32fast::Hasher;
let mut a = Hasher::new();
a.update(left);
let mut b = Hasher::new();
b.update(right);
a.combine(&b);
let value = a.finalize();
```

```rust
// After
use rscrypto::checksum::{Checksum, ChecksumCombine, Crc32};
let crc_a = Crc32::checksum(left);
let crc_b = Crc32::checksum(right);
let value = Crc32::combine(crc_a, crc_b, right.len());
```

rscrypto's combine works on the finalized `u32` outputs and the right-hand length, not on hasher state. Cheaper to pass around (just two `u32`s) and trivial to checkpoint to disk between phases.

## Notes

- **Single algorithm.** `crc32fast` is CRC-32/IEEE only. If you need Castagnoli, use rscrypto's `Crc32C` (same feature flag, no extra dependency).
- **`finalize` consumes vs. borrows.** `crc32fast::Hasher::finalize(self)` consumes the hasher. rscrypto's `Crc32::finalize(&self)` borrows, so you can read the running CRC at any point and continue feeding bytes.
- **`Hasher::new_with_initial(u32)`.** rscrypto's equivalent is `Crc32::with_initial(u32)` (resume from a raw initial state) and `Crc32::resume(prev_finalized)` (resume from a previously finalized CRC, which inverts the XOR-out). The former matches `crc32fast`'s semantics directly.
- **`no_std`.** `crc32fast` requires `std` for runtime SIMD detection. rscrypto detects at runtime when `std` is enabled and falls back to compile-time `target_feature` selection in `no_std` builds. The `portable-only` feature pins to the portable kernel for FIPS / DO-178C / IEC 62443 lanes.
- **`std::hash::Hasher`.** `crc32fast::Hasher` does not implement `core::hash::Hasher`. Neither does rscrypto's `Crc32`. If you need a `core::hash::Hasher` adapter, wrap manually.
