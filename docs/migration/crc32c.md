# Migration: `crc32c` → `rscrypto`

> Replace the three free functions `crc32c::crc32c`, `crc32c_append`, `crc32c_combine` with `rscrypto::Crc32C`'s trait-method equivalents. Same Castagnoli polynomial, byte-identical output, hardware-accelerated dispatch on every supported target.

Output is covered by the CRC-32C oracle/property tests in `tests/crc32_properties.rs`.

## TL;DR

| | Before (`crc32c` 0.6.x) | After (`rscrypto` 0.5.0) |
|---|---|---|
| Cargo dep | `crc32c = "0.6"` | `rscrypto = { version = "0.5.0", features = ["crc32"] }` |
| Import | `use crc32c::{crc32c, crc32c_append, crc32c_combine};` | `use rscrypto::checksum::{Checksum, ChecksumCombine, Crc32C};` |
| Call | `crc32c(data)` | `Crc32C::checksum(data)` |

## Cargo.toml

```toml
# Before
[dependencies]
crc32c = "0.6"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.5.0", features = ["crc32"] }
```

The `crc32` feature enables both `Crc32` (IEEE) and `Crc32C` (Castagnoli) at no extra binary cost.

## API patterns

### One-shot

```rust
// Before
let crc: u32 = crc32c::crc32c(b"123456789");
```

```rust
// After
use rscrypto::checksum::{Checksum, Crc32C};
let crc: u32 = Crc32C::checksum(b"123456789");
```

### Resume from a previously computed CRC

```rust
// Before
let crc_a = crc32c::crc32c(first_chunk);
let crc_full = crc32c::crc32c_append(crc_a, second_chunk);
```

```rust
// After
use rscrypto::checksum::{Checksum, Crc32C};
let crc_a = Crc32C::checksum(first_chunk);
let mut h = Crc32C::resume(crc_a);
h.update(second_chunk);
let crc_full = h.finalize();
```

### Combine (parallel chunks)

```rust
// Before
let crc_a = crc32c::crc32c(left);
let crc_b = crc32c::crc32c(right);
let crc_full = crc32c::crc32c_combine(crc_a, crc_b, right.len());
```

```rust
// After
use rscrypto::checksum::{Checksum, ChecksumCombine, Crc32C};
let crc_a = Crc32C::checksum(left);
let crc_b = Crc32C::checksum(right);
let crc_full = Crc32C::combine(crc_a, crc_b, right.len());
```

### Streaming (rscrypto-only)

`crc32c` has no streaming type — every call is a function. rscrypto adds it through the `Checksum` trait:

```rust
// After
use rscrypto::checksum::{Checksum, Crc32C};
let mut h = Crc32C::new();
h.update(b"foo");
h.update(b"bar");
let crc = h.finalize();
```

## Notes

- **Single algorithm.** `crc32c` is CRC-32C (Castagnoli) only. If you also need CRC-32 IEEE, both crates would normally require separate dependencies — rscrypto's `crc32` feature covers both with one dep.
- **Hardware acceleration parity.** Both crates dispatch to the SSE4.2 `crc32` instruction on x86_64 and the ARMv8 CRC extension on aarch64. rscrypto adds VPCLMULQDQ folding (large buffers on x86_64), SVE2-PMULL (aarch64), VPMSUMD (Power), VGFM (s390x), and Zbc/Zvbc (RISC-V).
- **Force a backend.** rscrypto honors `RSCRYPTO_CRC32C_FORCE=portable` (std only) and the crate's `portable-only` feature for FIPS / DO-178C / IEC 62443 lanes. `crc32c` has no equivalent.
- **`no_std`.** Both crates support `no_std`. rscrypto's `Buffered*` wrappers and IO adapters require `alloc` / `std`; the core `Checksum` trait API is pure `no_std`.
- **`Checksum` / `ChecksumCombine` trait imports.** Required at the call site so the trait methods (`::checksum`, `::combine`, etc.) are in scope. RustCrypto-style users will recognise the pattern.
