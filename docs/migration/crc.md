# Migration: `crc` → `rscrypto`

Replace supported `Crc::<W>::new(&CRC_*)` configurations with rscrypto's named
CRC types. The mapped algorithms retain their output parameters and have a
portable fallback plus selected target-specific backends.

Verified against `crc = "3.4.0"` and the `rscrypto` 0.8.1 line.
Evidence: `tests/crc16_properties.rs`, `tests/crc24_properties.rs`, `tests/crc32_properties.rs`, and `tests/crc64_properties.rs`.

## TL;DR

|           | Before (`crc` 3.x)                                 | After (`rscrypto` 0.8.1)                                     |
| --------- | -------------------------------------------------- | ------------------------------------------------------------ |
| Cargo dep | `crc = "3.4"`                                      | `rscrypto = { version = "0.8.1", features = ["checksums"] }` |
| Import    | `use crc::{Crc, CRC_32_ISO_HDLC};`                 | `use rscrypto::checksum::{Checksum, Crc32};`                 |
| Call      | `Crc::<u32>::new(&CRC_32_ISO_HDLC).checksum(data)` | `Crc32::checksum(data)`                                      |

## Cargo.toml

```toml
# Before
[dependencies]
crc = "3.4"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.8.1", features = ["checksums"] }
```

`features = ["checksums"]` enables every CRC family. To trim the build, pick only what you use: `crc16`, `crc24`, `crc32` (covers IEEE + Castagnoli), `crc64` (covers XZ + NVME).

## Algorithm map

| `crc` constant                 | rscrypto type                      | Feature flag |
| ------------------------------ | ---------------------------------- | ------------ |
| `CRC_16_IBM_SDLC` (CRC-16/X25) | `Crc16Ccitt`                       | `crc16`      |
| `CRC_16_ARC` (CRC-16/IBM)      | `Crc16Ibm`                         | `crc16`      |
| `CRC_24_OPENPGP`               | `Crc24OpenPgp`                     | `crc24`      |
| `CRC_32_ISO_HDLC` (IEEE 802.3) | `Crc32` (alias `Crc32Ieee`)        | `crc32`      |
| `CRC_32_ISCSI` (Castagnoli)    | `Crc32C` (alias `Crc32Castagnoli`) | `crc32`      |
| `CRC_64_XZ` (ECMA-182)         | `Crc64` (alias `Crc64Xz`)          | `crc64`      |
| `CRC_64_NVME`                  | `Crc64Nvme`                        | `crc64`      |

## API patterns

### One-shot

```rust
// Before
use crc::{Crc, CRC_32_ISO_HDLC};
const CRC32: Crc<u32> = Crc::<u32>::new(&CRC_32_ISO_HDLC);
let value = CRC32.checksum(b"123456789");
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
use crc::{Crc, CRC_64_XZ};
const CRC64: Crc<u64> = Crc::<u64>::new(&CRC_64_XZ);
let mut digest = CRC64.digest();
digest.update(b"foo");
digest.update(b"bar");
let value = digest.finalize();
```

```rust
// After
use rscrypto::checksum::{Checksum, Crc64};
let mut hasher = Crc64::new();
hasher.update(b"foo");
hasher.update(b"bar");
let value = hasher.finalize();
```

`finalize` borrows `&self` in rscrypto (it consumes `Digest` in `crc`). You can finalize repeatedly without rebuilding the hasher; call `.reset()` between independent inputs to reuse the allocation.

### Combine (parallel chunks)

`crc` exposes no public `combine`. rscrypto provides one directly:

```rust
// After
use rscrypto::checksum::{Checksum, ChecksumCombine, Crc32};

let (a, b) = b"hello world".split_at(5);
let crc_a = Crc32::checksum(a);
let crc_b = Crc32::checksum(b);
let combined = Crc32::combine(crc_a, crc_b, b.len());
assert_eq!(combined, Crc32::checksum(b"hello world"));
```

### Resume from a stored CRC

`crc` has no resume API. rscrypto seeds a hasher from a previously stored value:

```rust
// After
use rscrypto::checksum::{Checksum, Crc32};

let crc_a = Crc32::checksum(first_chunk);
let mut hasher = Crc32::resume(crc_a);
hasher.update(second_chunk);
let final_crc = hasher.finalize();
```

## Notes

- **Backend coverage.** rscrypto has selected x86_64, AArch64, Power,
  s390x, and RISC-V backends plus a portable fallback. See
  [`docs/platforms.md`](../platforms.md) for the maintained target matrix.
  `RSCRYPTO_CRC32_FORCE=portable` affects the CRC-32 runtime dispatcher in
  `std` builds; `portable-only` makes runtime capability detection ignore host
  acceleration.
- **Unsupported catalogue entries.** rscrypto does not expose every `crc`
  constant, including `CRC_8_*`, `CRC_82_DARC`, and alternate CRC-32
  polynomials. Keep `crc` for any unmapped variant.
- **`no_std`.** Both crates are `no_std`-capable. rscrypto's `Buffered*` wrappers require `alloc`; `ChecksumReader` / `ChecksumWriter` require `std`.
- **Output widths match.** CRC-16 → `u16`, CRC-32 → `u32`, CRC-64 → `u64`. CRC-24 returns `u32` masked to 24 bits (`0x00FFFFFF`): same as `crc::Crc<u32>::new(&CRC_24_OPENPGP)`.
