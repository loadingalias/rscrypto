# Migration: `crc` → `rscrypto`

> Replace the parameterized `Crc::<W>::new(&CRC_*)` builder with rscrypto's named CRC types. Same outputs, hardware-accelerated dispatch, no generic width to thread through call sites.

Verified against `crc = "3.4.0"` and the `rscrypto` 0.1 line.

## TL;DR

| | Before (`crc` 3.x) | After (`rscrypto` 0.1) |
|---|---|---|
| Cargo dep | `crc = "3.4"` | `rscrypto = { version = "0.1", features = ["checksums"] }` |
| Import | `use crc::{Crc, CRC_32_ISO_HDLC};` | `use rscrypto::checksum::{Checksum, Crc32};` |
| Call | `Crc::<u32>::new(&CRC_32_ISO_HDLC).checksum(data)` | `Crc32::checksum(data)` |

## Cargo.toml

```toml
# Before
[dependencies]
crc = "3.4"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.1", features = ["checksums"] }
```

`features = ["checksums"]` enables every CRC family. To trim the build, pick only what you use: `crc16`, `crc24`, `crc32` (covers IEEE + Castagnoli), `crc64` (covers XZ + NVME).

## Algorithm map

| `crc` constant | rscrypto type | Feature flag |
|---|---|---|
| `CRC_16_IBM_SDLC` (CRC-16/X25) | `Crc16Ccitt` | `crc16` |
| `CRC_16_ARC` (CRC-16/IBM) | `Crc16Ibm` | `crc16` |
| `CRC_24_OPENPGP` | `Crc24OpenPgp` | `crc24` |
| `CRC_32_ISO_HDLC` (IEEE 802.3) | `Crc32` (alias `Crc32Ieee`) | `crc32` |
| `CRC_32_ISCSI` (Castagnoli) | `Crc32C` (alias `Crc32Castagnoli`) | `crc32` |
| `CRC_64_XZ` (ECMA-182) | `Crc64` (alias `Crc64Xz`) | `crc64` |
| `CRC_64_NVME` | `Crc64Nvme` | `crc64` |

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

- **No SIMD in `crc`.** rscrypto dispatches to PCLMULQDQ / VPCLMULQDQ on x86_64, PMULL / SVE2 on aarch64, VPMSUMD on Power, VGFM on s390x, Zbc/Zvbc on RISC-V, with a portable fallback. Force the portable kernel via `RSCRYPTO_CRC32_FORCE=portable` (std only) or the `portable-only` feature for FIPS / DO-178C / IEC 62443 lanes.
- **Long tail not yet covered.** rscrypto does not (yet) ship every constant in the `crc` catalogue (`CRC_8_*`, `CRC_82_DARC`, alternate CRC-32 polynomials, etc.). If you depend on one of those, keep `crc` as a dependency for that variant or open a feature request.
- **`no_std`.** Both crates are `no_std`-capable. rscrypto's `Buffered*` wrappers require `alloc`; `ChecksumReader` / `ChecksumWriter` require `std`.
- **Output widths match.** CRC-16 → `u16`, CRC-32 → `u32`, CRC-64 → `u64`. CRC-24 returns `u32` masked to 24 bits (`0x00FFFFFF`) — same as `crc::Crc<u32>::new(&CRC_24_OPENPGP)`.
