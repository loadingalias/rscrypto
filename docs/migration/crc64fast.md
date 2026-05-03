# Migration: `crc64fast` → `rscrypto`

> Drop-in replacement for `crc64fast::Digest`. Same polynomial (CRC-64/XZ, ECMA-182), broader hardware coverage. The aside below also covers `crc64fast-nvme` → `Crc64Nvme`.

Verified against `crc64fast = "1.1.0"`, `crc64fast-nvme = "1.2.1"`, and the `rscrypto` 0.1 line.

## TL;DR

| | Before (`crc64fast` 1.x) | After (`rscrypto` 0.1) |
|---|---|---|
| Cargo dep | `crc64fast = "1.1"` | `rscrypto = { version = "0.1", features = ["crc64"] }` |
| Import | `use crc64fast::Digest;` | `use rscrypto::checksum::{Checksum, Crc64};` |
| Call | `Digest::new(); .write(data); .sum64()` | `Crc64::new(); .update(data); .finalize()` |

## Cargo.toml

```toml
# Before
[dependencies]
crc64fast = "1.1"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.1", features = ["crc64"] }
```

`features = ["crc64"]` enables both `Crc64` (XZ / ECMA-182) and `Crc64Nvme`.

## Algorithm map

| Upstream crate | Polynomial label | rscrypto type |
|---|---|---|
| `crc64fast::Digest` | "ECMA" (== ECMA-182, == XZ) | `Crc64` (alias `Crc64Xz`) |
| `crc64fast_nvme::Digest` | NVMe (Rocksoft) | `Crc64Nvme` |

`crc64fast` calls its polynomial "ECMA". This is the ECMA-182 polynomial used by XZ Utils, 7-Zip, and `CRC_64_XZ` in `crc`. Map it to `Crc64` / `Crc64Xz`.

## API patterns

### Streaming (the only API)

```rust
// Before
use crc64fast::Digest;
let mut digest = Digest::new();
digest.write(b"hello ");
digest.write(b"world!");
let value = digest.sum64();
```

```rust
// After
use rscrypto::checksum::{Checksum, Crc64};
let mut hasher = Crc64::new();
hasher.update(b"hello ");
hasher.update(b"world!");
let value = hasher.finalize();
```

Three method renames:

| `crc64fast` | rscrypto |
|---|---|
| `Digest::new()` | `Crc64::new()` |
| `.write(&[u8])` | `.update(&[u8])` |
| `.sum64() -> u64` | `.finalize() -> u64` |

### One-shot (rscrypto-only)

`crc64fast` has no one-shot. rscrypto exposes one through the `Checksum` trait:

```rust
// After
use rscrypto::checksum::{Checksum, Crc64};
let value = Crc64::checksum(b"hello world!");
```

### Combine (rscrypto-only)

```rust
// After
use rscrypto::checksum::{Checksum, ChecksumCombine, Crc64};
let crc_a = Crc64::checksum(left);
let crc_b = Crc64::checksum(right);
let value = Crc64::combine(crc_a, crc_b, right.len());
```

## Aside: migrating from `crc64fast-nvme`

Same shape, same renames, different rscrypto type:

```rust
// Before
use crc64fast_nvme::Digest;
let mut digest = Digest::new();
digest.write(b"hello ");
digest.write(b"world!");
let value = digest.sum64();
```

```rust
// After
use rscrypto::checksum::{Checksum, Crc64Nvme};
let mut hasher = Crc64Nvme::new();
hasher.update(b"hello ");
hasher.update(b"world!");
let value = hasher.finalize();
```

Drop both `crc64fast` and `crc64fast-nvme` from Cargo.toml; `features = ["crc64"]` covers both.

## Notes

- **Polynomial label confusion.** "ECMA" in `crc64fast` is the polynomial `0x42F0E1EBA9EA3693`, which is what `crc::CRC_64_XZ` and `crc-fast::CrcAlgorithm::Crc64Xz` call XZ. There is also a distinct polynomial often called "CRC-64/ECMA-182" in some references — same one. There is **no** ISO 3309 ("CRC-64/ISO") variant in either crate; do not confuse the two.
- **No reset, no resume in `crc64fast`.** Build a fresh `Digest` per checksum. rscrypto adds `.reset()` and `Crc64::resume(prev)` on top of the same shape.
- **`no_std`.** `crc64fast` requires `std` for SIMD detection. rscrypto's `Crc64` is `no_std`-capable; runtime detection is gated on the `std` feature, with compile-time `target_feature` selection in `no_std` builds and a portable fallback always present.
- **Hardware coverage.** `crc64fast` ships x86_64 (PCLMUL) and aarch64 (PMULL) backends. rscrypto adds VPCLMULQDQ (large buffers on x86_64), SVE2 PMULL (aarch64), VPMSUMD (Power), VGFM (s390x), and Zbc/Zvbc (RISC-V).
- **Force a backend.** `RSCRYPTO_CRC64_FORCE=portable` (std only) or the `portable-only` feature pins the audited portable kernel for FIPS / DO-178C / IEC 62443 lanes. `crc64fast` has no equivalent.
