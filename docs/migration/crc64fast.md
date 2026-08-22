# Migration: `crc64fast` → `rscrypto`

> Same CRC-64/XZ output with method renames from `Digest` / `write` / `sum64` to `Crc64` / `update` / `finalize`. The aside below also covers `crc64fast-nvme` → `Crc64Nvme`.

Verified against `crc64fast = "1.1.0"` for `Crc64`; `Crc64Nvme` oracle coverage uses `crc-fast = "1.10.0"` and the `rscrypto` 0.8.1 line.
Evidence: `tests/crc64_properties.rs` compares one-shot, streaming, and combine output against those oracle crates.

## TL;DR

|           | Before (`crc64fast` 1.x)                | After (`rscrypto` 0.8.1)                                 |
| --------- | --------------------------------------- | -------------------------------------------------------- |
| Cargo dep | `crc64fast = "1.1"`                     | `rscrypto = { version = "0.8.1", features = ["crc64"] }` |
| Import    | `use crc64fast::Digest;`                | `use rscrypto::checksum::{Checksum, Crc64};`             |
| Call      | `Digest::new(); .write(data); .sum64()` | `Crc64::new(); .update(data); .finalize()`               |

## Cargo.toml

```toml
# Before
[dependencies]
crc64fast = "1.1"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.8.1", features = ["crc64"] }
```

`features = ["crc64"]` enables both `Crc64` (XZ / ECMA-182) and `Crc64Nvme`.

## Algorithm map

| Upstream crate           | Polynomial label            | rscrypto type             |
| ------------------------ | --------------------------- | ------------------------- |
| `crc64fast::Digest`      | "ECMA" (== ECMA-182, == XZ) | `Crc64` (alias `Crc64Xz`) |
| `crc64fast_nvme::Digest` | NVMe (Rocksoft)             | `Crc64Nvme`               |

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

| `crc64fast`       | rscrypto             |
| ----------------- | -------------------- |
| `Digest::new()`   | `Crc64::new()`       |
| `.write(&[u8])`   | `.update(&[u8])`     |
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

- **CRC labels are insufficient.** `crc64fast::Digest` is verified
  byte-for-byte against rscrypto's `Crc64` / `Crc64Xz`. A polynomial alone
  does not identify a CRC; initialization, reflection, and final XOR parameters
  also matter. Do not substitute CRC-64/ISO.
- **No reset, no resume in `crc64fast`.** Build a fresh `Digest` per checksum. rscrypto adds `.reset()` and `Crc64::resume(prev)` on top of the same shape.
- **`no_std`.** `crc64fast` requires `std` for SIMD detection. rscrypto's `Crc64` is `no_std`-capable; runtime detection is gated on the `std` feature, with compile-time `target_feature` selection in `no_std` builds and a portable fallback always present.
- **Hardware coverage.** `crc64fast` ships x86_64 (PCLMUL) and aarch64 (PMULL) backends. rscrypto adds VPCLMULQDQ (large buffers on x86_64), SVE2 PMULL (aarch64), VPMSUMD (Power), and VGFM (s390x). RISC-V uses the portable slice-by-16 implementation.
- **Force a backend.** `RSCRYPTO_CRC64_FORCE=portable` selects the portable
  CRC-64 runtime backend in `std` builds. The `portable-only` feature makes
  runtime capability detection ignore host acceleration; see
  [`docs/features.md`](../features.md#portable-only) for its limits.
