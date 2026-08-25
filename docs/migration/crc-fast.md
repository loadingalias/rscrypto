# Migration: `crc-fast` → `rscrypto`

Replace `crc-fast`'s enum-driven `checksum(CrcAlgorithm::*, data)` calls with
rscrypto's named CRC types. The mapped algorithms keep the same CRC parameters,
but rscrypto returns each algorithm's natural integer width instead of `u64`.

Verified against `crc-fast = "1.10.0"` and the `rscrypto` 0.8.1 line.
Evidence: `tests/crc16_properties.rs`, `tests/crc32_properties.rs`, and
`tests/crc64_properties.rs`.

## TL;DR

|           | Before (`crc-fast` 1.x)                             | After (`rscrypto` 0.8.1)                                          |
| --------- | --------------------------------------------------- | ----------------------------------------------------------------- |
| Cargo dep | `crc-fast = "1.10"`                                 | `rscrypto = { version = "0.8.1", features = ["crc32", "crc64"] }` |
| Import    | `use crc_fast::{checksum, CrcAlgorithm};`           | `use rscrypto::checksum::{Checksum, Crc32};`                      |
| Call      | `checksum(CrcAlgorithm::Crc32IsoHdlc, data) as u32` | `Crc32::checksum(data)`                                           |

## Cargo.toml

```toml
# Before
[dependencies]
crc-fast = "1.10"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.8.1", features = ["crc32", "crc64"] }
```

Add `crc16` or `crc24` only when you migrate a mapped variant from those
families. Use `features = ["checksums"]` to enable every CRC family.

## Algorithm map

`crc-fast` exposes the full RevEng catalogue via `CrcAlgorithm`. The intersection with rscrypto:

| `CrcAlgorithm` variant      | rscrypto type            | Feature flag |
| --------------------------- | ------------------------ | ------------ |
| `Crc32IsoHdlc` (IEEE 802.3) | `Crc32`                  | `crc32`      |
| `Crc32Iscsi` (Castagnoli)   | `Crc32C`                 | `crc32`      |
| `Crc64Xz` (ECMA-182)        | `Crc64`                  | `crc64`      |
| `Crc64Nvme`                 | `Crc64Nvme`              | `crc64`      |
| (CRC-16 variants)           | `Crc16Ccitt`, `Crc16Ibm` | `crc16`      |
| (CRC-24 variants)           | `Crc24OpenPgp`           | `crc24`      |

For variants outside this list, keep `crc-fast` as a sibling dependency or open a feature request.

## API patterns

### One-shot

```rust
// Before
use crc_fast::{checksum, CrcAlgorithm};
let value = checksum(CrcAlgorithm::Crc32IsoHdlc, b"123456789");
// `value: u64`: even for CRC-32. Cast to u32 at boundaries.
```

```rust
// After
use rscrypto::checksum::{Checksum, Crc32};
let value: u32 = Crc32::checksum(b"123456789");
```

The one-shot also exists as a typed shortcut in `crc-fast` (`crc32_iso_hdlc(data) -> u32`); both forms collapse to the same rscrypto call.

### Streaming

```rust
// Before
use crc_fast::{Digest, CrcAlgorithm};
let mut digest = Digest::new(CrcAlgorithm::Crc32IsoHdlc);
digest.update(b"foo");
digest.update(b"bar");
let value = digest.finalize();        // consumes self, returns u64
```

```rust
// After
use rscrypto::checksum::{Checksum, Crc32};
let mut hasher = Crc32::new();
hasher.update(b"foo");
hasher.update(b"bar");
let value = hasher.finalize();        // borrows &self, returns u32
```

`Digest::finalize` consumes `self` in `crc-fast`; `Checksum::finalize` borrows
in rscrypto. You can read an intermediate finalized value and continue updating
the same rscrypto hasher.

### Combine (parallel chunks)

```rust
// Before
use crc_fast::{checksum, checksum_combine, CrcAlgorithm};
let crc_a = checksum(CrcAlgorithm::Crc32IsoHdlc, a);
let crc_b = checksum(CrcAlgorithm::Crc32IsoHdlc, b);
let combined = checksum_combine(
    CrcAlgorithm::Crc32IsoHdlc,
    crc_a,
    crc_b,
    b.len() as u64,                   // u64, not usize
) as u32;
```

```rust
// After
use rscrypto::checksum::{Checksum, ChecksumCombine, Crc32};
let crc_a = Crc32::checksum(a);
let crc_b = Crc32::checksum(b);
let combined = Crc32::combine(crc_a, crc_b, b.len());   // usize
```

### File checksums

`crc-fast` ships `checksum_file(algorithm, path, buf_size)`. rscrypto delegates to its IO adapter:

```rust
// After (std feature)
use rscrypto::checksum::{Checksum, Crc32};
use std::{fs::File, io};

let mut reader = Crc32::reader(File::open(path)?);
io::copy(&mut reader, &mut io::sink())?;
let value = reader.checksum();
```

## Notes

- **Return-type widening.** Every generic `crc-fast::checksum` returns `u64`. rscrypto preserves the algorithm's natural width (`u16` / `u32` / `u64`). Apply `as u32` at the boundary or drop it once the call site matches the new type.
- **Combine length parameter.** `crc-fast::checksum_combine` takes `u64`; `rscrypto::Crc32::combine` takes `usize`. Drop the cast.
- **Custom CRCs (`CrcParams`).** `crc-fast` accepts user-supplied polynomials; rscrypto's named types are a fixed set. Stay on `crc-fast` if you compute non-catalogue CRCs.
- **`digest::DynDigest`.** rscrypto does not expose a `digest` crate
  implementation. Keep `crc-fast` for call sites that require
  `digest::DynDigest`.
- **`std::io::Write`.** `crc-fast::Digest: Write`. rscrypto exposes the same shape via `Crc32::writer(sink)` returning a `ChecksumWriter` that wraps an inner writer; the rscrypto hasher itself does not implement `Write` directly.
- **Force a backend.** `RSCRYPTO_CRC32_FORCE=portable` selects the portable
  CRC-32 runtime backend in `std` builds. The `portable-only` feature makes
  runtime capability detection ignore host acceleration; see
  [`docs/features.md`](../features.md#portable-only) for its limits.
