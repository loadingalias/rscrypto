# rscrypto

Pure Rust cryptography. Zero dependencies. Hardware accelerated.

**Supported:** x86-64, ARM64, Apple Silicon, RISC-V, IBM POWER, s390x.

> **Note:** Only checksums are production-ready. Hash/AEAD/PQC crates are in development.

## Install

```toml
[dependencies]
rscrypto = "0.1"
```

## Usage

```rust
use rscrypto::{Crc32C, Checksum};

let crc = Crc32C::checksum(b"hello world");
```

## Performance (GiB/s)

| Algorithm | Zen 4 | Apple M3 | Graviton 2 |
|-----------|-------|----------|------------|
| CRC-64/XZ | 72 | 63 | 33 |
| CRC-64/NVME | 75 | 62 | 33 |
| CRC-32C | 72 | 75 | 40 |
| CRC-32 | 78 | 74 | 40 |
| CRC-16 | 80 | 61 | 33 |

Automatic dispatch: AVX-512, VPCLMUL, PMULL, EOR3, hardware CRC.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `checksums` | yes | CRC-16, CRC-24, CRC-32, CRC-64 |
| `hashes` | no | SHA-2, SHA-3, BLAKE3, and fast hash families |
| `std` | yes | Runtime CPU detection |
| `alloc` | yes | Buffered streaming APIs |

For `no_std`:

```toml
rscrypto = { version = "0.1", default-features = false, features = ["checksums"] }
```

## Contributing

Run `just tune` for active Blake3 boundary capture.
See `docs/tuning.md` for the tuning workflow and artifact format.
For perf-gap work on Blake3, use `just bench-blake3-core`.
Runner/architecture policy for CI/bench/tune is pinned in `ARCHITECTURE_MATRIX.md`.

## License

MIT OR Apache-2.0
