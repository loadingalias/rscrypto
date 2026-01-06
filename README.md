# rscrypto

Pure Rust cryptography with hardware acceleration.

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

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `checksums` | yes | CRC16, CRC24, CRC32, CRC64 |
| `std` | yes | Runtime CPU detection |
| `alloc` | yes | Buffered streaming APIs |

For `no_std`:

```toml
rscrypto = { version = "0.1", default-features = false, features = ["checksums"] }
```

## Performance

Automatic dispatch to fastest available:

- x86_64: AVX-512, AVX2, PCLMULQDQ, SSE4.2
- aarch64: PMULL, CRC, NEON
- Portable fallback on all targets

## Tuning Coverage

rscrypto uses microarchitecture-specific tuning for optimal performance. The tuning
data comes from benchmark measurements on real hardware.

### Measured (benchmark data available)

| Platform | Status |
|----------|--------|
| AMD Zen 4/5/5c | ✓ Measured |
| Apple M1-M4 | ✓ Measured |
| AWS Graviton 2/3/4 | ✓ Measured |
| Intel SPR/GNR/ICL | ✓ Family inference from Zen4 |

### Family Inference (extrapolated from similar architectures)

| Platform | Inferred From |
|----------|---------------|
| Apple M5 | Apple M4 |
| AWS Graviton 5 | Graviton 4 |
| ARM Neoverse N2/N3/V3 | Graviton family |
| NVIDIA Grace | Neoverse V2 |
| Ampere Altra | Graviton 2 |

### Contributions Wanted

We especially need benchmark data from:

| Priority | Platform |
|----------|----------|
| High | Intel Sapphire/Granite/Ice Lake (direct measurements) |
| Medium | IBM Power9/10 |
| Medium | IBM z14/z15 |
| Low | RISC-V with vector extensions |

Run `just tune-contribute` and submit the output via GitHub issue.
See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT OR Apache-2.0
