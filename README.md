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

## License

MIT OR Apache-2.0
