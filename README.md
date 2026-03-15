# rscrypto

[![Crates.io](https://img.shields.io/crates/v/rscrypto.svg)](https://crates.io/crates/rscrypto)
[![docs.rs](https://img.shields.io/docsrs/rscrypto)](https://docs.rs/rscrypto)
[![CI](https://github.com/loadingalias/rscrypto/actions/workflows/commit.yaml/badge.svg)](https://github.com/loadingalias/rscrypto/actions/workflows/commit.yaml)
[![MSRV](https://img.shields.io/badge/MSRV-1.94.0-blue.svg)](https://blog.rust-lang.org/)
[![License](https://img.shields.io/crates/l/rscrypto.svg)](https://github.com/loadingalias/rscrypto)

Pure Rust cryptography. Zero dependencies. Hardware accelerated.

**Platforms:** x86-64, ARM64, Apple Silicon, RISC-V, IBM POWER, s390x.

## Install

```toml
[dependencies]
rscrypto = "0.1"
```

Zero runtime dependencies by default. Only `rayon` is pulled in with the
optional `parallel` feature.

## Usage

### Checksums

```rust
use rscrypto::{Checksum, Crc32C};

// One-shot
let crc = Crc32C::checksum(b"hello world");

// Streaming
let mut hasher = Crc32C::new();
hasher.update(b"hello ");
hasher.update(b"world");
assert_eq!(hasher.finalize(), crc);
```

### Hashes

```rust
use rscrypto::{Digest, Sha256, Blake3};

let sha = Sha256::digest(b"hello world");
assert_eq!(sha.len(), 32);

let b3 = Blake3::digest(b"hello world");
assert_eq!(b3.len(), 32);
```

### `no_std`

```toml
rscrypto = { version = "0.1", default-features = false, features = ["checksums"] }
```

Without `std`, hardware acceleration uses compile-time feature detection only.

## Performance (GiB/s)

### Checksums

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
| `std` | yes | Runtime CPU detection, implies `alloc` |
| `alloc` | yes | Buffered streaming APIs |
| `checksums` | yes | CRC-16, CRC-24, CRC-32, CRC-64 |
| `hashes` | yes | SHA-2, SHA-3, BLAKE3, Ascon, XXH3, RapidHash |
| `parallel` | no | Rayon-based parallel hashing (Blake3) |

## Algorithms

### Checksums

CRC-16 (CCITT, IBM), CRC-24 (OpenPGP), CRC-32 (IEEE, Castagnoli), CRC-64 (XZ, NVMe).

### Cryptographic hashes

SHA-224, SHA-256, SHA-384, SHA-512, SHA-512/256, SHA3-224, SHA3-256, SHA3-384,
SHA3-512, SHAKE128, SHAKE256, BLAKE3, Ascon-Hash256, Ascon-Xof128.

### Fast hashes

XXH3-64, XXH3-128, RapidHash-64, RapidHash-128.

## Contributing

For perf-gap work on Blake3, start with `just bench-blake3-core`.
Runner/architecture policy for CI and benchmarking is pinned in `ARCHITECTURE_MATRIX.md`.

## License

MIT OR Apache-2.0
