# rscrypto

[![Crates.io](https://img.shields.io/crates/v/rscrypto.svg)](https://crates.io/crates/rscrypto)
[![docs.rs](https://img.shields.io/docsrs/rscrypto)](https://docs.rs/rscrypto)
[![CI](https://github.com/loadingalias/rscrypto/actions/workflows/commit.yaml/badge.svg)](https://github.com/loadingalias/rscrypto/actions/workflows/commit.yaml)
[![MSRV](https://img.shields.io/badge/MSRV-1.94.0-blue.svg)](https://blog.rust-lang.org/)

Zero-dependency-by-default Rust checksums and hashes. No C FFI, no vendored C/C++, `no_std` support, and ISA dispatch where it helps.

## Quick Start

### Checksum

```rust
use rscrypto::{Checksum, Crc32C};

let data = b"hello world";

let checksum = Crc32C::checksum(data);

let mut h = Crc32C::new();
h.update(b"hello ");
h.update(b"world");
assert_eq!(h.finalize(), checksum);
h.reset();
```

### Digest

```rust
use rscrypto::{Digest, Sha256};

let data = b"hello world";

let digest = Sha256::digest(data);

let mut h = Sha256::new();
h.update(b"hello ");
h.update(b"world");
assert_eq!(h.finalize(), digest);
h.reset();
```

### XOF

```rust
use rscrypto::{Shake256, Xof};

let data = b"hello world";

let mut h = Shake256::new();
h.update(data);
let mut xof = h.finalize_xof();
let mut out = [0u8; 64];
xof.squeeze(&mut out);
h.reset();

let mut oneshot = Shake256::xof(data);
let mut same = [0u8; 64];
oneshot.squeeze(&mut same);
assert_eq!(out, same);
```

## Canonical Surface

```rust
use rscrypto::{
  Blake3, Checksum, Crc32C, Digest, RapidHash, Sha256, Shake256, Xof, Xxh3,
};

let checksum = Crc32C::checksum(b"data");
let digest = Blake3::digest(b"data");

let mut xof = Shake256::xof(b"data");
let mut out = [0u8; 32];
xof.squeeze(&mut out);

let mut h = Sha256::new();
h.update(b"part1");
h.update(b"part2");
let _ = h.finalize();
h.reset();

let _ = Xxh3::hash(b"data");
let _ = RapidHash::hash(b"data");
```

Root exports are intentionally small:

```rust
pub use traits::{Checksum, ChecksumCombine, Digest, Xof, FastHash, VerificationError};
pub use traits::ct;

pub use checksum::{Crc16Ccitt, Crc16Ibm, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme};
pub use hashes::crypto::{
  Sha224, Sha256, Sha384, Sha512, Sha512_256,
  Sha3_224, Sha3_256, Sha3_384, Sha3_512,
  Shake128, Shake128Xof, Shake256, Shake256Xof,
  Blake3, Blake3Xof,
  AsconHash256,
  AsconXof, AsconXofReader,
};
pub use hashes::fast::{Xxh3, Xxh3_128, RapidHash, RapidHash128};
```

Advanced surfaces stay explicit:

- `rscrypto::checksum::config::*`
- `rscrypto::checksum::introspect::*`
- `rscrypto::hashes::introspect::*`
- `rscrypto::platform::*`
- `rscrypto::hashes::fast::*`

## Examples

- `cargo run --example basic`
  The canonical checksum, digest, XOF, fast-hash, and I/O adapter specimen.
- `cargo run --example introspect`
  Advanced checksum and hash dispatch reporting.
- `cargo run --example parallel --features parallel`
  CRC combine-based chunked processing.

## Feature Flags

| Feature | Default | Purpose |
|---------|---------|---------|
| `std` | Yes | runtime CPU detection and I/O adapters |
| `alloc` | Yes | buffered checksum wrappers |
| `checksums` | Yes | CRC families and checksum helpers |
| `hashes` | Yes | digest, XOF, and fast-hash families |
| `parallel` | No | rayon-backed BLAKE3 parallel work |
| `diag` | No | checksum dispatch diagnostics |
| `testing` | No | internal validation hooks |

## no_std

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["checksums"] }
```

Without `std`, runtime detection and I/O adapters are unavailable. Portable implementations remain usable.

## Development

```bash
just check-all
just test
```

## License

MIT OR Apache-2.0
