//! Pure Rust cryptography with hardware acceleration.
//!
//! `rscrypto` provides high-performance cryptographic primitives with automatic
//! CPU feature detection and optimal kernel selection. Zero dependencies, `no_std`
//! compatible, and hardware-accelerated on x86_64, aarch64, and more.
//!
//! # Quick Start
//!
//! ```
//! use rscrypto::{Checksum, Crc32C};
//!
//! // One-shot computation
//! let crc = Crc32C::checksum(b"hello world");
//! assert_eq!(crc, 0xC99465AA);
//!
//! // Streaming computation
//! let mut hasher = Crc32C::new();
//! hasher.update(b"hello ");
//! hasher.update(b"world");
//! assert_eq!(hasher.finalize(), crc);
//! ```
//!
//! Hash algorithms follow the same top-level pattern when the `hashes` feature
//! is enabled:
//!
//! ```
//! # #[cfg(feature = "hashes")]
//! # {
//! use rscrypto::{Digest, Sha256};
//!
//! let digest = Sha256::digest(b"hello world");
//! assert_eq!(digest.len(), 32);
//! # }
//! ```
//!
//! # Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std` | Yes | Enables runtime CPU detection for optimal dispatch |
//! | `alloc` | Yes | Enables buffered types (implied by `std`) |
//! | `checksums` | Yes | CRC-16, CRC-24, CRC-32, and CRC-64 algorithms |
//! | `hashes` | Yes | Cryptographic and fast hash families |
//! | `parallel` | No | Rayon-based parallel hashing (Blake3) |
//!
//! ## `no_std` Usage
//!
//! ```toml
//! [dependencies]
//! rscrypto = { version = "0.1", default-features = false, features = ["checksums"] }
//! ```
//!
//! Without `std`, hardware acceleration uses compile-time feature detection only.

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::indexing_slicing))]
// Power SIMD backends require nightly-only SIMD/asm + target-feature support.
#![cfg_attr(
  target_arch = "powerpc64",
  feature(asm_experimental_arch, portable_simd, powerpc_target_feature)
)]
// s390x VGFM backend uses vector asm + portable SIMD.
#![cfg_attr(
  target_arch = "s390x",
  feature(asm_experimental_arch, asm_experimental_reg, portable_simd)
)]
// riscv64 ZVBC backend uses vector target features + inline asm.
#![cfg_attr(
  target_arch = "riscv64",
  feature(asm_experimental_arch, asm_experimental_reg, riscv_target_feature, portable_simd)
)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

// Internal modules (not published as separate crates)
#[doc(hidden)]
pub mod backend;
pub mod platform;
pub mod traits;

#[cfg(feature = "checksums")]
pub mod checksum;

#[cfg(feature = "hashes")]
pub mod hashes;

// ─── Checksum re-exports ────────────────────────────────────────────────────

#[cfg(all(feature = "checksums", feature = "alloc"))]
pub use checksum::{
  BufferedCrc16Ccitt, BufferedCrc16Ibm, BufferedCrc24OpenPgp, BufferedCrc32, BufferedCrc32C, BufferedCrc32Castagnoli,
  BufferedCrc32Ieee, BufferedCrc64, BufferedCrc64Nvme, BufferedCrc64Xz,
};
#[cfg(feature = "checksums")]
pub use checksum::{
  Crc16Ccitt, Crc16Config, Crc16Force, Crc16Ibm, Crc24Config, Crc24Force, Crc24OpenPgp, Crc32, Crc32C, Crc32Castagnoli,
  Crc32Config, Crc32Force, Crc32Ieee, Crc64, Crc64Config, Crc64Force, Crc64Nvme, Crc64Xz, DispatchInfo,
  KernelIntrospect, is_hardware_accelerated, kernel_for, platform_describe,
};
// ─── Hash re-exports ────────────────────────────────────────────────────────
#[cfg(feature = "hashes")]
pub use hashes::crypto::{
  AsconHash256, AsconXof128, AsconXof128Xof, Blake3, Blake3Xof, Sha3_224, Sha3_256, Sha3_384, Sha3_512, Sha224, Sha256,
  Sha384, Sha512, Sha512_256, Shake128, Shake128Xof, Shake256, Shake256Xof,
};
#[cfg(feature = "hashes")]
pub use hashes::fast::{RapidHash64, RapidHash128, Xxh3_64, Xxh3_128};
#[cfg(all(feature = "hashes", feature = "std"))]
pub use hashes::{DigestReader, DigestWriter};
pub use traits::ct;
// ─── Trait re-exports ───────────────────────────────────────────────────────
pub use traits::{Checksum, ChecksumCombine};
#[cfg(feature = "hashes")]
pub use traits::{Digest, FastHash, Xof};
