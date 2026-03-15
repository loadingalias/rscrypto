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

// ─── Compile-time Send + Sync assertions ──────────────────────────────────
//
// Every public type must be Send + Sync.  These static assertions fail the
// build if the contract is ever broken by a field change.

#[cfg(test)]
mod send_sync_assertions {
  #![allow(unused_imports)]
  use super::*;

  fn assert_send_sync<T: Send + Sync>() {}

  #[test]
  fn public_types_are_send_and_sync() {
    // ── Traits (object-safety is separate; this checks the types) ──
    assert_send_sync::<traits::error::VerificationError>();

    // ── Platform ──
    assert_send_sync::<platform::Caps>();
    assert_send_sync::<platform::Arch>();
    assert_send_sync::<platform::Detected>();
    assert_send_sync::<platform::OverrideError>();
    assert_send_sync::<platform::Description>();

    // ── Backend ──
    assert_send_sync::<backend::cache::OnceCache<u64>>();
    assert_send_sync::<backend::cache::PolicyCache<u32, u32>>();
    assert_send_sync::<backend::dispatch::Candidate<fn()>>();
    assert_send_sync::<backend::dispatch::Selected<fn()>>();
    assert_send_sync::<backend::dispatch::SelectionError>();
    assert_send_sync::<backend::tier::KernelTier>();
    assert_send_sync::<backend::family::KernelFamily>();
    assert_send_sync::<backend::family::KernelSubfamily>();
    assert_send_sync::<backend::policy::PolicyTunables>();
    assert_send_sync::<backend::policy::SelectionPolicy>();
    assert_send_sync::<backend::policy::ForceMode>();

    #[cfg(feature = "std")]
    assert_send_sync::<backend::dispatch::GenericDispatcher<fn()>>();
  }

  #[test]
  #[cfg(feature = "checksums")]
  fn checksum_types_are_send_and_sync() {
    // CRC-16
    assert_send_sync::<Crc16Ccitt>();
    assert_send_sync::<Crc16Ibm>();
    assert_send_sync::<Crc16Force>();
    assert_send_sync::<Crc16Config>();

    // CRC-24
    assert_send_sync::<Crc24OpenPgp>();
    assert_send_sync::<Crc24Force>();
    assert_send_sync::<Crc24Config>();

    // CRC-32
    assert_send_sync::<Crc32>();
    assert_send_sync::<Crc32C>();
    assert_send_sync::<Crc32Force>();
    assert_send_sync::<Crc32Config>();

    // CRC-64
    assert_send_sync::<Crc64>();
    assert_send_sync::<Crc64Nvme>();
    assert_send_sync::<Crc64Force>();
    assert_send_sync::<Crc64Config>();

    // Dispatch / Diagnostics
    assert_send_sync::<DispatchInfo>();

    #[cfg(feature = "diag")]
    {
      assert_send_sync::<checksum::diag::SelectionReason>();
      assert_send_sync::<checksum::diag::Crc32Polynomial>();
      assert_send_sync::<checksum::diag::Crc64Polynomial>();
      assert_send_sync::<checksum::diag::Crc32SelectionDiag>();
      assert_send_sync::<checksum::diag::Crc64SelectionDiag>();
    }
  }

  #[test]
  #[cfg(all(feature = "checksums", feature = "alloc"))]
  fn buffered_checksum_types_are_send_and_sync() {
    assert_send_sync::<BufferedCrc16Ccitt>();
    assert_send_sync::<BufferedCrc16Ibm>();
    assert_send_sync::<BufferedCrc24OpenPgp>();
    assert_send_sync::<BufferedCrc32>();
    assert_send_sync::<BufferedCrc32C>();
    assert_send_sync::<BufferedCrc64>();
    assert_send_sync::<BufferedCrc64Nvme>();
  }

  #[test]
  #[cfg(feature = "hashes")]
  fn hash_types_are_send_and_sync() {
    // SHA-2
    assert_send_sync::<Sha256>();
    assert_send_sync::<Sha224>();
    assert_send_sync::<Sha512>();
    assert_send_sync::<Sha384>();
    assert_send_sync::<Sha512_256>();

    // SHA-3
    assert_send_sync::<Sha3_256>();
    assert_send_sync::<Sha3_224>();
    assert_send_sync::<Sha3_512>();
    assert_send_sync::<Sha3_384>();
    assert_send_sync::<Shake128>();
    assert_send_sync::<Shake256>();
    assert_send_sync::<Shake128Xof>();
    assert_send_sync::<Shake256Xof>();

    // ASCON
    assert_send_sync::<AsconHash256>();
    assert_send_sync::<AsconXof128>();
    assert_send_sync::<AsconXof128Xof>();

    // BLAKE3
    assert_send_sync::<Blake3>();
    assert_send_sync::<Blake3Xof>();

    // Fast hashes
    assert_send_sync::<Xxh3_64>();
    assert_send_sync::<Xxh3_128>();
    assert_send_sync::<RapidHash64>();
    assert_send_sync::<RapidHash128>();
  }

  #[test]
  #[cfg(all(feature = "checksums", feature = "std"))]
  fn io_adapter_types_are_send_and_sync() {
    // ChecksumReader/Writer are Send+Sync when their inner types are
    assert_send_sync::<traits::io::ChecksumReader<std::io::Cursor<Vec<u8>>, Crc32C>>();
    assert_send_sync::<traits::io::ChecksumWriter<Vec<u8>, Crc32C>>();
  }

  #[test]
  #[cfg(all(feature = "hashes", feature = "std"))]
  fn digest_io_adapter_types_are_send_and_sync() {
    assert_send_sync::<DigestReader<std::io::Cursor<Vec<u8>>, Sha256>>();
    assert_send_sync::<DigestWriter<Vec<u8>, Sha256>>();
  }
}
