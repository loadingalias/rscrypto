//! Pure Rust cryptography with hardware acceleration.
//!
//! `rscrypto` provides high-performance cryptographic primitives with automatic
//! CPU feature detection and optimal kernel selection. Zero dependencies, `no_std`
//! compatible, and hardware-accelerated on x86_64, aarch64, and more.
//!
//! # Quick Start
//!
//! ```
//! # #[cfg(feature = "checksums")]
//! # {
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
//! # }
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
// NIGHTLY: riscv_ext_intrinsics provides sha256{sum,sig}{0,1} Zknh intrinsics.
#![cfg_attr(
  target_arch = "riscv64",
  feature(
    asm_experimental_arch,
    asm_experimental_reg,
    riscv_target_feature,
    portable_simd,
    riscv_ext_intrinsics
  )
)]
// NIGHTLY: riscv32 SHA-256 Zknh kernel uses scalar crypto intrinsics.
#![cfg_attr(target_arch = "riscv32", feature(riscv_ext_intrinsics))]
#![cfg_attr(docsrs, feature(doc_cfg))]
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

// ─── Compile-time trait assertions ─────────────────────────────────────────
//
// Every public type must be Send + Sync + Debug.  Most must also be Clone.
// These static assertions fail the build if any contract is broken.

#[cfg(test)]
mod send_sync_assertions {
  #![allow(unused_imports)]
  use super::*;

  fn assert_send_sync<T: Send + Sync>() {}
  fn assert_clone<T: Clone>() {}
  fn assert_debug<T: core::fmt::Debug>() {}

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

  // ── Clone + Debug assertions ──────────────────────────────────────────

  #[test]
  fn platform_types_are_clone_and_debug() {
    assert_clone::<platform::Caps>();
    assert_clone::<platform::Arch>();
    assert_clone::<platform::Detected>();
    assert_clone::<platform::OverrideError>();
    assert_clone::<platform::Description>();
    assert_clone::<traits::error::VerificationError>();

    assert_debug::<platform::Caps>();
    assert_debug::<platform::Arch>();
    assert_debug::<platform::Detected>();
    assert_debug::<platform::OverrideError>();
    assert_debug::<platform::Description>();
    assert_debug::<traits::error::VerificationError>();

    // Backend types with Clone + Debug
    assert_clone::<backend::dispatch::Candidate<fn()>>();
    assert_clone::<backend::dispatch::Selected<fn()>>();
    assert_clone::<backend::dispatch::SelectionError>();
    assert_clone::<backend::tier::KernelTier>();
    assert_clone::<backend::family::KernelFamily>();
    assert_clone::<backend::family::KernelSubfamily>();
    assert_clone::<backend::policy::PolicyTunables>();
    assert_clone::<backend::policy::SelectionPolicy>();
    assert_clone::<backend::policy::ForceMode>();

    assert_debug::<backend::dispatch::Candidate<fn()>>();
    assert_debug::<backend::dispatch::Selected<fn()>>();
    assert_debug::<backend::dispatch::SelectionError>();
    assert_debug::<backend::tier::KernelTier>();
    assert_debug::<backend::family::KernelFamily>();
    assert_debug::<backend::family::KernelSubfamily>();
    assert_debug::<backend::policy::PolicyTunables>();
    assert_debug::<backend::policy::SelectionPolicy>();
    assert_debug::<backend::policy::ForceMode>();
  }

  #[test]
  #[cfg(feature = "checksums")]
  fn checksum_types_are_clone_and_debug() {
    assert_clone::<Crc16Ccitt>();
    assert_clone::<Crc16Ibm>();
    assert_clone::<Crc24OpenPgp>();
    assert_clone::<Crc32>();
    assert_clone::<Crc32C>();
    assert_clone::<Crc64>();
    assert_clone::<Crc64Nvme>();
    assert_clone::<Crc16Force>();
    assert_clone::<Crc16Config>();
    assert_clone::<Crc24Force>();
    assert_clone::<Crc24Config>();
    assert_clone::<Crc32Force>();
    assert_clone::<Crc32Config>();
    assert_clone::<Crc64Force>();
    assert_clone::<Crc64Config>();
    assert_clone::<DispatchInfo>();

    assert_debug::<Crc16Ccitt>();
    assert_debug::<Crc16Ibm>();
    assert_debug::<Crc24OpenPgp>();
    assert_debug::<Crc32>();
    assert_debug::<Crc32C>();
    assert_debug::<Crc64>();
    assert_debug::<Crc64Nvme>();
    assert_debug::<Crc16Force>();
    assert_debug::<Crc16Config>();
    assert_debug::<Crc24Force>();
    assert_debug::<Crc24Config>();
    assert_debug::<Crc32Force>();
    assert_debug::<Crc32Config>();
    assert_debug::<Crc64Force>();
    assert_debug::<Crc64Config>();
    assert_debug::<DispatchInfo>();
  }

  #[test]
  #[cfg(all(feature = "checksums", feature = "alloc"))]
  fn buffered_checksum_types_are_clone_and_debug() {
    assert_debug::<BufferedCrc16Ccitt>();
    assert_debug::<BufferedCrc16Ibm>();
    assert_debug::<BufferedCrc24OpenPgp>();
    assert_debug::<BufferedCrc32>();
    assert_debug::<BufferedCrc32C>();
    assert_debug::<BufferedCrc64>();
    assert_debug::<BufferedCrc64Nvme>();
  }

  #[test]
  #[cfg(feature = "hashes")]
  fn hash_types_are_clone_and_debug() {
    assert_clone::<Sha256>();
    assert_clone::<Sha224>();
    assert_clone::<Sha512>();
    assert_clone::<Sha384>();
    assert_clone::<Sha512_256>();
    assert_clone::<Sha3_256>();
    assert_clone::<Sha3_224>();
    assert_clone::<Sha3_512>();
    assert_clone::<Sha3_384>();
    assert_clone::<Shake128>();
    assert_clone::<Shake256>();
    assert_clone::<Shake128Xof>();
    assert_clone::<Shake256Xof>();
    assert_clone::<AsconHash256>();
    assert_clone::<AsconXof128>();
    assert_clone::<AsconXof128Xof>();
    assert_clone::<Blake3>();
    assert_clone::<Blake3Xof>();
    assert_clone::<Xxh3_64>();
    assert_clone::<Xxh3_128>();
    assert_clone::<RapidHash64>();
    assert_clone::<RapidHash128>();

    assert_debug::<Sha256>();
    assert_debug::<Sha224>();
    assert_debug::<Sha512>();
    assert_debug::<Sha384>();
    assert_debug::<Sha512_256>();
    assert_debug::<Sha3_256>();
    assert_debug::<Sha3_224>();
    assert_debug::<Sha3_512>();
    assert_debug::<Sha3_384>();
    assert_debug::<Shake128>();
    assert_debug::<Shake256>();
    assert_debug::<Shake128Xof>();
    assert_debug::<Shake256Xof>();
    assert_debug::<AsconHash256>();
    assert_debug::<AsconXof128>();
    assert_debug::<AsconXof128Xof>();
    assert_debug::<Blake3>();
    assert_debug::<Blake3Xof>();
    assert_debug::<Xxh3_64>();
    assert_debug::<Xxh3_128>();
    assert_debug::<RapidHash64>();
    assert_debug::<RapidHash128>();
  }

  #[test]
  #[cfg(all(feature = "checksums", feature = "std"))]
  fn io_adapter_types_are_debug() {
    assert_debug::<traits::io::ChecksumReader<std::io::Cursor<Vec<u8>>, Crc32C>>();
    assert_debug::<traits::io::ChecksumWriter<Vec<u8>, Crc32C>>();
  }

  #[test]
  #[cfg(all(feature = "hashes", feature = "std"))]
  fn digest_io_adapter_types_are_debug() {
    assert_debug::<DigestReader<std::io::Cursor<Vec<u8>>, Sha256>>();
    assert_debug::<DigestWriter<Vec<u8>, Sha256>>();
  }
}
