//! High-performance CRC checksums with hardware acceleration.
//!
//! This crate provides implementations of common CRC algorithms with automatic
//! hardware acceleration on supported platforms.
//!
//! # Supported Algorithms
//!
//! | Type | Polynomial | Output | Use Cases |
//! |------|------------|--------|-----------|
//! | [`Crc16Ccitt`] | 0x1021 | `u16` | X.25, HDLC, PACTOR, SD |
//! | [`Crc16Ibm`] | 0x8005 | `u16` | Legacy protocols, ARC/IBM |
//! | [`Crc24OpenPgp`] | 0x864CFB | `u32` (24-bit) | OpenPGP (RFC 4880) |
//! | [`Crc32`] | 0x04C11DB7 | `u32` | Ethernet, gzip, zip, PNG |
//! | [`Crc32C`] | 0x1EDC6F41 | `u32` | iSCSI, SCTP, ext4, Btrfs |
//! | [`Crc64`] | 0x42F0E1EBA9EA3693 | `u64` | XZ Utils, 7-Zip |
//! | [`Crc64Nvme`] | 0xAD93D23594C93659 | `u64` | NVMe specification |
//!
//! # Hardware Acceleration
//!
//! The following hardware acceleration paths are automatically selected based on
//! detected CPU features:
//!
//! ## x86_64
//!
//! | Feature | Algorithms | Throughput |
//! |---------|------------|------------|
//! | SSE4.2 CRC32 | CRC-32C | ~20 GB/s |
//! | VPCLMULQDQ | CRC-64 | ~35-40 GB/s |
//! | PCLMULQDQ | CRC-64 | ~15 GB/s |
//!
//! ## aarch64
//!
//! | Feature | Algorithms | Throughput |
//! |---------|------------|------------|
//! | CRC extension | CRC-32/CRC-32C | ~15-25 GB/s |
//! | PMULL + EOR3 | CRC-64 | ~15 GB/s |
//! | PMULL | CRC-64 | ~12 GB/s |
//!
//! # Examples
//!
//! ```rust
//! use checksum::{Checksum, ChecksumCombine, Crc32};
//!
//! // One-shot computation (fastest for complete data)
//! let data = b"123456789";
//! let crc = Crc32::checksum(data);
//! assert_eq!(crc, 0xCBF4_3926);
//!
//! // Streaming computation
//! let mut hasher = Crc32::new();
//! hasher.update(b"1234");
//! hasher.update(b"56789");
//! assert_eq!(hasher.finalize(), crc);
//!
//! // Parallel combine (useful for multi-threaded processing)
//! let (a, b) = data.split_at(4);
//! let crc_a = Crc32::checksum(a);
//! let crc_b = Crc32::checksum(b);
//! let combined = Crc32::combine(crc_a, crc_b, b.len());
//! assert_eq!(combined, crc);
//! ```
//!
//! # Introspection
//!
//! Verify which kernels are selected for your platform:
//!
//! ```rust
//! use checksum::{Crc64, DispatchInfo};
//!
//! // Platform-level info
//! let info = DispatchInfo::current();
//! println!("{info}");
//! // Example output: "Caps(aarch64, [AES, PMULL, ...]) (Apple M1-M3)"
//!
//! // Per-algorithm kernel selection
//! println!("CRC-64 backend: {}", Crc64::backend_name());
//! println!("CRC-64 @ 4KB: {}", Crc64::kernel_name_for_len(4096));
//! ```
//!
//! For generic introspection across types:
//!
//! ```rust
//! use checksum::{Crc64, KernelIntrospect, kernel_for};
//!
//! fn show_kernel<T: KernelIntrospect>(name: &str, len: usize) {
//!   println!("{name} @ {len}B: {}", kernel_for::<T>(len));
//! }
//!
//! show_kernel::<Crc64>("CRC-64/XZ", 4096);
//! ```
//!
//! # no_std Support
//!
//! This crate is `no_std` compatible. Disable the `std` feature for embedded use:
//!
//! ```toml
//! [dependencies]
//! checksum = { version = "0.1", default-features = false }
//! ```

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::indexing_slicing))]
// Power SIMD backends currently require nightly-only SIMD/asm + target-feature support.
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
  feature(asm_experimental_arch, asm_experimental_reg, riscv_target_feature)
)]
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod common;

// Internal macros must be declared before modules that use them.
#[macro_use]
mod macros;

#[cfg(feature = "alloc")]
#[allow(clippy::indexing_slicing)] // All indexing uses enumerate() indices
pub mod bench;
mod crc16;
mod crc24;
mod crc32;
mod crc64;
#[cfg(feature = "diag")]
pub mod diag;
pub mod dispatch;
pub mod dispatchers;
mod introspect;
pub mod tune;

#[doc(hidden)]
pub mod __internal {
  pub use crate::common::kernels::stream_to_index;

  /// Kernel testing utilities (for fuzz targets and integration tests).
  #[cfg(feature = "alloc")]
  pub mod kernel_test {
    pub use crate::{
      crc16::kernel_test::{
        KernelResult as KernelResult16, run_all_crc16_ccitt_kernels, run_all_crc16_ibm_kernels,
        verify_crc16_ccitt_kernels, verify_crc16_ibm_kernels,
      },
      crc24::kernel_test::{
        KernelResult as KernelResult24, run_all_crc24_openpgp_kernels, verify_crc24_openpgp_kernels,
      },
      crc32::kernel_test::{
        KernelResult as KernelResult32, run_all_crc32_ieee_kernels, run_all_crc32c_kernels, verify_crc32_ieee_kernels,
        verify_crc32c_kernels,
      },
      crc64::kernel_test::{
        KernelResult, run_all_crc64_nvme_kernels, run_all_crc64_xz_kernels, verify_crc64_nvme_kernels,
        verify_crc64_xz_kernels,
      },
    };
  }

  /// Property test internals - bitwise reference implementations and constants.
  ///
  /// These are exposed for integration tests in `tests/` that need access to
  /// the mathematical CRC definitions for correctness verification.
  pub mod proptest_internals {
    // Bitwise reference implementations (the mathematical definition of CRC)
    // Polynomial constants for all CRC variants
    // Portable kernel wrappers for cross-validation tests
    pub use crate::{
      common::{
        reference::{crc16_bitwise, crc24_bitwise, crc32_bitwise, crc64_bitwise},
        tables::{
          CRC16_CCITT_POLY, CRC16_IBM_POLY, CRC24_OPENPGP_POLY, CRC32_IEEE_POLY, CRC32C_POLY, CRC64_NVME_POLY,
          CRC64_XZ_POLY,
        },
      },
      crc16::portable::{crc16_ccitt_slice8, crc16_ibm_slice8},
      crc24::portable::crc24_openpgp_slice8,
      crc32::portable::{crc32_slice16_ieee, crc32c_slice16},
      crc64::portable::{crc64_slice16_nvme, crc64_slice16_xz},
    };
  }
}

// Re-export public types
// Re-export buffered types (requires alloc)
#[cfg(feature = "alloc")]
pub use crc16::{BufferedCrc16Ccitt, BufferedCrc16Ibm};
pub use crc16::{Crc16Ccitt, Crc16Config, Crc16Force, Crc16Ibm};
#[cfg(feature = "alloc")]
pub use crc24::BufferedCrc24OpenPgp;
pub use crc24::{Crc24Config, Crc24Force, Crc24OpenPgp};
#[cfg(feature = "alloc")]
pub use crc32::{BufferedCrc32, BufferedCrc32C};
#[cfg(feature = "alloc")]
pub use crc32::{BufferedCrc32Castagnoli, BufferedCrc32Ieee};
pub use crc32::{Crc32, Crc32C, Crc32Castagnoli, Crc32Config, Crc32Force, Crc32Ieee};
#[cfg(feature = "alloc")]
pub use crc64::{BufferedCrc64, BufferedCrc64Nvme, BufferedCrc64Xz};
pub use crc64::{Crc64, Crc64Config, Crc64Force, Crc64Nvme, Crc64Xz};
// Re-export introspection API
pub use dispatch::is_hardware_accelerated;
pub use introspect::{DispatchInfo, KernelIntrospect, kernel_for};
// Re-export platform::describe for convenience
pub use platform::describe as platform_describe;
// Re-export traits for convenience
pub use traits::{Checksum, ChecksumCombine};
