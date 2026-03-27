//! High-performance CRC checksums.
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
//! # Examples
//!
//! ```rust
//! use rscrypto::checksum::{Checksum, ChecksumCombine, Crc32};
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
//! # Advanced
//!
//! Dispatch is automatic by default.
//!
//! - Use [`crate::checksum::config`] for force/config controls.
//! - Use [`crate::checksum::introspect`] for kernel reporting and selection details.
//! - Use [`crate::platform`] for platform detection and override control.
//!
//! # no_std Support
//!
//! This crate is `no_std` compatible. Disable the `std` feature for embedded use:
//!
//! ```toml
//! [dependencies]
//! checksum = { version = "0.1", default-features = false }
//! ```

mod common;

// Internal macros must be declared before modules that use them.
#[macro_use]
mod macros;

#[doc(hidden)]
#[cfg(feature = "alloc")]
#[allow(clippy::indexing_slicing)] // All indexing uses enumerate() indices
pub mod bench;
mod crc16;
mod crc24;
mod crc32;
mod crc64;
#[doc(hidden)]
#[cfg(feature = "diag")]
pub mod diag;
#[doc(hidden)]
pub mod dispatch;
#[doc(hidden)]
pub mod dispatchers;
pub mod introspect;
#[cfg(feature = "std")]
pub mod io;
#[cfg(feature = "alloc")]
pub mod buffered {
  pub use crate::checksum::{
    crc16::{BufferedCrc16Ccitt, BufferedCrc16Ibm},
    crc24::BufferedCrc24OpenPgp,
    crc32::{BufferedCrc32, BufferedCrc32C},
    crc64::{BufferedCrc64, BufferedCrc64Nvme},
  };
}

/// Advanced checksum configuration and force-mode controls.
pub mod config {
  pub use crate::checksum::{
    crc16::{Crc16Config, Crc16Force},
    crc24::{Crc24Config, Crc24Force},
    crc32::{Crc32Config, Crc32Force},
    crc64::{Crc64Config, Crc64Force},
  };
}

#[doc(hidden)]
pub mod __internal {
  /// Kernel testing utilities.
  #[cfg(all(test, feature = "alloc"))]
  pub mod kernel_test {
    pub use crate::checksum::{
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
    pub use crate::checksum::{
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
pub use crc16::{Crc16Ccitt, Crc16Ibm};
pub use crc24::Crc24OpenPgp;
pub use crc32::{Crc32, Crc32C, Crc32Castagnoli, Crc32Ieee};
pub use crc64::{Crc64, Crc64Nvme, Crc64Xz};
// Re-export I/O adapters (requires std)
#[cfg(feature = "std")]
pub use io::{ChecksumReader, ChecksumWriter};

// Re-export traits for convenience
pub use crate::traits::{Checksum, ChecksumCombine};
