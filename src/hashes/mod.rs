//! Cryptographic digests and fast non-cryptographic hashes.
//!
//! This crate is `no_std` compatible and has zero library dependencies outside
//! the rscrypto workspace. Dev-only dependencies are used for oracle testing
//! and benchmarking.
//!
//! # Modules
//!
//! - [`crypto`] - Cryptographic hash functions (safe by default).
//! - [`fast`] - Non-cryptographic hashes (**NOT CRYPTO**).
#[doc(hidden)]
pub mod common;
pub mod crypto;
pub mod fast;
#[cfg(feature = "std")]
pub mod io;

mod util;

// Re-export I/O adapters (requires std)
#[cfg(feature = "std")]
pub use io::{DigestReader, DigestWriter};

pub use crate::traits::{Digest, FastHash, Xof};

// Tuning and benchmarking hooks (std-only).
#[doc(hidden)]
#[cfg(feature = "std")]
pub mod bench;

/// Internal testing utilities (for fuzz targets and integration tests).
#[doc(hidden)]
#[cfg(feature = "std")]
pub mod __internal {
  pub mod kernel_test {
    pub use crate::hashes::crypto::{
      ascon::kernel_test::{KernelResult as AsconKernelResult, run_all_ascon_p12_kernels, verify_ascon_p12_kernels},
      blake3::kernel_test::{KernelResult as Blake3KernelResult, run_all_blake3_kernels, verify_blake3_kernels},
      keccak::kernel_test::{
        KernelResult as KeccakKernelResult, run_all_keccakf1600_kernels, verify_keccakf1600_kernels,
      },
      sha224::kernel_test::{KernelResult as Sha224KernelResult, run_all_sha224_kernels, verify_sha224_kernels},
      sha256::kernel_test::{KernelResult as Sha256KernelResult, run_all_sha256_kernels, verify_sha256_kernels},
      sha384::kernel_test::{KernelResult as Sha384KernelResult, run_all_sha384_kernels, verify_sha384_kernels},
      sha512::kernel_test::{KernelResult as Sha512KernelResult, run_all_sha512_kernels, verify_sha512_kernels},
      sha512_256::kernel_test::{
        KernelResult as Sha512_256KernelResult, run_all_sha512_256_kernels, verify_sha512_256_kernels,
      },
    };
  }
}
