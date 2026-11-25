//! Kernel dispatch introspection for hashes.
//!
//! This module provides user-friendly APIs to inspect which kernels are selected
//! for the current platform, without impacting hot-path performance.
//!
//! # Examples
//!
//! ```
//! use rscrypto::{
//!   Blake3, Sha256,
//!   hashes::introspect::{DispatchInfo, kernel_for},
//! };
//!
//! let info = DispatchInfo::current();
//! assert!(!format!("{info}").is_empty());
//!
//! // Per-algorithm kernel selection for a specific input size
//! assert!(!kernel_for::<Sha256>(1024).is_empty());
//! assert!(!kernel_for::<Blake3>(4096).is_empty());
//! ```

pub use crate::platform::{DispatchInfo, KernelIntrospect};

/// Returns the kernel name selected for a specific algorithm and buffer size.
///
/// This is useful for verifying size-based kernel transitions.
///
/// # Examples
///
/// ```
/// use rscrypto::{Sha256, hashes::introspect::kernel_for};
///
/// let small = kernel_for::<Sha256>(64);
/// let large = kernel_for::<Sha256>(65536);
/// assert!(!small.is_empty());
/// assert!(!large.is_empty());
/// ```
#[inline]
#[must_use]
pub fn kernel_for<T: KernelIntrospect>(len: usize) -> &'static str {
  T::kernel_name_for_len(len)
}

/// Compress SHA-256 blocks through the currently selected kernel.
///
/// This requires the internal compiler cfg and `diag`, and is hidden from docs. It exists so
/// benchmark probes can isolate raw selected-kernel block compression from the
/// public digest/update/finalize wrappers.
#[cfg(all(rscrypto_internal, feature = "diag", feature = "sha2"))]
#[doc(hidden)]
#[inline]
pub fn sha256_compress_blocks_for_bench(state: &mut [u32; 8], blocks: &[u8]) {
  let compress = crate::hashes::crypto::sha256::dispatch::compress_dispatch();
  compress.call(state, blocks);
}

macro_rules! impl_hash_kernel_introspect {
  ($ty:path, $kernel_for_len:path) => {
    impl KernelIntrospect for $ty {
      #[inline]
      fn kernel_name_for_len(len: usize) -> &'static str {
        $kernel_for_len(len)
      }
    }
  };
}

#[cfg(feature = "sha2")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha224,
  crate::hashes::crypto::sha224::dispatch::kernel_name_for_len
);
#[cfg(feature = "sha2")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha256,
  crate::hashes::crypto::sha256::dispatch::kernel_name_for_len
);
#[cfg(feature = "sha2")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha384,
  crate::hashes::crypto::sha384::dispatch::kernel_name_for_len
);
#[cfg(feature = "sha2")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha512,
  crate::hashes::crypto::sha512::dispatch::kernel_name_for_len
);
#[cfg(feature = "sha2")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha512_256,
  crate::hashes::crypto::sha512_256::dispatch::kernel_name_for_len
);

#[cfg(feature = "sha3")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha3_224,
  crate::hashes::crypto::keccak::kernel_name_for_len
);
#[cfg(feature = "sha3")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha3_256,
  crate::hashes::crypto::keccak::kernel_name_for_len
);
#[cfg(feature = "sha3")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha3_384,
  crate::hashes::crypto::keccak::kernel_name_for_len
);
#[cfg(feature = "sha3")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha3_512,
  crate::hashes::crypto::keccak::kernel_name_for_len
);
#[cfg(feature = "sha3")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Shake128,
  crate::hashes::crypto::keccak::kernel_name_for_len
);
#[cfg(feature = "sha3")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Shake256,
  crate::hashes::crypto::keccak::kernel_name_for_len
);
#[cfg(feature = "sha3")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Cshake128,
  crate::hashes::crypto::keccak::kernel_name_for_len
);
#[cfg(feature = "sha3")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Cshake256,
  crate::hashes::crypto::keccak::kernel_name_for_len
);

#[cfg(feature = "blake3")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Blake3,
  crate::hashes::crypto::blake3::dispatch::kernel_name_for_len
);

#[cfg(feature = "blake2b")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Blake2b256,
  crate::hashes::crypto::blake2b::kernel_name_for_len
);
#[cfg(feature = "blake2b")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Blake2b512,
  crate::hashes::crypto::blake2b::kernel_name_for_len
);
#[cfg(feature = "blake2s")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Blake2s128,
  crate::hashes::crypto::blake2s::kernel_name_for_len
);
#[cfg(feature = "blake2s")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Blake2s256,
  crate::hashes::crypto::blake2s::kernel_name_for_len
);

#[cfg(feature = "ascon-hash")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::AsconHash256,
  crate::hashes::crypto::ascon::dispatch::kernel_name_for_len
);
#[cfg(feature = "ascon-hash")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::AsconXof,
  crate::hashes::crypto::ascon::dispatch::kernel_name_for_len
);
#[cfg(feature = "ascon-hash")]
impl_hash_kernel_introspect!(
  crate::hashes::crypto::AsconCxof128,
  crate::hashes::crypto::ascon::dispatch::kernel_name_for_len
);

#[cfg(feature = "xxh3")]
impl_hash_kernel_introspect!(
  crate::hashes::fast::xxh3::Xxh3_64,
  crate::hashes::fast::xxh3::dispatch::kernel_name64_for_len
);
#[cfg(feature = "xxh3")]
impl_hash_kernel_introspect!(
  crate::hashes::fast::xxh3::Xxh3_128,
  crate::hashes::fast::xxh3::dispatch::kernel_name128_for_len
);
#[cfg(test)]
mod tests {
  /// Apple AArch64 always has the SHA3 instructions, and production Keccak runs its SHA3 rounds
  /// there. The report once read a separate table and said "portable".
  #[test]
  #[cfg(all(feature = "sha3", target_vendor = "apple", target_feature = "sha3"))]
  fn keccak_reports_the_kernel_that_runs() {
    for len in [0, 136, 4096, 1 << 20] {
      assert_eq!(super::kernel_for::<crate::Sha3_256>(len), "aarch64-sha3");
      assert_eq!(super::kernel_for::<crate::Shake256>(len), "aarch64-sha3");
    }
  }
}
