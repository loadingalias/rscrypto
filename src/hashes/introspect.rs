//! Kernel dispatch introspection for hashes.
//!
//! This module provides user-friendly APIs to inspect which kernels are selected
//! for the current platform, without impacting hot-path performance.
//!
//! # Examples
//!
//! ```
//! use rscrypto::{Blake3, Sha256, hashes::introspect::kernel_for};
//!
//! // Per-algorithm kernel selection for a specific input size
//! println!("SHA-256 @ 1KB: {}", kernel_for::<Sha256>(1024));
//! println!("BLAKE3 @ 4KB: {}", kernel_for::<Blake3>(4096));
//! ```

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
/// println!("Small buffers: {small}");
/// println!("Large buffers: {large}");
/// ```
#[inline]
#[must_use]
pub fn kernel_for<T: HashKernelIntrospect>(len: usize) -> &'static str {
  T::kernel_name_for_len(len)
}

/// Trait for hash types that support kernel introspection.
///
/// This trait is implemented for all public hash algorithm types that support
/// dispatch introspection.
pub trait HashKernelIntrospect {
  /// Returns the kernel name that would be selected for a buffer of `len` bytes.
  fn kernel_name_for_len(len: usize) -> &'static str;
}

macro_rules! impl_hash_kernel_introspect {
  ($ty:path, $kernel_for_len:path) => {
    impl HashKernelIntrospect for $ty {
      #[inline]
      fn kernel_name_for_len(len: usize) -> &'static str {
        $kernel_for_len(len)
      }
    }
  };
}

impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha224,
  crate::hashes::crypto::sha224::dispatch::kernel_name_for_len
);
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha256,
  crate::hashes::crypto::sha256::dispatch::kernel_name_for_len
);
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha384,
  crate::hashes::crypto::sha384::dispatch::kernel_name_for_len
);
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha512,
  crate::hashes::crypto::sha512::dispatch::kernel_name_for_len
);
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha512_256,
  crate::hashes::crypto::sha512_256::dispatch::kernel_name_for_len
);

impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha3_224,
  crate::hashes::crypto::keccak::dispatch::kernel_name_for_len
);
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha3_256,
  crate::hashes::crypto::keccak::dispatch::kernel_name_for_len
);
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha3_384,
  crate::hashes::crypto::keccak::dispatch::kernel_name_for_len
);
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Sha3_512,
  crate::hashes::crypto::keccak::dispatch::kernel_name_for_len
);
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Shake128,
  crate::hashes::crypto::keccak::dispatch::kernel_name_for_len
);
impl_hash_kernel_introspect!(
  crate::hashes::crypto::Shake256,
  crate::hashes::crypto::keccak::dispatch::kernel_name_for_len
);

impl_hash_kernel_introspect!(
  crate::hashes::crypto::Blake3,
  crate::hashes::crypto::blake3::dispatch::kernel_name_for_len
);

impl_hash_kernel_introspect!(
  crate::hashes::crypto::AsconHash256,
  crate::hashes::crypto::ascon::dispatch::kernel_name_for_len
);
impl_hash_kernel_introspect!(
  crate::hashes::crypto::AsconXof,
  crate::hashes::crypto::ascon::dispatch::kernel_name_for_len
);

impl_hash_kernel_introspect!(
  crate::hashes::fast::xxh3::Xxh3_64,
  crate::hashes::fast::xxh3::dispatch::kernel_name_for_len
);
impl_hash_kernel_introspect!(
  crate::hashes::fast::xxh3::Xxh3_128,
  crate::hashes::fast::xxh3::dispatch::kernel_name_for_len
);
impl_hash_kernel_introspect!(
  crate::hashes::fast::rapidhash::RapidHash64,
  crate::hashes::fast::rapidhash::dispatch::kernel_name_for_len
);
impl_hash_kernel_introspect!(
  crate::hashes::fast::rapidhash::RapidHash128,
  crate::hashes::fast::rapidhash::dispatch::kernel_name_for_len
);

#[cfg(test)]
mod tests {
  extern crate alloc;

  use super::*;

  #[test]
  fn kernel_for_digest_is_not_empty() {
    let kernel = kernel_for::<crate::hashes::crypto::Sha256>(1024);
    assert!(!kernel.is_empty());
  }

  #[test]
  fn kernel_for_xof_is_not_empty() {
    let kernel = kernel_for::<crate::hashes::crypto::Shake256>(4096);
    assert!(!kernel.is_empty());
  }

  #[test]
  fn kernel_for_fast_hash_is_not_empty() {
    let kernel = kernel_for::<crate::hashes::fast::xxh3::Xxh3_64>(1024);
    assert!(!kernel.is_empty());
  }
}
