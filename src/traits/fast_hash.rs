//! Fast non-cryptographic hash traits (**NOT CRYPTO**).

use core::fmt::Debug;

/// A fast non-cryptographic hash.
///
/// These hashes are suitable for hash tables, sharding, fingerprints, and other
/// non-adversarial settings. They are **not** suitable for signatures, MACs,
/// password hashing, or untrusted inputs where collision attacks matter.
///
/// This trait is intentionally one-shot. Streaming APIs for fast hashes often
/// require algorithm-specific buffering and are exposed as concrete types.
///
/// # Examples
///
/// ```
/// use rscrypto::{FastHash, Xxh3};
///
/// let hash = Xxh3::hash(b"hello world");
/// let seeded = Xxh3::hash_with_seed(42, b"hello world");
/// assert_ne!(hash, seeded);
/// ```
pub trait FastHash {
  /// Output size in bytes.
  const OUTPUT_SIZE: usize;

  /// Hash output type.
  type Output: Copy + Eq + Debug + Default;

  /// Seed type (typically `u64`).
  type Seed: Copy + Debug + Default;

  /// Compute the hash of `data` using a default seed.
  #[inline(always)]
  #[must_use]
  fn hash(data: &[u8]) -> Self::Output {
    Self::hash_with_seed(Self::Seed::default(), data)
  }

  /// Compute the hash of `data` using `seed`.
  #[must_use]
  fn hash_with_seed(seed: Self::Seed, data: &[u8]) -> Self::Output;
}
