//! Constant-time utilities for cryptographic operations.

/// Constant-time equality comparison.
///
/// Implementors guarantee that the comparison examines every byte regardless
/// of mismatches, preventing timing side-channels.
///
/// # Examples
///
/// ```
/// use rscrypto::ConstantTimeEq;
///
/// let a = [1u8, 2, 3, 4];
/// let b = [1u8, 2, 3, 4];
/// let c = [1u8, 2, 3, 5];
///
/// assert!(a.ct_eq(&b));
/// assert!(!a.ct_eq(&c));
/// ```
pub trait ConstantTimeEq {
  /// Returns `true` if `self` and `other` are equal.
  ///
  /// The comparison is constant-time with respect to the data contents.
  #[must_use]
  fn ct_eq(&self, other: &Self) -> bool;
}

impl<const N: usize> ConstantTimeEq for [u8; N] {
  #[inline]
  fn ct_eq(&self, other: &Self) -> bool {
    constant_time_eq(self, other)
  }
}

impl ConstantTimeEq for [u8] {
  #[inline]
  fn ct_eq(&self, other: &Self) -> bool {
    constant_time_eq(self, other)
  }
}

/// Constant-time byte equality comparison.
///
/// Returns `true` if `a` and `b` have equal length and identical contents.
/// The comparison examines every byte regardless of mismatches, preventing
/// timing side-channels.
///
/// # Examples
///
/// ```
/// use rscrypto::traits::ct::constant_time_eq;
///
/// let a = [1u8, 2, 3, 4];
/// let b = [1u8, 2, 3, 4];
/// let c = [1u8, 2, 3, 5];
///
/// assert!(constant_time_eq(&a, &b));
/// assert!(!constant_time_eq(&a, &c));
/// assert!(!constant_time_eq(&a, &[]));
/// ```
#[inline]
#[must_use]
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
  if a.len() != b.len() {
    return false;
  }
  let mut acc = 0u8;
  for (&x, &y) in a.iter().zip(b.iter()) {
    acc |= x ^ y;
  }
  core::hint::black_box(acc) == 0
}

/// Volatile-zero a byte slice without emitting a compiler fence.
///
/// Use when batching multiple zeroizations under a single fence.
/// Call `compiler_fence(SeqCst)` once after all buffers are zeroed.
#[inline(always)]
pub(crate) fn zeroize_no_fence(buf: &mut [u8]) {
  // SAFETY: align_to_mut returns valid prefix/words/suffix over the same allocation.
  let (prefix, words, suffix) = unsafe { buf.align_to_mut::<u64>() };
  for byte in prefix.iter_mut() {
    // SAFETY: byte is a valid, aligned, dereferenceable pointer to initialized memory.
    unsafe { core::ptr::write_volatile(byte, 0) };
  }
  for word in words.iter_mut() {
    // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
    unsafe { core::ptr::write_volatile(word, 0) };
  }
  for byte in suffix.iter_mut() {
    // SAFETY: byte is a valid, aligned, dereferenceable pointer to initialized memory.
    unsafe { core::ptr::write_volatile(byte, 0) };
  }
}

/// Overwrite a byte slice with zeros using volatile writes.
///
/// The compiler cannot elide these writes, ensuring sensitive data
/// is cleared from memory even if the buffer is not read afterward.
///
/// # Examples
///
/// ```
/// use rscrypto::traits::ct::zeroize;
///
/// let mut buf = [0xFFu8; 16];
/// zeroize(&mut buf);
/// assert_eq!(buf, [0u8; 16]);
/// ```
#[inline(always)]
pub fn zeroize(buf: &mut [u8]) {
  zeroize_no_fence(buf);
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}
