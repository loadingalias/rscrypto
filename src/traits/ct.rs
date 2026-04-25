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
/// timing side-channels for equal-length inputs. Length mismatches return
/// `false` immediately.
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

  let len = a.len();
  let mut acc0 = 0u64;
  let mut acc1 = 0u64;
  let mut acc2 = 0u64;
  let mut acc3 = 0u64;
  let mut i = 0usize;
  let end32 = len & !31;
  let a_ptr = a.as_ptr();
  let b_ptr = b.as_ptr();

  while i < end32 {
    // SAFETY: `end32` rounds `len` down to a multiple of 32, so each unaligned
    // 8-byte load stays within the slice bounds.
    unsafe {
      acc0 |=
        core::ptr::read_unaligned(a_ptr.add(i).cast::<u64>()) ^ core::ptr::read_unaligned(b_ptr.add(i).cast::<u64>());
      acc1 |= core::ptr::read_unaligned(a_ptr.add(i.strict_add(8)).cast::<u64>())
        ^ core::ptr::read_unaligned(b_ptr.add(i.strict_add(8)).cast::<u64>());
      acc2 |= core::ptr::read_unaligned(a_ptr.add(i.strict_add(16)).cast::<u64>())
        ^ core::ptr::read_unaligned(b_ptr.add(i.strict_add(16)).cast::<u64>());
      acc3 |= core::ptr::read_unaligned(a_ptr.add(i.strict_add(24)).cast::<u64>())
        ^ core::ptr::read_unaligned(b_ptr.add(i.strict_add(24)).cast::<u64>());
    }
    i = i.strict_add(32);
  }

  let mut acc = acc0 | acc1 | acc2 | acc3;
  let end8 = len & !7;
  while i < end8 {
    // SAFETY: `end8` rounds `len` down to a multiple of 8, so each unaligned
    // 8-byte load stays within the slice bounds.
    unsafe {
      acc |=
        core::ptr::read_unaligned(a_ptr.add(i).cast::<u64>()) ^ core::ptr::read_unaligned(b_ptr.add(i).cast::<u64>());
    }
    i = i.strict_add(8);
  }

  while i < len {
    // SAFETY: `i < len`, so these raw-pointer reads are in-bounds.
    unsafe {
      acc |= (*a_ptr.add(i) ^ *b_ptr.add(i)) as u64;
    }
    i = i.strict_add(1);
  }

  let mut observed = acc;
  // SAFETY: `observed` is a valid stack slot. The volatile round-trip prevents
  // the optimizer from collapsing the loop into an early-exit byte compare.
  unsafe {
    core::ptr::write_volatile(&mut observed, acc);
    core::ptr::read_volatile(&observed) == 0
  }
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

#[cfg(feature = "pbkdf2")]
mod word_zero_sealed {
  /// Marker for primitive integer types whose zero representation is
  /// `0` and whose `write_volatile` of zero is a sound clear.
  ///
  /// Sealed: only the integer types listed here are accepted as scratch
  /// types for [`zeroize_words_no_fence`] / [`zeroize_words`]. New
  /// implementors must be reviewed for soundness (no padding, no Drop).
  pub trait WordZero: Copy {
    const ZERO: Self;
  }

  impl WordZero for u8 {
    const ZERO: Self = 0;
  }
  impl WordZero for u16 {
    const ZERO: Self = 0;
  }
  impl WordZero for u32 {
    const ZERO: Self = 0;
  }
  impl WordZero for u64 {
    const ZERO: Self = 0;
  }
  impl WordZero for u128 {
    const ZERO: Self = 0;
  }
  impl WordZero for usize {
    const ZERO: Self = 0;
  }
}

#[cfg(feature = "pbkdf2")]
pub(crate) use word_zero_sealed::WordZero;

/// Volatile-zero a slice of `WordZero` integers without a compiler fence.
///
/// Use for word-shaped scratch buffers (compression states, Argon2 working
/// blocks, HMAC accumulators) that hand-rolled `for word in words { ... }`
/// loops over `core::ptr::write_volatile` patterns. Caller is responsible
/// for emitting a single `compiler_fence(SeqCst)` after all related
/// zeroizations.
#[cfg(feature = "pbkdf2")]
#[inline(always)]
pub(crate) fn zeroize_words_no_fence<T: WordZero>(words: &mut [T]) {
  for word in words {
    // SAFETY: `word` is a valid, aligned, dereferenceable pointer to `T`.
    // `T: WordZero` guarantees `T` is a primitive integer with no padding
    // or Drop, so `write_volatile(word, T::ZERO)` is a sound clear.
    unsafe { core::ptr::write_volatile(word, T::ZERO) };
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  // ── constant_time_eq ────────────────────────────────────────────────────

  #[test]
  fn equal_slices() {
    assert!(constant_time_eq(b"abcdef", b"abcdef"));
  }

  #[test]
  fn differ_first_byte() {
    assert!(!constant_time_eq(b"\x00bcdef", b"\xFFbcdef"));
  }

  #[test]
  fn differ_last_byte() {
    assert!(!constant_time_eq(b"abcde\x00", b"abcde\xFF"));
  }

  #[test]
  fn differ_middle_byte() {
    assert!(!constant_time_eq(b"ab\x00def", b"ab\xFFdef"));
  }

  #[test]
  fn both_empty() {
    assert!(constant_time_eq(b"", b""));
  }

  #[test]
  fn one_empty_one_not() {
    assert!(!constant_time_eq(b"", b"x"));
    assert!(!constant_time_eq(b"x", b""));
  }

  #[test]
  fn length_mismatch() {
    assert!(!constant_time_eq(b"short", b"longer"));
    assert!(!constant_time_eq(b"longer", b"short"));
  }

  #[test]
  fn single_byte_equal() {
    assert!(constant_time_eq(&[0x42], &[0x42]));
  }

  #[test]
  fn single_byte_differ() {
    assert!(!constant_time_eq(&[0x00], &[0x01]));
  }

  #[test]
  fn all_zeros_equal() {
    assert!(constant_time_eq(&[0u8; 64], &[0u8; 64]));
  }

  #[test]
  fn all_ones_equal() {
    assert!(constant_time_eq(&[0xFF; 64], &[0xFF; 64]));
  }

  #[test]
  fn equal_misaligned_long_slices() {
    let a = [0x5Au8; 73];
    let b = [0x5Au8; 73];
    assert!(constant_time_eq(&a[1..], &b[1..]));
  }

  #[test]
  fn differ_misaligned_long_slices() {
    let a = [0x5Au8; 73];
    let mut b = [0x5Au8; 73];
    b[64] ^= 0x80;
    assert!(!constant_time_eq(&a[1..], &b[1..]));
  }

  // ── ConstantTimeEq trait ────────────────────────────────────────────────

  #[test]
  fn ct_eq_fixed_array() {
    let a = [1u8, 2, 3, 4];
    let b = [1u8, 2, 3, 4];
    let c = [1u8, 2, 3, 5];
    assert!(a.ct_eq(&b));
    assert!(!a.ct_eq(&c));
  }

  #[test]
  fn ct_eq_slice() {
    let a: &[u8] = &[10, 20, 30];
    let b: &[u8] = &[10, 20, 30];
    let c: &[u8] = &[10, 20, 31];
    assert!(a.ct_eq(b));
    assert!(!a.ct_eq(c));
  }

  // ── zeroize ─────────────────────────────────────────────────────────────

  #[test]
  fn zeroize_clears_buffer() {
    let mut buf = [0xFFu8; 37]; // odd size to exercise prefix/suffix
    zeroize(&mut buf);
    assert!(buf.iter().all(|&b| b == 0));
  }

  #[test]
  fn zeroize_empty_is_noop() {
    let mut buf = [];
    zeroize(&mut buf); // must not panic
  }
}
