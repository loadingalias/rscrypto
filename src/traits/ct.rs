//! Secret-handling utilities for cryptographic operations.

#[inline(always)]
fn byte_difference(left: &[u8], right: &[u8]) -> u64 {
  let mut difference = 0u64;
  let mut left_chunks = left.chunks_exact(8);
  let mut right_chunks = right.chunks_exact(8);

  for (left_chunk, right_chunk) in left_chunks.by_ref().zip(right_chunks.by_ref()) {
    let (Ok(left_bytes), Ok(right_bytes)) = (<&[u8; 8]>::try_from(left_chunk), <&[u8; 8]>::try_from(right_chunk))
    else {
      return u64::MAX;
    };
    difference |= u64::from_ne_bytes(*left_bytes) ^ u64::from_ne_bytes(*right_bytes);
  }

  let mut remainder = 0u8;
  for (left_byte, right_byte) in left_chunks.remainder().iter().zip(right_chunks.remainder()) {
    remainder |= left_byte ^ right_byte;
  }
  difference | u64::from(remainder)
}

/// Compare two fixed-size byte arrays with work independent of their contents.
///
/// This is crate-private by design. Public comparison policy belongs to the
/// concrete cryptographic type that owns the bytes. Optimized machine-code
/// evidence is tracked separately in `ct.toml`; source structure is not a
/// universal constant-time guarantee.
#[inline(always)]
#[must_use]
#[allow(dead_code)]
pub(crate) fn fixed_eq<const N: usize>(left: &[u8; N], right: &[u8; N]) -> bool {
  byte_difference(left, right) == 0
}

/// Compare two byte slices whose lengths are public protocol inputs.
///
/// Length mismatch is intentionally observable. Equal-length contents are
/// traversed without content-dependent exits. Keep callers individually
/// classified in `ct.toml`; fixed-shape owner types must use [`fixed_eq`].
#[inline]
#[must_use]
#[allow(dead_code)]
pub(crate) fn public_len_eq(left: &[u8], right: &[u8]) -> bool {
  if left.len() != right.len() {
    return false;
  }
  byte_difference(left, right) == 0
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

mod word_zero_sealed {
  /// Marker for primitive integer types whose zero representation is
  /// `0` and whose `write_volatile` of zero is a sound clear.
  ///
  /// Sealed: only the integer types listed here are accepted as scratch
  /// types for [`zeroize_words_no_fence`] / [`zeroize_words`]. New
  /// implementors must be reviewed for soundness (no padding, no Drop).
  #[allow(dead_code)]
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

pub(crate) use word_zero_sealed::WordZero;

/// Volatile-zero a slice of `WordZero` integers without a compiler fence.
///
/// Use for word-shaped scratch buffers (compression states, Argon2 working
/// blocks, HMAC accumulators) that hand-rolled `for word in words { ... }`
/// loops over `core::ptr::write_volatile` patterns. Caller is responsible
/// for emitting a single `compiler_fence(SeqCst)` after all related
/// zeroizations.
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn zeroize_words_no_fence<T: WordZero>(words: &mut [T]) {
  for word in words {
    // SAFETY: `word` is a valid, aligned, dereferenceable pointer to `T`.
    // `T: WordZero` guarantees `T` is a primitive integer with no padding
    // or Drop, so `write_volatile(word, T::ZERO)` is a sound clear.
    unsafe { core::ptr::write_volatile(word, T::ZERO) };
  }
}

/// Volatile-zero a slice of `WordZero` integers and emit a compiler fence.
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn zeroize_words<T: WordZero>(words: &mut [T]) {
  zeroize_words_no_fence(words);
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn fixed_eq_checks_every_position() {
    let value = [0x5a; 64];
    assert!(fixed_eq(&value, &value));
    for index in [0, value.len() / 2, value.len() - 1] {
      let mut different = value;
      different[index] ^= 1;
      assert!(!fixed_eq(&value, &different));
    }
  }

  #[test]
  fn public_len_eq_exposes_only_length_and_result() {
    assert!(public_len_eq(b"abcdef", b"abcdef"));
    assert!(!public_len_eq(b"abcdef", b"abcdeg"));
    assert!(!public_len_eq(b"abcdef", b"abcde"));
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
