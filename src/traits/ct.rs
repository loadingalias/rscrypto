//! Secret-handling utilities for cryptographic operations.

/// Opaque result of a content-independent cryptographic comparison.
///
/// Safe code can obtain a decision only from a semantic owner such as a key,
/// shared secret, or authentication tag. The private representation prevents
/// implicit conversion or inspection; [`declassify`](Self::declassify) is the
/// single explicit boundary that exposes the equality result as a `bool`.
///
/// `CtDecision` preserves the source-level comparison structure. Constant-time
/// claims still depend on the exact compiler, target, features, and generated
/// binary covered by the release evidence in `ct.toml`.
#[must_use = "a cryptographic comparison decision must be composed or explicitly declassified"]
pub struct CtDecision {
  mask: u8,
}

impl core::fmt::Debug for CtDecision {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.write_str("CtDecision(..)")
  }
}

impl CtDecision {
  const TRUE_MASK: u8 = u8::MAX;

  #[cfg(any(
    test,
    feature = "aegis256",
    feature = "aes-gcm",
    feature = "aes-gcm-siv",
    feature = "aes-siv",
    feature = "argon2",
    feature = "ascon-aead",
    feature = "blake3",
    feature = "chacha20poly1305",
    feature = "ecdsa-p256",
    feature = "ecdsa-p384",
    feature = "ed25519",
    feature = "hmac",
    feature = "hmac-sha3",
    feature = "kmac",
    feature = "ml-kem",
    feature = "poly1305",
    feature = "rsa",
    feature = "scrypt",
    feature = "x25519",
    feature = "xchacha20poly1305"
  ))]
  #[inline(always)]
  const fn from_difference(difference: u64) -> Self {
    let nonzero = ((difference | difference.wrapping_neg()) >> 63) as u8;
    Self {
      mask: nonzero.wrapping_sub(1),
    }
  }

  /// Expose the comparison result as a branchable boolean.
  ///
  /// Call this only at the semantic boundary where revealing equality is
  /// intended. Declassification consumes the decision so the opaque value
  /// cannot accidentally be reused after that boundary.
  #[inline(always)]
  #[must_use]
  pub const fn declassify(self) -> bool {
    self.mask == Self::TRUE_MASK
  }

  #[cfg(any(feature = "ed25519", feature = "kmac", feature = "rsa"))]
  #[inline(always)]
  pub(crate) const fn into_u8(self) -> u8 {
    self.mask & 1
  }

  #[cfg(feature = "ml-kem")]
  #[inline(always)]
  pub(crate) const fn into_mask(self) -> u8 {
    self.mask
  }
}

impl core::ops::BitAnd for CtDecision {
  type Output = Self;

  #[inline(always)]
  fn bitand(self, rhs: Self) -> Self::Output {
    Self {
      mask: self.mask & rhs.mask,
    }
  }
}

impl core::ops::BitOr for CtDecision {
  type Output = Self;

  #[inline(always)]
  fn bitor(self, rhs: Self) -> Self::Output {
    Self {
      mask: self.mask | rhs.mask,
    }
  }
}

impl core::ops::Not for CtDecision {
  type Output = Self;

  #[inline(always)]
  fn not(self) -> Self::Output {
    Self {
      mask: self.mask ^ Self::TRUE_MASK,
    }
  }
}

#[cfg(any(
  test,
  feature = "aegis256",
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "aes-siv",
  feature = "argon2",
  feature = "ascon-aead",
  feature = "blake3",
  feature = "chacha20poly1305",
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "hmac",
  feature = "hmac-sha3",
  feature = "kmac",
  feature = "ml-kem",
  feature = "poly1305",
  feature = "rsa",
  feature = "scrypt",
  feature = "x25519",
  feature = "xchacha20poly1305"
))]
#[inline(always)]
fn byte_difference(left: &[u8], right: &[u8]) -> u64 {
  let mut difference = 0u64;
  let (left_chunks, left_remainder) = left.as_chunks::<8>();
  let (right_chunks, right_remainder) = right.as_chunks::<8>();

  for (left_chunk, right_chunk) in left_chunks.iter().zip(right_chunks) {
    difference |= u64::from_ne_bytes(*left_chunk) ^ u64::from_ne_bytes(*right_chunk);
  }

  let mut remainder = 0u8;
  for (left_byte, right_byte) in left_remainder.iter().zip(right_remainder) {
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
#[cfg(any(
  test,
  feature = "aegis256",
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "aes-siv",
  all(feature = "argon2", feature = "phc-strings"),
  feature = "ascon-aead",
  feature = "blake3",
  feature = "chacha20poly1305",
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "hmac",
  feature = "hmac-sha3",
  feature = "ml-kem",
  feature = "poly1305",
  all(feature = "scrypt", feature = "phc-strings"),
  feature = "x25519",
  feature = "xchacha20poly1305"
))]
#[inline(always)]
pub(crate) fn fixed_eq<const N: usize>(left: &[u8; N], right: &[u8; N]) -> CtDecision {
  // SECURITY: Keep the accumulated word opaque before declassification. LLVM can otherwise fold
  // equality into target-specific vector reductions; exact binary evidence still owns the
  // constant-time claim.
  CtDecision::from_difference(core::hint::black_box(byte_difference(left, right)))
}

/// Compare two byte slices whose lengths are public protocol inputs.
///
/// Length mismatch is intentionally observable. Equal-length contents are
/// traversed without content-dependent exits. Keep callers individually
/// classified in `ct.toml`; fixed-shape owner types must use [`fixed_eq`].
#[cfg(any(
  test,
  feature = "argon2",
  feature = "kmac",
  feature = "ml-kem",
  feature = "rsa",
  feature = "scrypt"
))]
#[inline]
pub(crate) fn public_len_eq(left: &[u8], right: &[u8]) -> CtDecision {
  if left.len() != right.len() {
    return CtDecision::from_difference(1);
  }
  CtDecision::from_difference(byte_difference(left, right))
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

#[cfg(any(
  feature = "aes-gcm",
  feature = "ascon-aead",
  feature = "blake2b",
  feature = "blake2s",
  feature = "blake3",
  feature = "chacha20poly1305",
  feature = "poly1305",
  feature = "sha2",
  feature = "sha3",
  feature = "xchacha20poly1305"
))]
mod word_zero_sealed {
  /// Marker for primitive integer types whose zero representation is
  /// `0` and whose `write_volatile` of zero is a sound clear.
  ///
  /// Sealed: only the integer types listed here are accepted as scratch
  /// types for [`zeroize_words_no_fence`] / [`zeroize_words`]. New
  /// implementors must be reviewed for soundness (no padding, no Drop).
  pub(crate) trait WordZero: Copy {
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

#[cfg(any(
  feature = "aes-gcm",
  feature = "ascon-aead",
  feature = "blake2b",
  feature = "blake2s",
  feature = "blake3",
  feature = "chacha20poly1305",
  feature = "poly1305",
  feature = "sha2",
  feature = "sha3",
  feature = "xchacha20poly1305"
))]
pub(crate) use word_zero_sealed::WordZero;

/// Volatile-zero a slice of `WordZero` integers without a compiler fence.
///
/// Use for word-shaped scratch buffers (compression states, Argon2 working
/// blocks, HMAC accumulators) that hand-rolled `for word in words { ... }`
/// loops over `core::ptr::write_volatile` patterns. Caller is responsible
/// for emitting a single `compiler_fence(SeqCst)` after all related
/// zeroizations.
#[cfg(any(
  feature = "aes-gcm",
  feature = "ascon-aead",
  feature = "blake2b",
  feature = "blake2s",
  feature = "blake3",
  feature = "chacha20poly1305",
  feature = "poly1305",
  feature = "sha2",
  feature = "sha3",
  feature = "xchacha20poly1305"
))]
#[inline(always)]
pub(crate) fn zeroize_words_no_fence<T: WordZero>(words: &mut [T]) {
  for word in words {
    // SAFETY: `word` is a valid, aligned, dereferenceable pointer to `T`.
    // `T: WordZero` guarantees `T` is a primitive integer with no padding
    // or Drop, so `write_volatile(word, T::ZERO)` is a sound clear.
    unsafe { core::ptr::write_volatile(word, T::ZERO) };
  }
}

/// Volatile-zero a slice of `WordZero` integers and emit a compiler fence.
#[cfg(any(
  feature = "aes-gcm",
  feature = "ascon-aead",
  feature = "blake3",
  feature = "sha2",
  feature = "sha3"
))]
#[inline(always)]
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
    assert!(fixed_eq(&value, &value).declassify());
    for index in [0, value.len() / 2, value.len() - 1] {
      let mut different = value;
      different[index] ^= 1;
      assert!(!fixed_eq(&value, &different).declassify());
    }
  }

  #[test]
  fn public_len_eq_exposes_only_length_and_result() {
    assert!(public_len_eq(b"abcdef", b"abcdef").declassify());
    assert!(!public_len_eq(b"abcdef", b"abcdeg").declassify());
    assert!(!public_len_eq(b"abcdef", b"abcde").declassify());
  }

  #[test]
  fn decisions_compose_before_declassification() {
    let equal = fixed_eq(b"equal", b"equal");
    let different = fixed_eq(b"equal", b"other");
    assert!(!(equal & different).declassify());

    let equal = fixed_eq(b"equal", b"equal");
    let different = fixed_eq(b"equal", b"other");
    assert!((equal | different).declassify());

    assert!((!fixed_eq(b"equal", b"other")).declassify());
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
