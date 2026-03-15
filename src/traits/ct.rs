//! Constant-time utilities for cryptographic operations.

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
  acc == 0
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
  for byte in buf.iter_mut() {
    // SAFETY: byte is a valid, aligned, dereferenceable pointer to initialized memory.
    unsafe { core::ptr::write_volatile(byte, 0) };
  }
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}
