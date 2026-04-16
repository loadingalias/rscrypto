//! Extendable-output function (XOF) trait.

/// Extendable-output function producing an arbitrary number of bytes.
///
/// This trait intentionally has no `std::io::Read` dependency; it is usable in
/// `no_std` environments.
///
/// # Examples
///
/// ```rust
/// # use rscrypto::traits::Xof;
/// # #[derive(Clone)]
/// # struct MyXof(u8);
/// # impl MyXof {
/// #   fn xof(data: &[u8]) -> Self {
/// #     Self(data.iter().fold(0u8, |acc, &b| acc.wrapping_add(b)))
/// #   }
/// # }
/// # impl Xof for MyXof {
/// #   fn squeeze(&mut self, out: &mut [u8]) {
/// #     for b in out.iter_mut() { *b = self.0; self.0 = self.0.wrapping_add(1); }
/// #   }
/// # }
///
/// let mut xof = MyXof::xof(b"hello world");
/// let mut out = [0u8; 64];
/// xof.squeeze(&mut out);
/// assert_ne!(out, [0u8; 64]);
/// ```
pub trait Xof: Clone {
  /// Squeeze output bytes into `out`.
  fn squeeze(&mut self, out: &mut [u8]);

  /// Squeeze `len` bytes and return them as a `Vec<u8>`.
  #[cfg(feature = "alloc")]
  #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
  #[inline]
  #[must_use]
  fn squeeze_to_vec(&mut self, len: usize) -> alloc::vec::Vec<u8> {
    let mut out = alloc::vec![0u8; len];
    self.squeeze(&mut out);
    out
  }
}
