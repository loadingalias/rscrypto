//! Extendable-output function (XOF) trait.

/// Extendable-output function producing an arbitrary number of bytes.
///
/// This trait intentionally has no `std::io::Read` dependency; it is usable in
/// `no_std` environments.
pub trait Xof: Clone {
  /// Squeeze output bytes into `out`.
  fn squeeze(&mut self, out: &mut [u8]);
}
