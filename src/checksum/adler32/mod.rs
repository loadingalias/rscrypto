//! Adler-32 from RFC 1950 (**NOT CRYPTO**).
//!
//! This module provides [`Adler32`], the checksum of zlib streams.
//!
//! # Kernels
//!
//! - Portable Rust on every target, the sole semantic authority.

pub(crate) mod portable;

use crate::traits::{Checksum, ChecksumCombine};

use portable::{MOD, MOD64};

/// Adler-32 checksum from [RFC 1950](https://www.rfc-editor.org/rfc/rfc1950.html) (**NOT CRYPTO**).
///
/// zlib streams store Adler-32 big-endian: the checksum of the uncompressed data
/// after the compressed data, and the checksum of a preset dictionary in the
/// header. It detects accidental corruption only and gives no authentication.
///
/// [`Checksum::with_initial`] resumes from the checksum of the data so far, and
/// [`ChecksumCombine::combine`] joins the checksums of two parts, as zlib's
/// `adler32` and `adler32_combine` do. The state is two 16-bit sums and keeps
/// no input.
///
/// # Examples
///
/// ```rust
/// use rscrypto::checksum::{Adler32, Checksum, ChecksumCombine};
///
/// assert_eq!(Adler32::checksum(b"Wikipedia"), 0x11E6_0398);
///
/// let (a, b) = b"Wikipedia".split_at(4);
/// let combined = Adler32::combine(Adler32::checksum(a), Adler32::checksum(b), b.len());
/// assert_eq!(combined, 0x11E6_0398);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Adler32 {
  state: u32,
}

impl Adler32 {
  const INIT: u32 = 1;

  /// Returns the kernel name that the selector would choose for `len`.
  #[must_use]
  pub fn kernel_name_for_len(_len: usize) -> &'static str {
    portable::KERNEL_NAME
  }
}

impl Checksum for Adler32 {
  const OUTPUT_SIZE: usize = 4;
  type Output = u32;

  #[inline]
  fn new() -> Self {
    Self { state: Self::INIT }
  }

  #[inline]
  fn with_initial(initial: u32) -> Self {
    let (a, b) = sums(initial);
    Self { state: (b << 16) | a }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.state = portable::update(self.state, data);
  }

  #[inline]
  fn finalize(&self) -> u32 {
    self.state
  }

  #[inline]
  fn reset(&mut self) {
    self.state = Self::INIT;
  }
}

impl Default for Adler32 {
  #[inline]
  fn default() -> Self {
    <Self as Checksum>::new()
  }
}

impl ChecksumCombine for Adler32 {
  /// Computes zlib's `adler32_combine`: over `A || B`, `a = a(A) + a(B) - 1`
  /// and `b = b(A) + b(B) + len(B)·(a(A) - 1)`, both modulo 65521. Only
  /// `len(B)` modulo 65521 matters, so every `usize` length is exact.
  fn combine(crc_a: u32, crc_b: u32, len_b: usize) -> u32 {
    let (a1, b1) = sums(crc_a);
    let (a2, b2) = sums(crc_b);
    let (a1, b1, a2, b2) = (u64::from(a1), u64::from(b1), u64::from(a2), u64::from(b2));
    let len = (len_b as u64) % MOD64;
    let a = portable::reduce(a1.strict_add(a2).strict_add(MOD64 - 1));
    let b = portable::reduce(
      b1.strict_add(b2)
        .strict_add(len.strict_mul(a1))
        .strict_add(MOD64.strict_sub(len)),
    );
    (b << 16) | a
  }
}

/// Split a checksum into its reduced sums `a` (low half) and `b` (high half).
#[inline]
fn sums(value: u32) -> (u32, u32) {
  ((value & 0xffff) % MOD, (value >> 16) % MOD)
}

// Kernel Introspection

#[cfg(feature = "diag")]
impl crate::checksum::introspect::KernelIntrospect for Adler32 {
  fn kernel_name_for_len(len: usize) -> &'static str {
    Self::kernel_name_for_len(len)
  }
}
