#![allow(clippy::indexing_slicing)] // Fixed-size array indexing and block parsing

#[repr(C, align(64))]
pub(crate) struct Aligned64<T>(pub T);

impl<T> core::ops::Deref for Aligned64<T> {
  type Target = T;

  #[inline(always)]
  fn deref(&self) -> &T {
    &self.0
  }
}

#[inline(always)]
pub const fn rotr32(x: u32, n: u32) -> u32 {
  x.rotate_right(n)
}

#[inline(always)]
pub const fn rotr64(x: u64, n: u32) -> u64 {
  x.rotate_right(n)
}
