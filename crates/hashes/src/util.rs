#![allow(clippy::indexing_slicing)] // Fixed-size array indexing and block parsing

#[inline(always)]
pub const fn rotr32(x: u32, n: u32) -> u32 {
  x.rotate_right(n)
}

#[inline(always)]
pub const fn rotr64(x: u64, n: u32) -> u64 {
  x.rotate_right(n)
}
