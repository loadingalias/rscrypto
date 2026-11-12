//! IBM Z Adler-32 kernel (z/Vector `VSUMB`, `VMLEB`, `VMLOB`, and `VSUMH`).
//!
//! The kernel keeps the block sums of [`super::blocks`] in 32-bit lanes, four 16-byte vectors per
//! block. `VSUMB` adds each group of four bytes into one 32-bit lane for `bytes`. For `weighted`,
//! `VMLEB` and `VMLOB` multiply the even and the odd bytes by their weights into 16-bit lanes,
//! one add pairs them, and `VSUMH` adds each pair of 16-bit lanes into one 32-bit lane. IBM Z
//! numbers vector elements in memory order, so each byte meets its own weight.
//!
//! # Safety
//!
//! [`update_safe`] relies on Adler-32's `select`, which returns it only when cached detection
//! reports the vector facility: for automatic selection and for the force that `config::get()`
//! keeps only then. The IBM Z kernel tables that name it all require the vector facility.

use core::arch::s390x::{
  vec_add, vec_mule, vec_mulo, vec_sum4, vec_xl, vector_unsigned_char, vector_unsigned_int, vector_unsigned_short,
};

use super::blocks::{BLOCK, CHUNK_BLOCKS, WEIGHTS, finish, fold};

/// Canonical name of the kernel.
pub(in crate::checksum) const KERNEL_NAME: &str = "s390x/vsum";

/// A 32-bit `bytes` lane gains at most `4·4·255 = 4080` per block, so its `prefix` lane stays
/// below `4080·k²/2` after `k` blocks. A 16-bit product lane holds two products, at most
/// `255·(64 + 63) = 32385`, and a 32-bit `weighted` lane gains at most `255·616` per block, as for
/// the x86-64 SSSE3 kernel.
const _: () = assert!(4080 * CHUNK_BLOCKS * (CHUNK_BLOCKS - 1) / 2 <= u32::MAX as usize);
const _: () = assert!(255 * 616 * CHUNK_BLOCKS <= u32::MAX as usize);

/// # Safety
/// Calling outside a function that enables the vector facility requires it.
#[inline]
#[target_feature(enable = "vector")]
fn load(bytes: &[u8; 16]) -> vector_unsigned_char {
  // SAFETY: The function attribute enables the vector facility, and `bytes` is 16 readable
  // bytes; this load has no alignment requirement.
  unsafe { vec_xl(0, bytes.as_ptr()) }
}

/// The sum of the 32-bit lanes of `v`.
#[inline]
fn sum_lanes(v: vector_unsigned_int) -> u64 {
  // SAFETY: Both types are 16 bytes, and every bit pattern is a valid `[u32; 4]`.
  let lanes: [u32; 4] = unsafe { core::mem::transmute(v) };
  lanes.into_iter().map(u64::from).sum()
}

/// # Safety
/// Calling outside a function that enables the vector facility requires it.
#[target_feature(enable = "vector")]
fn update(mut adler: u32, data: &[u8]) -> u32 {
  let (blocks, tail) = data.as_chunks::<BLOCK>();
  let quarters = WEIGHTS.as_chunks::<16>().0;
  let weights = [
    load(&quarters[0]),
    load(&quarters[1]),
    load(&quarters[2]),
    load(&quarters[3]),
  ];
  let zero8 = load(&[0; 16]);
  // SAFETY: The function attribute enables the vector facility for these register-only operations;
  // the lane bounds above keep every 16-bit and 32-bit addition exact.
  unsafe {
    let zero16: vector_unsigned_short = core::mem::transmute(zero8);
    let zero32: vector_unsigned_int = core::mem::transmute(zero8);
    for chunk in blocks.chunks(CHUNK_BLOCKS) {
      let (mut bytes, mut prefix, mut weighted) = (zero32, zero32, zero32);
      for block in chunk {
        prefix = vec_add(prefix, bytes);
        for (quarter, &weight) in block.as_chunks::<16>().0.iter().zip(&weights) {
          let v = load(quarter);
          bytes = vec_add(bytes, vec_sum4(v, zero8));
          let products: vector_unsigned_short = vec_add(vec_mule(v, weight), vec_mulo(v, weight));
          weighted = vec_add(weighted, vec_sum4(products, zero16));
        }
      }
      let weighted = sum_lanes(prefix)
        .strict_mul(BLOCK as u64)
        .strict_add(sum_lanes(weighted));
      adler = fold(adler, chunk.len().strict_mul(BLOCK), (sum_lanes(bytes), weighted));
    }
  }
  finish(adler, tail)
}

/// Safe wrapper for the kernel.
#[inline]
pub(in crate::checksum) fn update_safe(adler: u32, data: &[u8]) -> u32 {
  // SAFETY: Only `select`, when cached detection reports the vector facility, and the IBM Z
  // kernel tables, which all require it, select this kernel.
  unsafe { update(adler, data) }
}

#[cfg(test)]
mod tests {
  use crate::checksum::adler32::{blocks::checks, config::Adler32Force};

  /// The kernel matches the portable kernel on a CPU with the vector facility.
  #[test]
  fn kernel_matches_portable() {
    if !cfg!(miri) && crate::platform::caps().has(crate::platform::caps::s390x::VECTOR) {
      checks::assert_matches_portable(super::KERNEL_NAME, super::update_safe);
    }
  }

  #[test]
  fn the_forced_kernel_runs_through_the_public_api() {
    checks::assert_forced_kernel_runs(Adler32Force::Vsum, super::KERNEL_NAME);
  }
}
