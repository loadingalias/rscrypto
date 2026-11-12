//! POWER Adler-32 kernel (AltiVec `vmsumubm`).
//!
//! The kernel keeps the block sums of [`super::blocks`] in four 16-byte vector positions per
//! block. `vmsumubm` multiplies sixteen unsigned bytes by sixteen unsigned bytes and adds each
//! group of four products to one 32-bit lane: with the block weights it gives a position's
//! `weighted` sum, and with weights of one its `bytes` sum. Each position keeps its own lanes, so
//! no accumulator waits on another position's multiply.
//!
//! Lane order within a register differs between little- and big-endian POWER, but every data
//! byte meets the weight byte that the same load order placed beside it, and the lanes are only
//! summed, so neither endianness changes a result.
//!
//! # Safety
//!
//! [`update_safe`] relies on Adler-32's `select`, which returns it only when cached detection
//! reports AltiVec: for automatic selection and for the force that `config::get()` keeps only
//! then.

use core::{
  arch::asm,
  simd::{i32x4, u8x16},
};

use super::blocks::{BLOCK, CHUNK_BLOCKS, WEIGHTS, finish, fold};

/// Canonical name of the kernel.
pub(in crate::checksum) const KERNEL_NAME: &str = "power/vmsum";

/// A position's 32-bit `weighted` lane gains at most `255·(64 + 63 + 62 + 61) = 63750` per block,
/// and its `bytes` lane at most `4·255`, so its `prefix` lane stays below `1020·k²/2` after `k`
/// blocks.
const _: () = assert!(63_750 * CHUNK_BLOCKS <= u32::MAX as usize);
const _: () = assert!(1020 * CHUNK_BLOCKS * CHUNK_BLOCKS / 2 <= u32::MAX as usize);

/// For each 32-bit lane, `acc` plus the four products of the lane's bytes of `a` and `b`, as
/// unsigned bytes, modulo 2^32.
///
/// # Safety
/// Calling outside an AltiVec-enabled function requires AltiVec support.
#[inline]
#[target_feature(enable = "altivec")]
fn vmsumubm(a: u8x16, b: u8x16, acc: i32x4) -> i32x4 {
  let out: i32x4;
  // SAFETY: The function attribute enables AltiVec. `vmsumubm` reads its three vector registers
  // and writes one; it accesses no memory and sets no status bit.
  unsafe {
    asm!(
      "vmsumubm {out}, {a}, {b}, {acc}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      acc = in(vreg) acc,
      options(pure, nomem, nostack, preserves_flags),
    );
  }
  out
}

/// Lane-wise `a + b`, modulo 2^32.
///
/// # Safety
/// Calling outside an AltiVec-enabled function requires AltiVec support.
#[inline]
#[target_feature(enable = "altivec")]
fn vadduwm(a: i32x4, b: i32x4) -> i32x4 {
  let out: i32x4;
  // SAFETY: The function attribute enables AltiVec. `vadduwm` reads two vector registers and
  // writes one; it accesses no memory and sets no status bit.
  unsafe {
    asm!(
      "vadduwm {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(pure, nomem, nostack, preserves_flags),
    );
  }
  out
}

/// The sum of every 32-bit lane of `lanes`, as unsigned values.
#[inline]
fn sum_lanes(lanes: &[i32x4; 4]) -> u64 {
  lanes
    .iter()
    .flat_map(|lane| lane.to_array())
    .map(|value| u64::from(value.cast_unsigned()))
    .sum()
}

/// # Safety
/// Calling outside an AltiVec-enabled function requires AltiVec support.
#[target_feature(enable = "altivec")]
fn update(mut adler: u32, data: &[u8]) -> u32 {
  let (blocks, tail) = data.as_chunks::<BLOCK>();
  let quarters = WEIGHTS.as_chunks::<16>().0;
  let weights = [0, 1, 2, 3].map(|i| u8x16::from_array(quarters[i]));
  let ones = u8x16::splat(1);
  let zero = i32x4::splat(0);
  for chunk in blocks.chunks(CHUNK_BLOCKS) {
    let mut bytes = [zero; 4];
    let mut prefix = [zero; 4];
    let mut weighted = [zero; 4];
    for block in chunk {
      for ((((quarter, &weight), bytes), prefix), weighted) in block
        .as_chunks::<16>()
        .0
        .iter()
        .zip(&weights)
        .zip(&mut bytes)
        .zip(&mut prefix)
        .zip(&mut weighted)
      {
        let v = u8x16::from_array(*quarter);
        // The lane bounds above keep each addition below 2^32, so the modular add is exact.
        *prefix = vadduwm(*prefix, *bytes);
        *bytes = vmsumubm(v, ones, *bytes);
        *weighted = vmsumubm(v, weight, *weighted);
      }
    }
    let weighted = sum_lanes(&prefix)
      .strict_mul(BLOCK as u64)
      .strict_add(sum_lanes(&weighted));
    adler = fold(adler, chunk.len().strict_mul(BLOCK), (sum_lanes(&bytes), weighted));
  }
  finish(adler, tail)
}

/// Safe wrapper for the kernel.
#[inline]
pub(in crate::checksum) fn update_safe(adler: u32, data: &[u8]) -> u32 {
  // SAFETY: Only `select` returns this kernel, and only when cached detection reports AltiVec.
  unsafe { update(adler, data) }
}

#[cfg(test)]
mod tests {
  use crate::checksum::adler32::{blocks::checks, config::Adler32Force};

  /// The kernel matches the portable kernel on a CPU with AltiVec.
  #[test]
  fn kernel_matches_portable() {
    if !cfg!(miri) && crate::platform::caps().has(crate::platform::caps::power::ALTIVEC) {
      checks::assert_matches_portable(super::KERNEL_NAME, super::update_safe);
    }
  }

  #[test]
  fn the_forced_kernel_runs_through_the_public_api() {
    checks::assert_forced_kernel_runs(Adler32Force::Vmsum, super::KERNEL_NAME);
  }
}
