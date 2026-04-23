//! SHA-256 POWER `vshasigmaw` kernel.
//!
//! POWER8+ exposes `vshasigmaw`, which computes the SHA-256 sigma functions
//! over 32-bit lanes in a vector register. We keep the portable round and
//! schedule structure authoritative, and only swap the sigma primitives onto
//! the hardware instruction.
//!
//! # Safety
//!
//! Requires `altivec`, `vsx`, `power8-vector`, and `power8-crypto`.

#![allow(unsafe_code)]
#![allow(clippy::indexing_slicing)]

use super::BLOCK_LEN;

#[inline(always)]
fn big_sigma0(x: u32) -> u32 {
  let result: u32;
  // SAFETY: caller guarantees POWER8 crypto support before entering the kernel.
  unsafe {
    core::arch::asm!(
      "mtvsrwz {tmp}, {x}",
      "vshasigmaw {tmp}, {tmp}, 1, 0",
      "mfvsrwz {out}, {tmp}",
      x = in(reg) x,
      out = lateout(reg) result,
      tmp = out(vreg) _,
      options(pure, nomem, nostack),
    );
  }
  result
}

#[inline(always)]
fn big_sigma1(x: u32) -> u32 {
  let result: u32;
  // SAFETY: caller guarantees POWER8 crypto support before entering the kernel.
  unsafe {
    core::arch::asm!(
      "mtvsrwz {tmp}, {x}",
      "vshasigmaw {tmp}, {tmp}, 1, 15",
      "mfvsrwz {out}, {tmp}",
      x = in(reg) x,
      out = lateout(reg) result,
      tmp = out(vreg) _,
      options(pure, nomem, nostack),
    );
  }
  result
}

#[inline(always)]
fn small_sigma0(x: u32) -> u32 {
  let result: u32;
  // SAFETY: caller guarantees POWER8 crypto support before entering the kernel.
  unsafe {
    core::arch::asm!(
      "mtvsrwz {tmp}, {x}",
      "vshasigmaw {tmp}, {tmp}, 0, 0",
      "mfvsrwz {out}, {tmp}",
      x = in(reg) x,
      out = lateout(reg) result,
      tmp = out(vreg) _,
      options(pure, nomem, nostack),
    );
  }
  result
}

#[inline(always)]
fn small_sigma1(x: u32) -> u32 {
  let result: u32;
  // SAFETY: caller guarantees POWER8 crypto support before entering the kernel.
  unsafe {
    core::arch::asm!(
      "mtvsrwz {tmp}, {x}",
      "vshasigmaw {tmp}, {tmp}, 0, 15",
      "mfvsrwz {out}, {tmp}",
      x = in(reg) x,
      out = lateout(reg) result,
      tmp = out(vreg) _,
      options(pure, nomem, nostack),
    );
  }
  result
}

/// SHA-256 multi-block compression using POWER crypto sigma instructions.
///
/// # Safety
///
/// Caller must ensure POWER8 crypto support is available.
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
pub(crate) unsafe fn compress_blocks_ppc64_crypto(state: &mut [u32; 8], blocks: &[u8]) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
  let mut chunks = blocks.chunks_exact(BLOCK_LEN);
  for chunk in &mut chunks {
    // SAFETY: `chunks_exact(BLOCK_LEN)` yields slices of exactly `BLOCK_LEN`.
    let block = unsafe { &*(chunk.as_ptr() as *const [u8; BLOCK_LEN]) };
    super::compress_block_with(state, block, big_sigma0, big_sigma1, small_sigma0, small_sigma1);
  }
  debug_assert!(chunks.remainder().is_empty());
}
