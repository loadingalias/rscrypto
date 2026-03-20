//! SHA-256 ppc64 `vshasigmaw` hardware kernel.
//!
//! Uses the POWER8 Crypto `vshasigmaw` (Vector SHA-256 Sigma Word)
//! instruction to accelerate the four sigma operations in SHA-256 compression.
//! Each replaces a 3-instruction rotate-shift-xor sequence with a single
//! vector instruction.
//!
//! `vshasigmaw VRT,VRA,ST,SIX` computes sigma on 4×u32 vector lanes:
//!   ST=0, SIX=0x0: small_sigma0 (σ₀): ROTR(7)  ^ ROTR(18) ^ SHR(3)
//!   ST=0, SIX=0xf: small_sigma1 (σ₁): ROTR(17) ^ ROTR(19) ^ SHR(10)
//!   ST=1, SIX=0x0: big_sigma0   (Σ₀): ROTR(2)  ^ ROTR(13) ^ ROTR(22)
//!   ST=1, SIX=0xf: big_sigma1   (Σ₁): ROTR(6)  ^ ROTR(11) ^ ROTR(25)
//!
//! Scalar wrapper: `mtvsrwz` (GPR→VSR word), `vshasigmaw`, `mfvsrwz`
//! (VSR word→GPR). No memory loads/stores, no endianness issues.
//!
//! # Safety
//!
//! Requires POWER8 Crypto. Caller must verify `power::POWER8_CRYPTO`.

#![allow(unsafe_code)]
#![allow(clippy::indexing_slicing)]

use super::BLOCK_LEN;

/// GPR -> VSR -> vshasigmaw -> VSR -> GPR. No memory loads/stores, no endianness issues.
#[inline(always)]
fn big_sigma0(x: u32) -> u32 {
  let result: u32;
  // SAFETY: power8-crypto target feature guaranteed by caller.
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
  // SAFETY: power8-crypto target feature guaranteed by caller.
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
  // SAFETY: power8-crypto target feature guaranteed by caller.
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
  // SAFETY: power8-crypto target feature guaranteed by caller.
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

/// SHA-256 multi-block compression using POWER8 Crypto vshasigmaw.
///
/// # Safety
///
/// Caller must ensure POWER8 Crypto features are available.
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
pub(crate) unsafe fn compress_blocks_ppc64_crypto(state: &mut [u32; 8], blocks: &[u8]) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
  let mut chunks = blocks.chunks_exact(BLOCK_LEN);
  for chunk in &mut chunks {
    // SAFETY: `chunks_exact(BLOCK_LEN)` yields slices of exactly `BLOCK_LEN` bytes.
    let block = unsafe { &*(chunk.as_ptr() as *const [u8; BLOCK_LEN]) };
    super::compress_block_with(state, block, big_sigma0, big_sigma1, small_sigma0, small_sigma1);
  }
  debug_assert!(chunks.remainder().is_empty());
}
