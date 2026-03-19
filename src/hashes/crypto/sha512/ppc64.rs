//! SHA-512 ppc64 `vshasigmad` hardware kernel.
//!
//! Uses the POWER8 Crypto `vshasigmad` (Vector SHA-512 Sigma Doubleword)
//! instruction to accelerate the four sigma operations in SHA-512 compression.
//! Each replaces a 3-instruction rotate-shift-xor sequence with a single
//! vector instruction.
//!
//! `vshasigmad VRT,VRA,ST,SIX` computes sigma on 2×u64 vector lanes:
//!   ST=0, SIX=0x0: small_sigma0 (σ₀)
//!   ST=0, SIX=0xf: small_sigma1 (σ₁)
//!   ST=1, SIX=0x0: big_sigma0   (Σ₀)
//!   ST=1, SIX=0xf: big_sigma1   (Σ₁)
//!
//! # Safety
//!
//! Requires POWER8 Crypto (`vcipherd` family). Caller must verify
//! `power::POWER8_CRYPTO`.

#![allow(unsafe_code)]
#![allow(clippy::indexing_slicing)]

use super::BLOCK_LEN;

/// GPR -> VSR -> vshasigmad -> VSR -> GPR. No memory loads/stores, no endianness issues.

#[inline(always)]
fn big_sigma0(x: u64) -> u64 {
  let result: u64;
  // SAFETY: power8-crypto target feature guaranteed by caller.
  unsafe {
    core::arch::asm!(
      "mtvsrd {tmp}, {x}",
      "vshasigmad {tmp}, {tmp}, 1, 0",
      "mfvsrd {out}, {tmp}",
      x = in(reg) x,
      out = lateout(reg) result,
      tmp = out(vreg) _,
      options(pure, nomem, nostack),
    );
  }
  result
}

#[inline(always)]
fn big_sigma1(x: u64) -> u64 {
  let result: u64;
  // SAFETY: power8-crypto target feature guaranteed by caller.
  unsafe {
    core::arch::asm!(
      "mtvsrd {tmp}, {x}",
      "vshasigmad {tmp}, {tmp}, 1, 15",
      "mfvsrd {out}, {tmp}",
      x = in(reg) x,
      out = lateout(reg) result,
      tmp = out(vreg) _,
      options(pure, nomem, nostack),
    );
  }
  result
}

#[inline(always)]
fn small_sigma0(x: u64) -> u64 {
  let result: u64;
  // SAFETY: power8-crypto target feature guaranteed by caller.
  unsafe {
    core::arch::asm!(
      "mtvsrd {tmp}, {x}",
      "vshasigmad {tmp}, {tmp}, 0, 0",
      "mfvsrd {out}, {tmp}",
      x = in(reg) x,
      out = lateout(reg) result,
      tmp = out(vreg) _,
      options(pure, nomem, nostack),
    );
  }
  result
}

#[inline(always)]
fn small_sigma1(x: u64) -> u64 {
  let result: u64;
  // SAFETY: power8-crypto target feature guaranteed by caller.
  unsafe {
    core::arch::asm!(
      "mtvsrd {tmp}, {x}",
      "vshasigmad {tmp}, {tmp}, 0, 15",
      "mfvsrd {out}, {tmp}",
      x = in(reg) x,
      out = lateout(reg) result,
      tmp = out(vreg) _,
      options(pure, nomem, nostack),
    );
  }
  result
}

/// SHA-512 multi-block compression using POWER8 Crypto vshasigmad.
///
/// # Safety
///
/// Caller must ensure POWER8 Crypto features are available.
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
pub(crate) unsafe fn compress_blocks_ppc64_crypto(state: &mut [u64; 8], blocks: &[u8]) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
  let mut chunks = blocks.chunks_exact(BLOCK_LEN);
  for chunk in &mut chunks {
    // SAFETY: `chunks_exact(BLOCK_LEN)` yields slices of exactly `BLOCK_LEN` bytes.
    let block = unsafe { &*(chunk.as_ptr() as *const [u8; BLOCK_LEN]) };
    super::compress_block_with(state, block, big_sigma0, big_sigma1, small_sigma0, small_sigma1);
  }
  debug_assert!(chunks.remainder().is_empty());
}
