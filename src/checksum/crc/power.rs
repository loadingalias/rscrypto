//! POWER CRC engine: `VPMSUMD` folding (POWER8 and later).

use core::{arch::asm, simd::u64x2};

use super::{
  Algo, Keys,
  dispatch::fold_with,
  vector::{Engine, SumOfProducts},
};

/// `VPMSUMD`.
pub(in crate::checksum) struct Vpmsumd;

impl SumOfProducts for Vpmsumd {
  #[inline]
  #[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
  unsafe fn sum(a: u64x2, b: u64x2) -> u64x2 {
    let out: u64x2;
    // SAFETY: the caller guarantees POWER8 crypto; the instruction reads two vector registers
    // and writes one, with no memory access.
    unsafe {
      asm!(
        "vpmsumd {out}, {a}, {b}",
        out = lateout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(nomem, nostack, pure)
      );
    }
    out
  }
}

/// VPMSUMD tier.
///
/// # Safety
///
/// The CPU must support POWER8 vector and crypto instructions, and `data` must hold at least 16
/// bytes.
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
pub(in crate::checksum) unsafe fn vpmsum<const REFLECT: bool>(state: u64, data: &[u8], keys: &Keys) -> u64 {
  // SAFETY: this function enables the instructions, and the caller guarantees the length.
  unsafe { Engine::<Vpmsumd>::update::<REFLECT>(state, data, keys) }
}

/// Multiplies an engine-domain value by `x^(8 * len)`, for `combine`.
///
/// # Safety
///
/// The CPU must support POWER8 vector and crypto instructions.
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
pub(in crate::checksum) unsafe fn shift<const REFLECT: bool>(
  value: u64,
  len: usize,
  powers: &[u64; 64],
  keys: &Keys,
) -> u64 {
  // SAFETY: this function enables the instructions.
  unsafe { Engine::<Vpmsumd>::shift::<REFLECT>(value, len, powers, keys) }
}

/// VPMSUMD tier. The selector runs it only on a CPU with POWER8 vector crypto.
pub(in crate::checksum) fn vpmsum_kernel<A: Algo>(register: A::W, data: &[u8]) -> A::W {
  // SAFETY: the selector checks POWER8 vector crypto.
  unsafe { fold_with::<A>(register, data, vpmsum::<true>, vpmsum::<false>) }
}

/// `combine` multiplier of the VPMSUMD tier.
pub(in crate::checksum) fn shift_kernel<A: Algo>(value: u64, len: usize) -> u64 {
  let crc = A::CRC;
  // SAFETY: the selector checks POWER8 vector crypto.
  unsafe {
    if crc.spec.reflected {
      shift::<true>(value, len, &crc.powers, &crc.keys)
    } else {
      shift::<false>(value, len, &crc.powers, &crc.keys)
    }
  }
}
