//! IBM Z CRC engine: `VGFMG` folding (z13 and later).

use core::{arch::asm, simd::u64x2};

use super::{
  Algo, Keys,
  dispatch::fold_with,
  vector::{Engine, SumOfProducts},
};

/// `VGFMG`: Galois-field multiply sum of doublewords.
pub(in crate::checksum) struct Vgfmg;

impl SumOfProducts for Vgfmg {
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn sum(a: u64x2, b: u64x2) -> u64x2 {
    let out: u64x2;
    // SAFETY: the caller guarantees the vector facility; the instruction reads two vector
    // registers and writes one, with no memory access.
    unsafe {
      asm!(
        "vgfmg {out}, {a}, {b}",
        out = lateout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(nomem, nostack, pure)
      );
    }
    out
  }
}

/// VGFM tier.
///
/// # Safety
///
/// The CPU must support the vector facility, and `data` must hold at least 16 bytes.
#[target_feature(enable = "vector")]
pub(in crate::checksum) unsafe fn vgfm<const REFLECT: bool>(state: u64, data: &[u8], keys: &Keys) -> u64 {
  // SAFETY: this function enables the instructions, and the caller guarantees the length.
  unsafe { Engine::<Vgfmg>::update::<REFLECT>(state, data, keys) }
}

/// Multiplies an engine-domain value by `x^(8 * len)`, for `combine`.
///
/// # Safety
///
/// The CPU must support the vector facility.
#[target_feature(enable = "vector")]
pub(in crate::checksum) unsafe fn shift<const REFLECT: bool>(
  value: u64,
  len: usize,
  powers: &[u64; 64],
  keys: &Keys,
) -> u64 {
  // SAFETY: this function enables the instructions.
  unsafe { Engine::<Vgfmg>::shift::<REFLECT>(value, len, powers, keys) }
}

/// VGFM tier. The selector runs it only on a CPU with the vector facility.
pub(in crate::checksum) fn vgfm_kernel<A: Algo>(register: A::W, data: &[u8]) -> A::W {
  // SAFETY: the selector checks the vector facility.
  unsafe { fold_with::<A>(register, data, vgfm::<true>, vgfm::<false>) }
}

/// `combine` multiplier of the VGFM tier.
pub(in crate::checksum) fn shift_kernel<A: Algo>(value: u64, len: usize) -> u64 {
  let crc = A::CRC;
  // SAFETY: the selector checks the vector facility.
  unsafe {
    if crc.spec.reflected {
      shift::<true>(value, len, &crc.powers, &crc.keys)
    } else {
      shift::<false>(value, len, &crc.powers, &crc.keys)
    }
  }
}
