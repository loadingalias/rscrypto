//! powerpc64 hardware-accelerated CRC-24/OPENPGP kernels (VPMSUMD-style).
//!
//! CRC-24/OPENPGP is MSB-first. These kernels reuse the reflected width32
//! folding/reduction structure by processing per-byte bit-reversed input and
//! converting the CRC state between OpenPGP and reflected forms.
//!
//! # Safety
//!
//! Uses `unsafe` for powerpc64 SIMD + inline assembly. Callers must ensure the
//! required CPU features are available before executing the accelerated path
//! (the dispatcher does this).
#![allow(unsafe_code)]
#![allow(dead_code)] // Kernels wired up via dispatcher
// SAFETY: All indexing is over fixed-size arrays with in-bounds constant indices.
#![allow(clippy::indexing_slicing)]
// This module is intrinsics-heavy; keep unsafe blocks readable.
#![allow(unsafe_op_in_unsafe_fn)]

use core::{
  arch::asm,
  ops::{BitAnd, BitXor, BitXorAssign},
  simd::i64x2,
};

use super::{
  keys::CRC24_OPENPGP_KEYS_REFLECTED,
  reflected::{crc24_reflected_update_bitrev_bytes, from_reflected_state, to_reflected_state},
};

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
struct Simd(i64x2);

impl BitXor for Simd {
  type Output = Self;

  #[inline]
  fn bitxor(self, other: Self) -> Self {
    Self(self.0 ^ other.0)
  }
}

impl BitXorAssign for Simd {
  #[inline]
  fn bitxor_assign(&mut self, other: Self) {
    *self = *self ^ other;
  }
}

impl BitAnd for Simd {
  type Output = Self;

  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    Self(self.0 & rhs.0)
  }
}

#[inline]
#[must_use]
const fn bitrev_bytes_u64(mut x: u64) -> u64 {
  x = ((x & 0x5555_5555_5555_5555).strict_shl(1)) | ((x.strict_shr(1)) & 0x5555_5555_5555_5555);
  x = ((x & 0x3333_3333_3333_3333).strict_shl(2)) | ((x.strict_shr(2)) & 0x3333_3333_3333_3333);
  x = ((x & 0x0F0F_0F0F_0F0F_0F0F).strict_shl(4)) | ((x.strict_shr(4)) & 0x0F0F_0F0F_0F0F_0F0F);
  x
}

impl Simd {
  #[inline]
  fn new(high: u64, low: u64) -> Self {
    // Match the x86/aarch64 lane layout: lane0 = low, lane1 = high.
    Self(i64x2::from_array([low as i64, high as i64]))
  }

  #[inline]
  fn low_64(self) -> u64 {
    self.0.to_array()[0] as u64
  }

  #[inline]
  fn high_64(self) -> u64 {
    self.0.to_array()[1] as u64
  }

  #[inline]
  fn swap_lanes(self) -> Self {
    let [lo, hi] = self.0.to_array();
    Self(i64x2::from_array([hi, lo]))
  }

  /// Normalize a loaded vector to little-endian lane encoding.
  #[inline]
  #[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
  unsafe fn to_le(self) -> Self {
    #[cfg(target_endian = "little")]
    {
      self
    }

    #[cfg(target_endian = "big")]
    {
      let out: i64x2;
      asm!(
        "xxbrd {out}, {inp}",
        out = lateout(vreg) out,
        inp = in(vreg) self.0,
        options(nomem, nostack, pure)
      );
      Self(out)
    }
  }

  #[inline]
  fn bitrev_bytes(self) -> Self {
    Self::new(bitrev_bytes_u64(self.high_64()), bitrev_bytes_u64(self.low_64()))
  }

  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn vpmsumd(a: i64x2, b: i64x2) -> i64x2 {
    let out: i64x2;
    asm!(
      "vpmsumd {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
    out
  }

  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn mul64(a: u64, b: u64) -> Self {
    let va = i64x2::from_array([a as i64, 0]);
    let vb = i64x2::from_array([b as i64, 0]);
    Self(Self::vpmsumd(va, vb))
  }

  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn fold_16(self, coeff: Self) -> Self {
    Self(Self::vpmsumd(self.0, coeff.swap_lanes().0))
  }

  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn fold_16_reflected(self, coeff: Self, data_to_xor: Self) -> Self {
    data_to_xor ^ self.fold_16(coeff)
  }

  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn fold_width32_reflected(self, high: u64, low: u64) -> Self {
    let clmul = Self::mul64(self.low_64(), low);
    let shifted = Self::new(0, self.high_64());
    let mut state = clmul ^ shifted;

    let mask2 = Self::new(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_0000_0000);
    let masked = state & mask2;
    let shifted_high = (state.low_64() & 0xFFFF_FFFF).strict_shl(32);
    let clmul = Self::mul64(shifted_high, high);
    state = clmul ^ masked;

    state
  }

  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn barrett_width32_reflected(self, poly: u64, mu: u64) -> u32 {
    let t1 = Self::mul64(self.low_64(), mu);
    let l = Self::mul64(t1.low_64(), poly);
    (self ^ l).high_64() as u32
  }
}

#[inline]
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn finalize_lanes_width32_reflected(x: [Simd; 8], keys: &[u64; 23]) -> u32 {
  let mut res = x[7];
  res = x[0].fold_16_reflected(Simd::new(keys[10], keys[9]), res);
  res = x[1].fold_16_reflected(Simd::new(keys[12], keys[11]), res);
  res = x[2].fold_16_reflected(Simd::new(keys[14], keys[13]), res);
  res = x[3].fold_16_reflected(Simd::new(keys[16], keys[15]), res);
  res = x[4].fold_16_reflected(Simd::new(keys[18], keys[17]), res);
  res = x[5].fold_16_reflected(Simd::new(keys[20], keys[19]), res);
  res = x[6].fold_16_reflected(Simd::new(keys[2], keys[1]), res);

  res = res.fold_width32_reflected(keys[6], keys[5]);
  res.barrett_width32_reflected(keys[8], keys[7])
}

#[inline]
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn update_simd_width32_reflected_bitrev(
  state: u32,
  first: &[Simd; 8],
  rest: &[[Simd; 8]],
  keys: &[u64; 23],
) -> u32 {
  let mut x = *first;
  x[0] = x[0].to_le().bitrev_bytes();
  x[1] = x[1].to_le().bitrev_bytes();
  x[2] = x[2].to_le().bitrev_bytes();
  x[3] = x[3].to_le().bitrev_bytes();
  x[4] = x[4].to_le().bitrev_bytes();
  x[5] = x[5].to_le().bitrev_bytes();
  x[6] = x[6].to_le().bitrev_bytes();
  x[7] = x[7].to_le().bitrev_bytes();

  x[0] ^= Simd::new(0, state as u64);

  let coeff_128b = Simd::new(keys[4], keys[3]);
  for chunk in rest {
    x[0] = x[0].fold_16_reflected(coeff_128b, chunk[0].to_le().bitrev_bytes());
    x[1] = x[1].fold_16_reflected(coeff_128b, chunk[1].to_le().bitrev_bytes());
    x[2] = x[2].fold_16_reflected(coeff_128b, chunk[2].to_le().bitrev_bytes());
    x[3] = x[3].fold_16_reflected(coeff_128b, chunk[3].to_le().bitrev_bytes());
    x[4] = x[4].fold_16_reflected(coeff_128b, chunk[4].to_le().bitrev_bytes());
    x[5] = x[5].fold_16_reflected(coeff_128b, chunk[5].to_le().bitrev_bytes());
    x[6] = x[6].fold_16_reflected(coeff_128b, chunk[6].to_le().bitrev_bytes());
    x[7] = x[7].fold_16_reflected(coeff_128b, chunk[7].to_le().bitrev_bytes());
  }

  finalize_lanes_width32_reflected(x, keys)
}

#[inline]
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn crc24_width32_vpmsum_bitrev(mut state: u32, data: &[u8], keys: &[u64; 23]) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd; 8]>();
  let Some((first, rest)) = middle.split_first() else {
    return crc24_reflected_update_bitrev_bytes(state, data);
  };

  state = crc24_reflected_update_bitrev_bytes(state, left);
  state = update_simd_width32_reflected_bitrev(state, first, rest, keys);
  crc24_reflected_update_bitrev_bytes(state, right)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public Safe Kernel
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-24/OPENPGP VPMSUMD kernel.
///
/// # Safety
///
/// Dispatcher verifies VPMSUMD before selecting this kernel.
#[inline]
pub fn crc24_openpgp_vpmsum_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies VPMSUMD before selecting this kernel.
  state = unsafe { crc24_width32_vpmsum_bitrev(state, data, &CRC24_OPENPGP_KEYS_REFLECTED) };
  from_reflected_state(state)
}
