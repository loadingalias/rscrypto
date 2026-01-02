//! powerpc64 hardware-accelerated CRC-16 kernels (VPMSUMD-style).
//!
//! This is a VPMSUMD implementation of the 128-byte width32 folding algorithm
//! used by the reflected CRC-16 CLMUL backends.
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

use super::keys::{CRC16_CCITT_KEYS_REFLECTED, CRC16_IBM_KEYS_REFLECTED};

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
  ///
  /// On `powerpc64le` this is a no-op. On big-endian `powerpc64`, we byte-swap
  /// each 64-bit lane so the folding algorithm sees the same lane values as on
  /// little-endian platforms.
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

  /// Fold 16 bytes (reflected width32 folding primitive):
  /// `self.low ⊗ coeff.high ⊕ self.high ⊗ coeff.low`.
  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn fold_16(self, coeff: Self) -> Self {
    // VPMSUMD computes: lane0(self)*lane0(coeff) ⊕ lane1(self)*lane1(coeff).
    // The folding primitive wants cross terms, so swap coefficient lanes.
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

  /// Fold 16 bytes down to the "width32" reduction state (reflected mode).
  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn fold_width32_reflected(self, high: u64, low: u64) -> Self {
    // Stage 1: 16B -> 8B (fold high 64 into low 64).
    let clmul = Self::mul64(self.low_64(), low);
    let shifted = Self::new(0, self.high_64());
    let mut state = clmul ^ shifted;

    // Stage 2: 8B -> 4B (fold top 32 bits of low64).
    let mask2 = Self::new(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_0000_0000);
    let masked = state & mask2;
    let shifted_high = (state.low_64() & 0xFFFF_FFFF).strict_shl(32);
    let clmul = Self::mul64(shifted_high, high);
    state = clmul ^ masked;

    state
  }

  /// Barrett reduction for reflected width32; returns the updated CRC state.
  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn barrett_width32_reflected(self, poly: u64, mu: u64) -> u32 {
    // Mirror the x86 reduction scheme (2 multiplies + xor, extract hi32).
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
unsafe fn update_simd_width32_reflected(state: u32, first: &[Simd; 8], rest: &[[Simd; 8]], keys: &[u64; 23]) -> u32 {
  let mut x = *first;
  x[0] = x[0].to_le();
  x[1] = x[1].to_le();
  x[2] = x[2].to_le();
  x[3] = x[3].to_le();
  x[4] = x[4].to_le();
  x[5] = x[5].to_le();
  x[6] = x[6].to_le();
  x[7] = x[7].to_le();

  x[0] ^= Simd::new(0, state as u64);

  let coeff_128b = Simd::new(keys[4], keys[3]);
  for chunk in rest {
    x[0] = x[0].fold_16_reflected(coeff_128b, chunk[0].to_le());
    x[1] = x[1].fold_16_reflected(coeff_128b, chunk[1].to_le());
    x[2] = x[2].fold_16_reflected(coeff_128b, chunk[2].to_le());
    x[3] = x[3].fold_16_reflected(coeff_128b, chunk[3].to_le());
    x[4] = x[4].fold_16_reflected(coeff_128b, chunk[4].to_le());
    x[5] = x[5].fold_16_reflected(coeff_128b, chunk[5].to_le());
    x[6] = x[6].fold_16_reflected(coeff_128b, chunk[6].to_le());
    x[7] = x[7].fold_16_reflected(coeff_128b, chunk[7].to_le());
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
unsafe fn crc16_width32_vpmsum(mut state: u16, data: &[u8], keys: &[u64; 23], portable: fn(u16, &[u8]) -> u16) -> u16 {
  let (left, middle, right) = data.align_to::<[Simd; 8]>();
  let Some((first, rest)) = middle.split_first() else {
    return portable(state, data);
  };

  state = portable(state, left);
  let state32 = update_simd_width32_reflected(state as u32, first, rest, keys);
  state = state32 as u16;
  portable(state, right)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public Safe Kernels
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-16/CCITT VPMSUMD kernel.
///
/// # Safety
///
/// Dispatcher verifies VPMSUMD before selecting this kernel.
#[inline]
pub fn crc16_ccitt_vpmsum_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies VPMSUMD before selecting this kernel.
  unsafe {
    crc16_width32_vpmsum(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/IBM VPMSUMD kernel.
///
/// # Safety
///
/// Dispatcher verifies VPMSUMD before selecting this kernel.
#[inline]
pub fn crc16_ibm_vpmsum_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies VPMSUMD before selecting this kernel.
  unsafe { crc16_width32_vpmsum(crc, data, &CRC16_IBM_KEYS_REFLECTED, super::portable::crc16_ibm_slice8) }
}
