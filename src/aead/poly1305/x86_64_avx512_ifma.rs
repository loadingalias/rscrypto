//! Eight-way Poly1305 for ChaCha20-Poly1305 with AVX-512 IFMA.
//!
//! Values use radix 2^44: limbs of 44, 44 and 42 bits for bits 0–43, 44–87 and 88–129.
//! Multiplication by `r` folds the terms of weight 2^132 and above with `2^130 ≡ 5`, so
//! `2^132 ≡ 20 (mod 2^130 - 5)`. Each 52-bit multiply-add gives the low or high 52 bits of a
//! product of limbs below 2^52.
//!
//! The AEAD message is `pad16(aad) || pad16(text) || lengths`, so every block carries the 2^128
//! bit. The authenticator `Σ c_i·r^(n-i+1)` is split as follows: AAD blocks run on the scalar
//! path; the text's full groups of eight blocks run in eight lanes that each multiply by `r^8`,
//! except the last group, whose lane `j` multiplies by `r^(8-j)`; the lanes are summed; the
//! remaining blocks and the length block continue on the scalar path. Only public lengths choose
//! the split. The tag is a function of the key and message alone, so it equals the portable
//! result.

use core::arch::x86_64::*;

use crate::aead::AeadByteLengths;

const MASK44: u64 = (1 << 44) - 1;
const MASK42: u64 = (1 << 42) - 1;

/// A residue in radix 2^44, limbs below 2^52.
#[derive(Clone, Copy)]
struct Limbs([u64; 3]);

impl Limbs {
  /// Split a 130-bit value given as 128 low bits and a high part below 4.
  fn from_u128(low: u128, high: u64) -> Self {
    let bytes = low.to_le_bytes();
    let lo = u64::from_le_bytes([
      bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    let hi = u64::from_le_bytes([
      bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    ]);
    Self([
      lo & MASK44,
      ((lo >> 44) | (hi << 20)) & MASK44,
      (hi >> 24) | (high << 40),
    ])
  }

  /// One message block with the 2^128 bit.
  fn from_block(block: &[u8; 16]) -> Self {
    Self::from_u128(u128::from_le_bytes(*block), 1)
  }

  /// `(self·r) mod (2^130 - 5)`, partially reduced (limbs below 2^45).
  ///
  /// `r` and `r20 = 20·r` hold limbs below 2^44 and 2^49; `self` holds limbs below 2^52.
  fn mul(self, r: Self, r20: Self) -> Self {
    let [h0, h1, h2] = self.0.map(u128::from);
    let [r0, r1, r2] = r.0.map(u128::from);
    let [_, s1, s2] = r20.0.map(u128::from);
    // Every product is below 2^101 and each sum of three is below 2^103.
    let d0 = h0
      .wrapping_mul(r0)
      .wrapping_add(h1.wrapping_mul(s2))
      .wrapping_add(h2.wrapping_mul(s1));
    let d1 = h0
      .wrapping_mul(r1)
      .wrapping_add(h1.wrapping_mul(r0))
      .wrapping_add(h2.wrapping_mul(s2));
    let d2 = h0
      .wrapping_mul(r2)
      .wrapping_add(h1.wrapping_mul(r1))
      .wrapping_add(h2.wrapping_mul(r0));
    carry_wide(d0, d1, d2)
  }

  fn add(self, other: Self) -> Self {
    Self([
      self.0[0].wrapping_add(other.0[0]),
      self.0[1].wrapping_add(other.0[1]),
      self.0[2].wrapping_add(other.0[2]),
    ])
  }

  /// The limbs times 20; inputs below 2^47 stay below 2^52.
  fn times20(self) -> Self {
    Self(self.0.map(|limb| limb.wrapping_mul(20)))
  }
}

/// Carry 44/44/42-bit limbs out of three wide column sums, folding bits at and above 2^130 by 5.
fn carry_wide(d0: u128, d1: u128, d2: u128) -> Limbs {
  let t0 = low_u64(d0) & MASK44;
  let d1 = d1.wrapping_add(d0 >> 44);
  let t1 = low_u64(d1) & MASK44;
  let d2 = d2.wrapping_add(d1 >> 44);
  let t2 = low_u64(d2) & MASK42;
  // `d2 >> 42` is below 2^62, and five times it plus t0 still fits in 64 bits.
  let t0 = t0.wrapping_add(low_u64(d2 >> 42).wrapping_mul(5));
  let t1 = t1.wrapping_add(t0 >> 44);
  Limbs([t0 & MASK44, t1, t2])
}

#[inline(always)]
fn low_u64(value: u128) -> u64 {
  let bytes = value.to_le_bytes();
  u64::from_le_bytes([
    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
  ])
}

/// Scalar Poly1305 state over the shared radix.
struct Scalar {
  h: Limbs,
  r: Limbs,
  r20: Limbs,
}

impl Scalar {
  fn absorb(&mut self, block: &[u8; 16]) {
    self.h = self.h.add(Limbs::from_block(block)).mul(self.r, self.r20);
  }

  /// Absorb a segment, zero-padding its last block.
  fn absorb_padded(&mut self, segment: &[u8]) {
    let (blocks, remainder) = segment.as_chunks::<16>();
    for block in blocks {
      self.absorb(block);
    }
    if !remainder.is_empty() {
      let mut block = [0u8; 16];
      block[..remainder.len()].copy_from_slice(remainder);
      self.absorb(&block);
      crate::traits::ct::zeroize(&mut block);
    }
  }
}

/// Eight lanes of radix-2^44 limbs.
#[derive(Clone, Copy)]
struct Lanes([__m512i; 3]);

/// A lane multiplier and its limbs times 20.
#[derive(Clone, Copy)]
struct Multiplier {
  r: Lanes,
  r20: Lanes,
}

/// `h·r` in every lane, with the same column sums and carries as [`Limbs::mul`].
///
/// # Safety
/// Calling outside an AVX-512F and AVX-512 IFMA function requires both features.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn multiply(h: Lanes, m: Multiplier) -> Lanes {
  let [h0, h1, h2] = h.0;
  let [r0, r1, r2] = m.r.0;
  let [_, s1, s2] = m.r20.0;
  let zero = _mm512_setzero_si512();
  // Each column keeps the low and high 52-bit halves apart: column = low + high·2^52.
  let d0_lo = _mm512_madd52lo_epu64(
    _mm512_madd52lo_epu64(_mm512_madd52lo_epu64(zero, h0, r0), h1, s2),
    h2,
    s1,
  );
  let d0_hi = _mm512_madd52hi_epu64(
    _mm512_madd52hi_epu64(_mm512_madd52hi_epu64(zero, h0, r0), h1, s2),
    h2,
    s1,
  );
  let d1_lo = _mm512_madd52lo_epu64(
    _mm512_madd52lo_epu64(_mm512_madd52lo_epu64(zero, h0, r1), h1, r0),
    h2,
    s2,
  );
  let d1_hi = _mm512_madd52hi_epu64(
    _mm512_madd52hi_epu64(_mm512_madd52hi_epu64(zero, h0, r1), h1, r0),
    h2,
    s2,
  );
  let d2_lo = _mm512_madd52lo_epu64(
    _mm512_madd52lo_epu64(_mm512_madd52lo_epu64(zero, h0, r2), h1, r1),
    h2,
    r0,
  );
  let d2_hi = _mm512_madd52hi_epu64(
    _mm512_madd52hi_epu64(_mm512_madd52hi_epu64(zero, h0, r2), h1, r1),
    h2,
    r0,
  );
  let mask44 = _mm512_set1_epi64(MASK44.cast_signed());
  let mask42 = _mm512_set1_epi64(MASK42.cast_signed());
  // Limb 0 keeps 44 bits; the rest of the column, `low >> 44` plus `high << 8`, carries on.
  let t0 = _mm512_and_si512(d0_lo, mask44);
  let c = _mm512_add_epi64(_mm512_srli_epi64::<44>(d0_lo), _mm512_slli_epi64::<8>(d0_hi));
  let d1 = _mm512_add_epi64(d1_lo, c);
  let t1 = _mm512_and_si512(d1, mask44);
  let c = _mm512_add_epi64(_mm512_srli_epi64::<44>(d1), _mm512_slli_epi64::<8>(d1_hi));
  let d2 = _mm512_add_epi64(d2_lo, c);
  let t2 = _mm512_and_si512(d2, mask42);
  // Limb 2 keeps 42 bits; the excess has weight 2^130 and folds into limb 0 times 5.
  let c = _mm512_add_epi64(_mm512_srli_epi64::<42>(d2), _mm512_slli_epi64::<10>(d2_hi));
  let t0 = _mm512_add_epi64(t0, _mm512_add_epi64(_mm512_slli_epi64::<2>(c), c));
  let t1 = _mm512_add_epi64(t1, _mm512_srli_epi64::<44>(t0));
  Lanes([_mm512_and_si512(t0, mask44), t1, t2])
}

/// Message limbs of eight consecutive blocks, lane `j` holding block `j`.
///
/// # Safety
/// Calling outside an AVX-512F-enabled function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn load_blocks(blocks: &[[u8; 16]; 8]) -> Lanes {
  let words = blocks.as_flattened();
  // SAFETY: `words` is 128 initialized bytes; the two unaligned loads read exactly them.
  let (first, second) = unsafe {
    (
      _mm512_loadu_si512(words.as_ptr().cast()),
      _mm512_loadu_si512(words.as_ptr().add(64).cast()),
    )
  };
  // Even qwords are each block's low half, odd qwords its high half.
  let low = _mm512_permutex2var_epi64(first, _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), second);
  let high = _mm512_permutex2var_epi64(first, _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15), second);
  let mask44 = _mm512_set1_epi64(MASK44.cast_signed());
  Lanes([
    _mm512_and_si512(low, mask44),
    _mm512_and_si512(
      _mm512_or_si512(_mm512_srli_epi64::<44>(low), _mm512_slli_epi64::<20>(high)),
      mask44,
    ),
    _mm512_or_si512(_mm512_srli_epi64::<24>(high), _mm512_set1_epi64(1 << 40)),
  ])
}

/// Broadcast limbs to every lane.
///
/// # Safety
/// Calling outside an AVX-512F-enabled function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn broadcast(limbs: Limbs) -> Lanes {
  let mut out = [_mm512_setzero_si512(); 3];
  for (lanes, &limb) in out.iter_mut().zip(&limbs.0) {
    *lanes = _mm512_set1_epi64(limb.cast_signed());
  }
  Lanes(out)
}

/// Lane `j` holds `values[j]`.
///
/// # Safety
/// Calling outside an AVX-512F-enabled function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn per_lane(values: &[Limbs; 8]) -> Lanes {
  let mut columns = [[0u64; 8]; 3];
  for (lane, value) in values.iter().enumerate() {
    for (column, &limb) in columns.iter_mut().zip(&value.0) {
      column[lane] = limb;
    }
  }
  let mut out = [_mm512_setzero_si512(); 3];
  for (lanes, column) in out.iter_mut().zip(&columns) {
    // SAFETY: Each column is eight initialized u64 values; the unaligned load reads exactly them.
    *lanes = unsafe { _mm512_loadu_si512(column.as_ptr().cast()) };
  }
  Lanes(out)
}

/// Lane `j` of each limb column, summed over the eight lanes.
///
/// # Safety
/// Calling outside an AVX-512F-enabled function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn sum_lanes(lanes: Lanes) -> Limbs {
  // Limbs below 2^52 sum to below 2^55, so the reductions do not overflow.
  let mut out = [0u64; 3];
  for (limb, &column) in out.iter_mut().zip(&lanes.0) {
    *limb = _mm512_reduce_add_epi64(column).cast_unsigned();
  }
  Limbs(out)
}

/// Lane-wise `h + m`.
///
/// # Safety
/// Calling outside an AVX-512F-enabled function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn add_lanes(h: Lanes, m: Lanes) -> Lanes {
  let mut out = h.0;
  for (lanes, &message) in out.iter_mut().zip(&m.0) {
    *lanes = _mm512_add_epi64(*lanes, message);
  }
  Lanes(out)
}

/// Poly1305 tag of the ChaCha20-Poly1305 message for `aad` and `ciphertext` under `key`.
///
/// # Safety
/// The caller must establish AVX-512F and AVX-512 IFMA.
#[target_feature(enable = "avx512f,avx512ifma")]
pub(super) unsafe fn authenticate_aead_ifma(
  aad: &[u8],
  ciphertext: &[u8],
  key: &[u8; 32],
  lengths: AeadByteLengths,
) -> [u8; 16] {
  let (r_bytes, s_bytes) = key.split_at(16);
  let r_value = u128::from_le_bytes(r_bytes.try_into().expect("16-byte r")) & 0x0fff_fffc_0fff_fffc_0fff_fffc_0fff_ffff;
  let r = Limbs::from_u128(r_value, 0);
  let r20 = r.times20();
  // r^1 ... r^8, each partially reduced.
  let mut powers = [r; 8];
  let mut previous = r;
  for power in powers.iter_mut().skip(1) {
    previous = previous.mul(r, r20);
    *power = previous;
  }
  let mut scalar = Scalar {
    h: Limbs([0; 3]),
    r,
    r20,
  };
  scalar.absorb_padded(aad);

  let (groups, rest) = ciphertext.as_chunks::<128>();
  if let Some((last, leading)) = groups.split_last() {
    let r8 = powers[7];
    let step = Multiplier {
      r: broadcast(r8),
      r20: broadcast(r8.times20()),
    };
    // Lane j of the last group multiplies by r^(8-j).
    let mut tail_powers = [r; 8];
    for (lane, power) in tail_powers.iter_mut().enumerate() {
      *power = powers[7usize.strict_sub(lane)];
    }
    let mut tail_powers20 = tail_powers;
    for power in &mut tail_powers20 {
      *power = power.times20();
    }
    let finish = Multiplier {
      r: per_lane(&tail_powers),
      r20: per_lane(&tail_powers20),
    };
    // The AAD's running value enters lane 0 with the first text block.
    let mut h = per_lane(&[
      scalar.h,
      Limbs([0; 3]),
      Limbs([0; 3]),
      Limbs([0; 3]),
      Limbs([0; 3]),
      Limbs([0; 3]),
      Limbs([0; 3]),
      Limbs([0; 3]),
    ]);
    for group in leading {
      let blocks = group.as_chunks::<16>().0.try_into().expect("eight blocks per group");
      let m = load_blocks(blocks);
      h = multiply(add_lanes(h, m), step);
    }
    let blocks = last.as_chunks::<16>().0.try_into().expect("eight blocks per group");
    let m = load_blocks(blocks);
    h = multiply(add_lanes(h, m), finish);
    scalar.h = sum_lanes(h);
  }
  scalar.absorb_padded(rest);
  scalar.absorb(&lengths.to_le_bytes_block());
  finalize(scalar.h, s_bytes)
}

/// Reduce `h` (limbs below 2^56) fully modulo `2^130 - 5` and add `s` modulo 2^128.
fn finalize(h: Limbs, s_bytes: &[u8]) -> [u8; 16] {
  let [mut h0, mut h1, mut h2] = h.0;
  // Two carry passes, folding bits at and above 2^130 by 5, leave h0 below 2^44 + 10 and the
  // other limbs canonical; a third pass makes h0 and h1 canonical and h2 at most 2^42.
  for _ in 0..2 {
    h1 = h1.wrapping_add(h0 >> 44);
    h0 &= MASK44;
    h2 = h2.wrapping_add(h1 >> 44);
    h1 &= MASK44;
    h0 = h0.wrapping_add((h2 >> 42).wrapping_mul(5));
    h2 &= MASK42;
  }
  h1 = h1.wrapping_add(h0 >> 44);
  h0 &= MASK44;
  h2 = h2.wrapping_add(h1 >> 44);
  h1 &= MASK44;
  // h < 2^130 + 2^88 < 2·p. g = h + 5 - 2^130 does not borrow exactly when h >= p.
  let g0 = h0.wrapping_add(5);
  let g1 = h1.wrapping_add(g0 >> 44);
  let g2 = h2.wrapping_add(g1 >> 44).wrapping_sub(1 << 42);
  let keep_g = (g2 >> 63).wrapping_sub(1);
  let h0 = (h0 & !keep_g) | (g0 & MASK44 & keep_g);
  let h1 = (h1 & !keep_g) | (g1 & MASK44 & keep_g);
  let h2 = (h2 & !keep_g) | (g2 & keep_g);
  // The limbs are canonical, so the value is below 2^130; the tag keeps its low 128 bits.
  let value = u128::from(h0)
    .wrapping_add(u128::from(h1) << 44)
    .wrapping_add(u128::from(h2) << 88);
  let s = u128::from_le_bytes(s_bytes.try_into().expect("16-byte s"));
  value.wrapping_add(s).to_le_bytes()
}
