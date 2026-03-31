#![allow(clippy::indexing_slicing)]

//! Constant-time POLYVAL universal hash (RFC 8452).
//!
//! POLYVAL operates in GF(2^128) with the irreducible polynomial
//! x^128 + x^127 + x^126 + x^121 + 1 (the bit-reversal of GHASH's
//! polynomial).
//!
//! The portable implementation uses:
//! - **Pornin's bmul64** for constant-time 64×64 carryless multiplication (16 integer multiplies
//!   per call, no table lookups)
//! - **Karatsuba** decomposition for the 128×128 product (6 bmul64 calls)
//! - **Montgomery reduction** (2-pass fold from the bottom) for modular reduction — the key
//!   structural insight that makes POLYVAL's high-tap polynomial efficient to reduce

/// POLYVAL block size in bytes (128 bits).
pub(crate) const BLOCK_SIZE: usize = 16;

/// POLYVAL key size in bytes.
pub(crate) const KEY_SIZE: usize = 16;

/// POLYVAL reduction polynomial (without x^128):
/// x^127 + x^126 + x^121 + 1
#[cfg(test)]
const POLY: u128 = (1u128 << 127) | (1u128 << 126) | (1u128 << 121) | 1;

// ---------------------------------------------------------------------------
// x86_64 PCLMULQDQ backend
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod pclmul {
  use core::arch::x86_64::*;

  /// Combined 128×128 carryless multiply + Montgomery reduce using PCLMULQDQ.
  ///
  /// Returns the 128-bit field product in GF(2^128) mod POLYVAL's polynomial.
  ///
  /// # Safety
  /// Caller must ensure the CPU supports PCLMULQDQ and SSE2.
  #[target_feature(enable = "pclmulqdq,sse2")]
  pub(super) unsafe fn clmul128_reduce(a: u128, b: u128) -> u128 {
    let a_xmm = _mm_loadu_si128((&a as *const u128).cast());
    let b_xmm = _mm_loadu_si128((&b as *const u128).cast());

    // Schoolbook 128×128 → 256-bit product (4 PCLMULQDQ instructions).
    let lo = _mm_clmulepi64_si128(a_xmm, b_xmm, 0x00); // a_lo × b_lo
    let hi = _mm_clmulepi64_si128(a_xmm, b_xmm, 0x11); // a_hi × b_hi
    let m1 = _mm_clmulepi64_si128(a_xmm, b_xmm, 0x10); // a_hi × b_lo
    let m2 = _mm_clmulepi64_si128(a_xmm, b_xmm, 0x01); // a_lo × b_hi
    let mid = _mm_xor_si128(m1, m2);

    // Assemble 256-bit product [lo_128 : hi_128].
    let lo_128 = _mm_xor_si128(lo, _mm_slli_si128(mid, 8));
    let hi_128 = _mm_xor_si128(hi, _mm_srli_si128(mid, 8));

    // Montgomery reduction for x^128 + x^127 + x^126 + x^121 + 1.
    let result = mont_reduce_sse2(lo_128, hi_128);

    let mut out = 0u128;
    _mm_storeu_si128((&mut out as *mut u128).cast(), result);
    out
  }

  /// Montgomery reduction of a 256-bit product [lo:hi] modulo POLYVAL's polynomial.
  ///
  /// Equivalent to the portable `mont_reduce` but uses SSE2 lane-parallel shifts.
  #[inline(always)]
  unsafe fn mont_reduce_sse2(lo: __m128i, hi: __m128i) -> __m128i {
    // Phase 1: Compute left-shift contribution from both lo lanes.
    // For POLYVAL: shifts by 63, 62, 57 correspond to polynomial taps 127, 126, 121.
    let left = _mm_xor_si128(
      _mm_xor_si128(_mm_slli_epi64(lo, 63), _mm_slli_epi64(lo, 62)),
      _mm_slli_epi64(lo, 57),
    );

    // Fold v0's left shifts into v1 (cross-lane: lo→hi via byte shift).
    // slli_si128(left, 8) moves the low 64 bits to the high 64 bits.
    let lo_folded = _mm_xor_si128(lo, _mm_slli_si128(left, 8));
    // lo_folded = [v0 : v1'] where v1' = v1 ^ (v0<<63)^(v0<<62)^(v0<<57)

    // Right-shift + identity contribution from [v0 : v1'].
    // v0's contribution → v2 (Phase 1); v1's contribution → v3 (Phase 2).
    let right = _mm_xor_si128(
      _mm_xor_si128(lo_folded, _mm_srli_epi64(lo_folded, 1)),
      _mm_xor_si128(_mm_srli_epi64(lo_folded, 2), _mm_srli_epi64(lo_folded, 7)),
    );

    // Left-shift contribution from v1' (Phase 2 → v2).
    let left2 = _mm_xor_si128(
      _mm_xor_si128(_mm_slli_epi64(lo_folded, 63), _mm_slli_epi64(lo_folded, 62)),
      _mm_slli_epi64(lo_folded, 57),
    );

    // Final: hi XOR right XOR (left2's high lane shifted to low position).
    // srli_si128(left2, 8) moves the high 64 bits (v1'_left) to the low position.
    _mm_xor_si128(_mm_xor_si128(hi, right), _mm_srli_si128(left2, 8))
  }
}

// ---------------------------------------------------------------------------
// Combined multiply + reduce with hardware dispatch
// ---------------------------------------------------------------------------

/// Combined 128×128 carryless multiply + Montgomery reduce.
///
/// Dispatches to PCLMULQDQ on x86_64 when available, otherwise uses the
/// portable Pornin bmul64 + Karatsuba path.
pub(super) fn clmul128_reduce(a: u128, b: u128) -> u128 {
  #[cfg(target_arch = "x86_64")]
  {
    if crate::platform::caps().has(crate::platform::caps::x86::PCLMULQDQ) {
      // SAFETY: PCLMULQDQ availability verified via CPUID.
      return unsafe { pclmul::clmul128_reduce(a, b) };
    }
  }
  mont_reduce(clmul128(a, b))
}

/// POLYVAL accumulator state.
///
/// The hash key H is stored as-is (no domain conversion). The "dot"
/// product uses a 2-pass fold-from-bottom reduction that is structurally
/// a Montgomery reduction, but for POLYVAL's polynomial and LE
/// representation this directly produces the correct field product.
pub(crate) struct Polyval {
  /// Hash key H (raw field element, no conversion).
  h: u128,
  /// Running accumulator.
  acc: u128,
}

impl Polyval {
  /// Create a new POLYVAL instance with the given 128-bit key.
  #[inline]
  pub(crate) fn new(key: &[u8; KEY_SIZE]) -> Self {
    Self {
      h: u128::from_le_bytes(*key),
      acc: 0,
    }
  }

  /// Feed a single 16-byte block into the accumulator.
  ///
  /// Computes: acc = dot(acc XOR block, H)
  ///
  /// The "dot" product is the POLYVAL field multiplication, which is
  /// carryless polynomial multiplication followed by the fold-from-bottom
  /// Montgomery reduction. This naturally computes the correct GF(2^128)
  /// multiplication for POLYVAL's polynomial and LE bit representation.
  #[inline]
  pub(crate) fn update_block(&mut self, block: &[u8; BLOCK_SIZE]) {
    self.acc ^= u128::from_le_bytes(*block);
    self.acc = clmul128_reduce(self.acc, self.h);
  }

  /// Feed arbitrary-length data, padding the last block with zeros.
  pub(crate) fn update_padded(&mut self, data: &[u8]) {
    let mut offset = 0usize;
    while offset.strict_add(BLOCK_SIZE) <= data.len() {
      // Split avoids fallible conversion — the loop condition guarantees 16+ bytes remain.
      let (_, tail) = data.split_at(offset);
      let (head, _) = tail.split_at(BLOCK_SIZE);
      let block: &[u8; BLOCK_SIZE] = match head.try_into() {
        Ok(b) => b,
        Err(_) => unreachable!(),
      };
      self.update_block(block);
      offset = offset.strict_add(BLOCK_SIZE);
    }

    let remaining = data.len().strict_sub(offset);
    if remaining > 0 {
      let mut block = [0u8; BLOCK_SIZE];
      block[..remaining].copy_from_slice(&data[offset..]);
      self.update_block(&block);
    }
  }

  /// Finalize and return the 16-byte POLYVAL digest.
  #[inline]
  pub(crate) fn finalize(self) -> [u8; BLOCK_SIZE] {
    self.acc.to_le_bytes()
  }
}

impl Drop for Polyval {
  fn drop(&mut self) {
    // SAFETY: self.acc/self.h are valid, aligned, dereferenceable pointers to initialized memory.
    unsafe {
      core::ptr::write_volatile(&mut self.acc, 0);
      core::ptr::write_volatile(&mut self.h, 0);
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

// ---------------------------------------------------------------------------
// Pornin's bmul64: constant-time 64×64 → low-64 carryless multiplication
// ---------------------------------------------------------------------------

/// Carryless multiply of two 64-bit values, returning the LOW 64 bits.
///
/// Uses bit-strided decomposition: each operand is split into 4 interleaved
/// streams (every 4th bit), each with 16 data bits separated by 3 zero-bit
/// "holes." Regular integer multiplication on these strides is equivalent to
/// carryless multiplication because the holes absorb the carry spillover
/// (max accumulation per group is 16, which fits in 4+1 bits, and the carry
/// only reaches position 64 at the boundary, where it wraps and is discarded).
///
/// The 4 strides are recombined via cyclic convolution, giving the correct
/// carryless product at each bit position.
///
/// Cost: 16 integer multiplies + 20 bitwise ops.
/// Constant-time: no branches, no table lookups, no data-dependent memory access.
#[inline]
fn bmul64(x: u64, y: u64) -> u64 {
  let x0 = x & 0x1111_1111_1111_1111;
  let x1 = x & 0x2222_2222_2222_2222;
  let x2 = x & 0x4444_4444_4444_4444;
  let x3 = x & 0x8888_8888_8888_8888;

  let y0 = y & 0x1111_1111_1111_1111;
  let y1 = y & 0x2222_2222_2222_2222;
  let y2 = y & 0x4444_4444_4444_4444;
  let y3 = y & 0x8888_8888_8888_8888;

  // Cyclic convolution mod x^4 + 1 in GF(2).
  let mut z0 = (x0.wrapping_mul(y0)) ^ (x1.wrapping_mul(y3)) ^ (x2.wrapping_mul(y2)) ^ (x3.wrapping_mul(y1));
  let mut z1 = (x0.wrapping_mul(y1)) ^ (x1.wrapping_mul(y0)) ^ (x2.wrapping_mul(y3)) ^ (x3.wrapping_mul(y2));
  let mut z2 = (x0.wrapping_mul(y2)) ^ (x1.wrapping_mul(y1)) ^ (x2.wrapping_mul(y0)) ^ (x3.wrapping_mul(y3));
  let mut z3 = (x0.wrapping_mul(y3)) ^ (x1.wrapping_mul(y2)) ^ (x2.wrapping_mul(y1)) ^ (x3.wrapping_mul(y0));

  // Mask to keep only the data bits (discard carry residue in holes).
  z0 &= 0x1111_1111_1111_1111;
  z1 &= 0x2222_2222_2222_2222;
  z2 &= 0x4444_4444_4444_4444;
  z3 &= 0x8888_8888_8888_8888;

  z0 | z1 | z2 | z3
}

// ---------------------------------------------------------------------------
// 128×128 → 256-bit carryless multiplication (Karatsuba)
// ---------------------------------------------------------------------------

/// Multiply two 128-bit field elements, producing a 256-bit result
/// as four u64 words [v0, v1, v2, v3] (v0 = lowest).
///
/// Uses Karatsuba decomposition: 3 sub-products × 2 (normal + bit-reversed
/// for the high halves) = 6 bmul64 calls = 96 integer multiplies.
pub(super) fn clmul128(a: u128, b: u128) -> [u64; 4] {
  let a0 = a as u64;
  let a1 = (a >> 64) as u64;
  let b0 = b as u64;
  let b1 = (b >> 64) as u64;

  // Karatsuba middle term operands.
  let a2 = a0 ^ a1;
  let b2 = b0 ^ b1;

  // Bit-reversed operands (for computing the high 64 bits of each sub-product).
  let a0r = a0.reverse_bits();
  let a1r = a1.reverse_bits();
  let a2r = a2.reverse_bits();
  let b0r = b0.reverse_bits();
  let b1r = b1.reverse_bits();
  let b2r = b2.reverse_bits();

  // Low halves of sub-products.
  let z0 = bmul64(a0, b0);
  let z1 = bmul64(a1, b1);
  let mut z2 = bmul64(a2, b2);

  // High halves via bit-reversal identity:
  // high_64(A*B) = bmul64(rev(A), rev(B)).reverse_bits() >> 1
  let mut z0h = bmul64(a0r, b0r).reverse_bits() >> 1;
  let z1h = bmul64(a1r, b1r).reverse_bits() >> 1;
  let mut z2h = bmul64(a2r, b2r).reverse_bits() >> 1;

  // Karatsuba recombination: middle = z2 ^ z0 ^ z1.
  z2 ^= z0 ^ z1;
  z2h ^= z0h ^ z1h;

  // Assemble into [v0, v1, v2, v3].
  // v0 = z0_lo
  // v1 = z0_hi ^ z2_lo   (cross of low-product high and middle-product low)
  // v2 = z1_lo ^ z2_hi   (cross of high-product low and middle-product high)
  // v3 = z1_hi
  z0h ^= z2;

  [z0, z0h, z1 ^ z2h, z1h]
}

// ---------------------------------------------------------------------------
// Montgomery reduction for POLYVAL
// ---------------------------------------------------------------------------

/// Reduce a 256-bit product to 128 bits using Montgomery reduction.
///
/// Computes: [v3:v2:v1:v0] * x^{-128} mod p(x)
///
/// Two-pass fold from the bottom:
/// 1. Fold v0 into v1/v2 using the polynomial taps (121, 126, 127, 128).
/// 2. Fold v1 into v2/v3 using the same taps.
/// 3. Result is [v2:v3] (the upper 128 bits after folding).
///
/// This works because POLYVAL's polynomial x^128 + x^127 + x^126 + x^121 + 1
/// has its non-trivial taps near x^128. Folding from the bottom, each 64-bit
/// word's contribution stays within the adjacent 2 words — no cascading
/// overflow. Exactly 2 passes suffice.
#[inline]
pub(super) fn mont_reduce(v: [u64; 4]) -> u128 {
  let (v0, mut v1, mut v2, mut v3) = (v[0], v[1], v[2], v[3]);

  // Phase 1: fold v0 into v1, v2.
  // For each bit k in v0: add v0[k] * p(x) * x^k at positions 121+k..128+k.
  // The shifts 63/62/57 correspond to 64−1/64−2/64−7 (polynomial taps 127/126/121
  // relative to x^128, offset into the next 64-bit word).
  v2 ^= v0 ^ (v0 >> 1) ^ (v0 >> 2) ^ (v0 >> 7);
  v1 ^= (v0 << 63) ^ (v0 << 62) ^ (v0 << 57);

  // Phase 2: fold v1 into v2, v3 (identical structure).
  v3 ^= v1 ^ (v1 >> 1) ^ (v1 >> 2) ^ (v1 >> 7);
  v2 ^= (v1 << 63) ^ (v1 << 62) ^ (v1 << 57);

  // Result: [v2 : v3]
  (v2 as u128) | ((v3 as u128) << 64)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  /// RFC 8452 Appendix A: POLYVAL test vector.
  #[test]
  fn polyval_rfc8452_appendix_a() {
    let h = hex_to_16("25629347589242761d31f826ba4b757b");
    let x1 = hex_to_16("4f4f95668c83dfb6401762bb2d01a262");
    let x2 = hex_to_16("d1a24ddd2721d006bbe45f20d3c9f362");
    let expected = hex_to_16("f7a3b47b846119fae5b7866cf5e5b77e");

    let mut pv = Polyval::new(&h);
    pv.update_block(&x1);
    pv.update_block(&x2);
    let result = pv.finalize();

    assert_eq!(result, expected, "POLYVAL mismatch");
  }

  /// POLYVAL with empty input should return zero.
  #[test]
  fn polyval_empty() {
    let h = [0x42u8; 16];
    let pv = Polyval::new(&h);
    assert_eq!(pv.finalize(), [0u8; 16]);
  }

  /// POLYVAL with zero key should always return zero.
  #[test]
  fn polyval_zero_key() {
    let h = [0u8; 16];
    let x = [0xffu8; 16];
    let mut pv = Polyval::new(&h);
    pv.update_block(&x);
    assert_eq!(pv.finalize(), [0u8; 16]);
  }

  /// Verify that update_padded matches manual block-by-block.
  #[test]
  fn polyval_padded_matches_manual() {
    let h = hex_to_16("25629347589242761d31f826ba4b757b");
    let data = b"Hello, World! This is test data for POLYVAL padding.";

    // Manual: split into 16-byte blocks, pad last one.
    let mut manual = Polyval::new(&h);
    let mut offset = 0;
    while offset + 16 <= data.len() {
      let block: [u8; 16] = data[offset..offset + 16].try_into().unwrap();
      manual.update_block(&block);
      offset += 16;
    }
    if offset < data.len() {
      let mut block = [0u8; 16];
      block[..data.len() - offset].copy_from_slice(&data[offset..]);
      manual.update_block(&block);
    }
    let manual_result = manual.finalize();

    // Padded API.
    let mut padded = Polyval::new(&h);
    padded.update_padded(data);
    let padded_result = padded.finalize();

    assert_eq!(manual_result, padded_result);
  }

  /// bmul64 spot checks.
  #[test]
  fn bmul64_basic() {
    // 1 * x = x
    assert_eq!(bmul64(1, 0x42), 0x42);
    assert_eq!(bmul64(0x42, 1), 0x42);
    // 0 * x = 0
    assert_eq!(bmul64(0, 0xDEAD_BEEF), 0);
    // (x^7+...+1)^2 low bits: x^14+x^12+...+x^2+1 = 0x5555
    assert_eq!(bmul64(0xFF, 0xFF), 0x5555);
    // Commutativity
    assert_eq!(bmul64(0x1234, 0x5678), bmul64(0x5678, 0x1234));
  }

  /// Verify clmul128 produces correct 256-bit product for simple inputs.
  #[test]
  fn clmul128_identity() {
    // 1 * 1 = 1
    let v = clmul128(1, 1);
    assert_eq!(v, [1, 0, 0, 0]);

    // x * x = x^2 (= 4 in LE bit representation)
    let v = clmul128(2, 2);
    assert_eq!(v, [4, 0, 0, 0]);

    // (x^64) * (x^64) = x^128 → hi word
    let x64 = 1u128 << 64;
    let v = clmul128(x64, x64);
    assert_eq!(v, [0, 0, 1, 0]); // bit 128 = v2[0]
  }

  /// clmul128(1, x) should produce [x_lo, x_hi, 0, 0].
  #[test]
  fn clmul128_by_one() {
    let val: u128 = 0x7b75_4bba_26f8_311d_7642_9258_4793_6225;
    let v = clmul128(1, val);
    assert_eq!(v[0], val as u64, "v0 should be val_lo");
    assert_eq!(v[1], (val >> 64) as u64, "v1 should be val_hi");
    assert_eq!(v[2], 0, "v2 should be 0");
    assert_eq!(v[3], 0, "v3 should be 0");
  }

  /// mont_reduce of POLY (= x^128 mod p) treated as a 256-bit value
  /// in the low half gives x^128 * x^{-128} = 1.
  #[test]
  fn mont_reduce_of_poly() {
    let v = [POLY as u64, (POLY >> 64) as u64, 0u64, 0u64];
    let result = mont_reduce(v);
    assert_eq!(result, 1, "mont_reduce(POLY) should be 1");
  }

  fn hex_to_16(hex: &str) -> [u8; 16] {
    let mut out = [0u8; 16];
    let mut i = 0;
    while i < 16 {
      out[i] = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
      i = i.strict_add(1);
    }
    out
  }
}
