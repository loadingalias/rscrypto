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

use crate::backend::cache::OnceCache;

/// POLYVAL block size in bytes (128 bits).
#[cfg(feature = "aes-gcm-siv")]
pub(crate) const BLOCK_SIZE: usize = 16;

/// POLYVAL key size in bytes.
#[cfg(feature = "aes-gcm-siv")]
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
    // SAFETY: target_feature gate guarantees PCLMULQDQ + SSE2.
    unsafe {
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
  }

  /// Process 4 POLYVAL-domain blocks using PCLMULQDQ and one shared reduction.
  ///
  /// Computes `(acc ^ b0) * H^4 ^ b1 * H^3 ^ b2 * H^2 ^ b3 * H`.
  ///
  /// # Safety
  /// Caller must ensure PCLMULQDQ and SSE2 are available.
  #[cfg(any(feature = "aes-gcm", feature = "aes-gcm-siv"))]
  #[target_feature(enable = "pclmulqdq,sse2")]
  pub(super) unsafe fn aggregate_4blocks(acc: u128, h_powers_rev: &[u128; 4], blocks: &[u128; 4]) -> u128 {
    // SAFETY: target_feature gate guarantees PCLMULQDQ + SSE2; all loads read initialized
    // fixed-size stack/reference values.
    unsafe {
      let b0 = acc ^ blocks[0];
      let d0 = _mm_loadu_si128((&b0 as *const u128).cast());
      let d1 = _mm_loadu_si128((&blocks[1] as *const u128).cast());
      let d2 = _mm_loadu_si128((&blocks[2] as *const u128).cast());
      let d3 = _mm_loadu_si128((&blocks[3] as *const u128).cast());
      let h0 = _mm_loadu_si128((&h_powers_rev[0] as *const u128).cast());
      let h1 = _mm_loadu_si128((&h_powers_rev[1] as *const u128).cast());
      let h2 = _mm_loadu_si128((&h_powers_rev[2] as *const u128).cast());
      let h3 = _mm_loadu_si128((&h_powers_rev[3] as *const u128).cast());

      aggregate_xmms([d0, d1, d2, d3], [h0, h1, h2, h3])
    }
  }

  /// Process 4 big-endian GHASH blocks already held in XMM registers.
  ///
  /// The input registers contain ciphertext bytes in memory order. This helper reverses each lane
  /// into the POLYVAL-domain representation used by GHASH internally, folds `acc` into the first
  /// lane, and performs one 4-block aggregate.
  ///
  /// # Safety
  /// Caller must ensure PCLMULQDQ, SSE2, and SSSE3 are available.
  #[cfg(feature = "aes-gcm")]
  #[target_feature(enable = "pclmulqdq,sse2,ssse3")]
  pub(super) unsafe fn aggregate_4blocks_be_xmm(
    acc: u128,
    h_powers_rev: &[u128; 4],
    raw0: __m128i,
    raw1: __m128i,
    raw2: __m128i,
    raw3: __m128i,
  ) -> u128 {
    // SAFETY: target_feature gate guarantees PCLMULQDQ + SSE2 + SSSE3; all loads read initialized
    // fixed-size references, and PSHUFB only shuffles bytes within each register.
    unsafe {
      let reverse_bytes = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
      let acc_xmm = _mm_loadu_si128((&acc as *const u128).cast());
      let d0 = _mm_xor_si128(_mm_shuffle_epi8(raw0, reverse_bytes), acc_xmm);
      let d1 = _mm_shuffle_epi8(raw1, reverse_bytes);
      let d2 = _mm_shuffle_epi8(raw2, reverse_bytes);
      let d3 = _mm_shuffle_epi8(raw3, reverse_bytes);
      let h0 = _mm_loadu_si128((&h_powers_rev[0] as *const u128).cast());
      let h1 = _mm_loadu_si128((&h_powers_rev[1] as *const u128).cast());
      let h2 = _mm_loadu_si128((&h_powers_rev[2] as *const u128).cast());
      let h3 = _mm_loadu_si128((&h_powers_rev[3] as *const u128).cast());

      aggregate_xmms([d0, d1, d2, d3], [h0, h1, h2, h3])
    }
  }

  #[target_feature(enable = "pclmulqdq,sse2")]
  #[inline]
  unsafe fn aggregate_xmms(data: [__m128i; 4], h: [__m128i; 4]) -> u128 {
    // SAFETY: caller is inside a PCLMULQDQ + SSE2 target-feature chain.
    unsafe {
      let mut lo_sum = _mm_setzero_si128();
      let mut hi_sum = _mm_setzero_si128();

      macro_rules! fold_product {
        ($data:expr, $h:expr) => {{
          let lo = _mm_clmulepi64_si128($data, $h, 0x00);
          let hi = _mm_clmulepi64_si128($data, $h, 0x11);
          let m1 = _mm_clmulepi64_si128($data, $h, 0x10);
          let m2 = _mm_clmulepi64_si128($data, $h, 0x01);
          let mid = _mm_xor_si128(m1, m2);
          lo_sum = _mm_xor_si128(lo_sum, _mm_xor_si128(lo, _mm_slli_si128(mid, 8)));
          hi_sum = _mm_xor_si128(hi_sum, _mm_xor_si128(hi, _mm_srli_si128(mid, 8)));
        }};
      }

      fold_product!(data[0], h[0]);
      fold_product!(data[1], h[1]);
      fold_product!(data[2], h[2]);
      fold_product!(data[3], h[3]);

      let result = mont_reduce_sse2(lo_sum, hi_sum);
      let mut out = 0u128;
      _mm_storeu_si128((&mut out as *mut u128).cast(), result);
      out
    }
  }

  /// Montgomery reduction of a 256-bit product [lo:hi] modulo POLYVAL's polynomial.
  ///
  /// Equivalent to the portable `mont_reduce` but uses SSE2 lane-parallel shifts.
  #[inline]
  pub(super) unsafe fn mont_reduce_sse2(lo: __m128i, hi: __m128i) -> __m128i {
    // SAFETY: caller guarantees SSE2 availability via target_feature chain.
    unsafe {
      // Phase 1: Compute left-shift contribution from both lo lanes.
      // For POLYVAL: shifts by 63, 62, 57 correspond to polynomial taps 127, 126, 121.
      let left = _mm_xor_si128(
        _mm_xor_si128(_mm_slli_epi64(lo, 63), _mm_slli_epi64(lo, 62)),
        _mm_slli_epi64(lo, 57),
      );

      // Fold v0's left shifts into v1 (cross-lane: lo→hi via byte shift).
      // slli_si128(left, 8) moves the low 64 bits to the high 64 bits.
      let lo_folded = _mm_xor_si128(lo, _mm_slli_si128(left, 8));

      // Right-shift + identity contribution from [v0 : v1'].
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
      _mm_xor_si128(_mm_xor_si128(hi, right), _mm_srli_si128(left2, 8))
    }
  }
}

// ---------------------------------------------------------------------------
// aarch64 PMULL backend
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod pmull {
  use core::arch::aarch64::*;

  /// Core PMULL multiply + reduce — `#[inline(always)]` for guaranteed inlining.
  ///
  /// Uses `#[target_feature]` because aarch64 intrinsics require matching
  /// features on the immediate caller. Combined with `#[inline(always)]`,
  /// the function body is inlined into the caller — no function call boundary,
  /// no register spills.
  #[target_feature(enable = "neon", enable = "aes")]
  #[inline]
  pub(super) unsafe fn clmul128_reduce_core(a: u128, b: u128) -> u128 {
    // SAFETY: caller guarantees NEON + PMULL via target_feature chain.
    unsafe {
      let a_lo = a as u64;
      let a_hi = (a >> 64) as u64;
      let b_lo = b as u64;
      let b_hi = (b >> 64) as u64;

      // Karatsuba 128×128 → 256-bit product (3 PMULL instructions).
      let ll = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
      let hh = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));
      let mm = vreinterpretq_u64_p128(vmull_p64(a_lo ^ a_hi, b_lo ^ b_hi));
      let mid = veorq_u64(veorq_u64(mm, ll), hh);

      // Assemble 256-bit product [lo_128 : hi_128].
      let zero = vdupq_n_u64(0);
      let lo = veorq_u64(ll, vextq_u64(zero, mid, 1)); // ll XOR (mid << 64)
      let hi = veorq_u64(hh, vextq_u64(mid, zero, 1)); // hh XOR (mid >> 64)

      // Montgomery reduction for POLYVAL's polynomial.
      let result = mont_reduce_neon(lo, hi);

      // Extract result as u128 (LE).
      let r_lo = vgetq_lane_u64(result, 0) as u128;
      let r_hi = vgetq_lane_u64(result, 1) as u128;
      r_lo | (r_hi << 64)
    }
  }

  /// Combined 128×128 carryless multiply + Montgomery reduce using PMULL.
  ///
  /// Uses 4 `vmull_p64` instructions for the schoolbook product, then
  /// lane-parallel NEON shifts for Montgomery reduction — the same
  /// 2-pass fold-from-bottom structure as the SSE2 and portable paths.
  ///
  /// # Safety
  /// Caller must ensure the CPU supports PMULL (gated via `target_feature = "aes"`
  /// because ARM bundles PMULL in the crypto extension alongside AES).
  #[target_feature(enable = "neon", enable = "aes")]
  pub(super) unsafe fn clmul128_reduce(a: u128, b: u128) -> u128 {
    // SAFETY: target_feature gate guarantees NEON + PMULL.
    unsafe { clmul128_reduce_core(a, b) }
  }

  /// Montgomery reduction of [lo:hi] modulo POLYVAL's polynomial.
  ///
  /// Polynomial: x^128 + x^127 + x^126 + x^121 + 1
  /// Shifts: 63, 62, 57 (complement of taps relative to 64-bit boundary)
  ///
  /// Uses NEON lane-parallel shifts (`vshlq_n_u64`, `vshrq_n_u64`) and
  /// `vextq_u64` for cross-lane propagation — structurally identical to
  /// the SSE2 `mont_reduce_sse2` path.
  #[inline]
  unsafe fn mont_reduce_neon(lo: uint64x2_t, hi: uint64x2_t) -> uint64x2_t {
    // SAFETY: caller guarantees NEON availability via target_feature chain.
    unsafe {
      let zero = vdupq_n_u64(0);

      // Phase 1: left-shift contribution from both lo lanes.
      let left = veorq_u64(veorq_u64(vshlq_n_u64(lo, 63), vshlq_n_u64(lo, 62)), vshlq_n_u64(lo, 57));

      // Fold lo[0]'s left shifts into lo[1] (cross-lane: lo → hi).
      let lo_folded = veorq_u64(lo, vextq_u64(zero, left, 1));

      // Right-shift + identity contribution.
      let right = veorq_u64(
        veorq_u64(lo_folded, vshrq_n_u64(lo_folded, 1)),
        veorq_u64(vshrq_n_u64(lo_folded, 2), vshrq_n_u64(lo_folded, 7)),
      );

      // Phase 2: left-shift from lo_folded.
      let left2 = veorq_u64(
        veorq_u64(vshlq_n_u64(lo_folded, 63), vshlq_n_u64(lo_folded, 62)),
        vshlq_n_u64(lo_folded, 57),
      );

      // Final: hi XOR right XOR (left2 >> 64).
      veorq_u64(veorq_u64(hi, right), vextq_u64(left2, zero, 1))
    }
  }

  /// 4-block wide Karatsuba multiply + single Montgomery reduction.
  ///
  /// Computes `(acc ^ b0) * H^4 ^ b1 * H^3 ^ b2 * H^2 ^ b3 * H` using
  /// 12 independent `vmull_p64` instructions that the OOO core on Neoverse
  /// V1/V2 (2 crypto pipes) can schedule freely, then a single reduction.
  #[cfg(any(feature = "aes-gcm", feature = "aes-gcm-siv"))]
  #[target_feature(enable = "neon", enable = "aes")]
  #[inline]
  pub(super) unsafe fn aggregate_4blocks(acc: u128, h_powers_rev: &[u128; 4], blocks: &[u128; 4]) -> u128 {
    // SAFETY: target_feature gate guarantees NEON + PMULL.
    unsafe {
      let b0 = acc ^ blocks[0];
      let b1 = blocks[1];
      let b2 = blocks[2];
      let b3 = blocks[3];

      // 12 vmull_p64: 4 blocks × 3 Karatsuba products.
      let b0_lo = b0 as u64;
      let b0_hi = (b0 >> 64) as u64;
      let h0_lo = h_powers_rev[0] as u64;
      let h0_hi = (h_powers_rev[0] >> 64) as u64;
      let ll0 = vreinterpretq_u64_p128(vmull_p64(b0_lo, h0_lo));
      let hh0 = vreinterpretq_u64_p128(vmull_p64(b0_hi, h0_hi));
      let mm0 = vreinterpretq_u64_p128(vmull_p64(b0_lo ^ b0_hi, h0_lo ^ h0_hi));

      let b1_lo = b1 as u64;
      let b1_hi = (b1 >> 64) as u64;
      let h1_lo = h_powers_rev[1] as u64;
      let h1_hi = (h_powers_rev[1] >> 64) as u64;
      let ll1 = vreinterpretq_u64_p128(vmull_p64(b1_lo, h1_lo));
      let hh1 = vreinterpretq_u64_p128(vmull_p64(b1_hi, h1_hi));
      let mm1 = vreinterpretq_u64_p128(vmull_p64(b1_lo ^ b1_hi, h1_lo ^ h1_hi));

      let b2_lo = b2 as u64;
      let b2_hi = (b2 >> 64) as u64;
      let h2_lo = h_powers_rev[2] as u64;
      let h2_hi = (h_powers_rev[2] >> 64) as u64;
      let ll2 = vreinterpretq_u64_p128(vmull_p64(b2_lo, h2_lo));
      let hh2 = vreinterpretq_u64_p128(vmull_p64(b2_hi, h2_hi));
      let mm2 = vreinterpretq_u64_p128(vmull_p64(b2_lo ^ b2_hi, h2_lo ^ h2_hi));

      let b3_lo = b3 as u64;
      let b3_hi = (b3 >> 64) as u64;
      let h3_lo = h_powers_rev[3] as u64;
      let h3_hi = (h_powers_rev[3] >> 64) as u64;
      let ll3 = vreinterpretq_u64_p128(vmull_p64(b3_lo, h3_lo));
      let hh3 = vreinterpretq_u64_p128(vmull_p64(b3_hi, h3_hi));
      let mm3 = vreinterpretq_u64_p128(vmull_p64(b3_lo ^ b3_hi, h3_lo ^ h3_hi));

      // XOR the 4 products.
      let ll = veorq_u64(veorq_u64(ll0, ll1), veorq_u64(ll2, ll3));
      let hh = veorq_u64(veorq_u64(hh0, hh1), veorq_u64(hh2, hh3));
      let mm = veorq_u64(veorq_u64(mm0, mm1), veorq_u64(mm2, mm3));

      // Assemble + reduce (same as single-block path).
      let mid = veorq_u64(veorq_u64(mm, ll), hh);
      let zero = vdupq_n_u64(0);
      let lo = veorq_u64(ll, vextq_u64(zero, mid, 1));
      let hi = veorq_u64(hh, vextq_u64(mid, zero, 1));
      let result = mont_reduce_neon(lo, hi);

      let r_lo = vgetq_lane_u64(result, 0) as u128;
      let r_hi = vgetq_lane_u64(result, 1) as u128;
      r_lo | (r_hi << 64)
    }
  }

  /// 8-block GHASH aggregate from big-endian ciphertext lanes already held in NEON registers.
  #[cfg(feature = "aes-gcm")]
  #[target_feature(enable = "neon", enable = "aes")]
  #[inline]
  pub(super) unsafe fn aggregate_8blocks_be_lanes(
    acc: u128,
    h_powers_rev: &[u128; 8],
    blocks: &[uint8x16_t; 8],
  ) -> u128 {
    // SAFETY: target_feature gate guarantees NEON + PMULL. The byte-reversal maps memory-order
    // GHASH lanes to the little-endian POLYVAL-domain lane representation.
    unsafe {
      #[inline(always)]
      unsafe fn load_power(power: *const u128) -> uint64x2_t {
        // SAFETY: caller passes a pointer into `h_powers_rev`; `vld1q_u64`
        // accepts arbitrary alignment and reads exactly one initialized u128.
        unsafe { vld1q_u64(power.cast::<u64>()) }
      }

      #[inline(always)]
      unsafe fn u128_to_lanes(x: u128) -> uint64x2_t {
        // SAFETY: caller is inside a NEON target scope. `vcreate_u64`
        // initializes one 64-bit lane and `vcombine_u64` builds the pair.
        unsafe { vcombine_u64(vcreate_u64(x as u64), vcreate_u64((x >> 64) as u64)) }
      }

      let mut ll = vdupq_n_u64(0);
      let mut hh = vdupq_n_u64(0);
      let mut mm = vdupq_n_u64(0);
      let acc_lanes = u128_to_lanes(acc);

      macro_rules! fold_product_with_acc {
        ($idx:expr, $acc_lanes:expr) => {{
          let rev64 = vrev64q_u8(blocks[$idx]);
          let rev = vextq_u8(rev64, rev64, 8);
          let lanes = veorq_u64(vreinterpretq_u64_u8(rev), $acc_lanes);
          let h = load_power(h_powers_rev.as_ptr().add($idx));

          let b_poly = vreinterpretq_p64_u64(lanes);
          let h_poly = vreinterpretq_p64_u64(h);
          ll = veorq_u64(
            ll,
            vreinterpretq_u64_p128(vmull_p64(vgetq_lane_p64(b_poly, 0), vgetq_lane_p64(h_poly, 0))),
          );
          hh = veorq_u64(hh, vreinterpretq_u64_p128(vmull_high_p64(b_poly, h_poly)));

          let b_mid = veorq_u64(lanes, vextq_u64(lanes, lanes, 1));
          let h_mid = veorq_u64(h, vextq_u64(h, h, 1));
          let b_mid_poly = vreinterpretq_p64_u64(b_mid);
          let h_mid_poly = vreinterpretq_p64_u64(h_mid);
          mm = veorq_u64(
            mm,
            vreinterpretq_u64_p128(vmull_p64(
              vgetq_lane_p64(b_mid_poly, 0),
              vgetq_lane_p64(h_mid_poly, 0),
            )),
          );
        }};
      }

      fold_product_with_acc!(0, acc_lanes);
      fold_product_with_acc!(1, vdupq_n_u64(0));
      fold_product_with_acc!(2, vdupq_n_u64(0));
      fold_product_with_acc!(3, vdupq_n_u64(0));
      fold_product_with_acc!(4, vdupq_n_u64(0));
      fold_product_with_acc!(5, vdupq_n_u64(0));
      fold_product_with_acc!(6, vdupq_n_u64(0));
      fold_product_with_acc!(7, vdupq_n_u64(0));

      let mid = veorq_u64(veorq_u64(mm, ll), hh);
      let zero = vdupq_n_u64(0);
      let lo = veorq_u64(ll, vextq_u64(zero, mid, 1));
      let hi = veorq_u64(hh, vextq_u64(mid, zero, 1));
      let result = mont_reduce_neon(lo, hi);

      let r_lo = vgetq_lane_u64(result, 0) as u128;
      let r_hi = vgetq_lane_u64(result, 1) as u128;
      r_lo | (r_hi << 64)
    }
  }
}

// ---------------------------------------------------------------------------
// s390x VGFM backend (Galois field multiply)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "s390x")]
#[allow(unsafe_code)]
mod s390x_vgfm {
  use core::{arch::asm, simd::i64x2};

  /// 64×64→128 carryless multiply using VGFM mode 3 (doubleword).
  ///
  /// Places operands in the low lane (element 1) with the high lane zeroed,
  /// so VGFM computes `0*0 XOR a*b = a*b`.
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn mul64(a: u64, b: u64) -> i64x2 {
    let va = i64x2::from_array([0, a as i64]);
    let vb = i64x2::from_array([0, b as i64]);
    // SAFETY: Caller guarantees z/Vector facility is available.
    unsafe {
      let out: i64x2;
      asm!(
        "vgfm {out}, {a}, {b}, 3",
        out = lateout(vreg) out,
        a = in(vreg) va,
        b = in(vreg) vb,
        options(nomem, nostack, pure),
      );
      out
    }
  }

  /// Per-lane left shift of both 64-bit elements.
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn veslg<const N: u32>(a: i64x2) -> i64x2 {
    // SAFETY: Caller guarantees z/Vector facility is available via target_feature gate.
    unsafe {
      let out: i64x2;
      asm!(
        "veslg {out}, {a}, {n}",
        out = lateout(vreg) out,
        a = in(vreg) a,
        n = const N,
        options(nomem, nostack, pure),
      );
      out
    }
  }

  /// Per-lane logical right shift of both 64-bit elements.
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn vesrlg<const N: u32>(a: i64x2) -> i64x2 {
    // SAFETY: Caller guarantees z/Vector facility is available via target_feature gate.
    unsafe {
      let out: i64x2;
      asm!(
        "vesrlg {out}, {a}, {n}",
        out = lateout(vreg) out,
        a = in(vreg) a,
        n = const N,
        options(nomem, nostack, pure),
      );
      out
    }
  }

  /// Cross-lane byte shift: `high128((a || b) << N*8 bits)`.
  ///
  /// - `vsldb(v, zero, 8)`: moves low lane to high, zeros low = `v << 64`
  /// - `vsldb(zero, v, 8)`: moves high lane to low, zeros high = `v >> 64`
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn vsldb<const N: u32>(a: i64x2, b: i64x2) -> i64x2 {
    // SAFETY: Caller guarantees z/Vector facility is available via target_feature gate.
    unsafe {
      let out: i64x2;
      asm!(
        "vsldb {out}, {a}, {b}, {n}",
        out = lateout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        n = const N,
        options(nomem, nostack, pure),
      );
      out
    }
  }

  /// Montgomery reduction of a 256-bit product [lo:hi] modulo POLYVAL's polynomial.
  ///
  /// Structurally identical to `mont_reduce_sse2` / `mont_reduce_neon`,
  /// using s390x VESLG/VESRLG for per-lane shifts and VSLDB for cross-lane
  /// byte shifts.
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn mont_reduce(lo: i64x2, hi: i64x2) -> i64x2 {
    // SAFETY: target_feature gate guarantees z/Vector availability.
    unsafe {
      let zero = i64x2::from_array([0, 0]);

      // Phase 1: left-shift contribution from both lo lanes.
      // Shifts 63, 62, 57 correspond to polynomial taps 127, 126, 121.
      let left = veslg::<63>(lo) ^ veslg::<62>(lo) ^ veslg::<57>(lo);

      // Fold: move low lane's left shifts to high lane (lo → hi propagation).
      // On s390x (BE): element 1 (low) → element 0 (high).
      let lo_folded = lo ^ vsldb::<8>(left, zero);

      // Right-shift + identity contribution.
      let right = lo_folded ^ vesrlg::<1>(lo_folded) ^ vesrlg::<2>(lo_folded) ^ vesrlg::<7>(lo_folded);

      // Phase 2: left-shift from lo_folded.
      let left2 = veslg::<63>(lo_folded) ^ veslg::<62>(lo_folded) ^ veslg::<57>(lo_folded);

      // Final: hi XOR right XOR (high lane of left2 moved to low position).
      hi ^ right ^ vsldb::<8>(zero, left2)
    }
  }

  /// Core VGFM multiply + reduce — `#[inline(always)]` for guaranteed inlining.
  #[target_feature(enable = "vector")]
  #[inline]
  pub(super) unsafe fn clmul128_reduce_core(a: u128, b: u128) -> u128 {
    // SAFETY: caller guarantees z/Vector availability via target_feature chain.
    unsafe {
      let a_lo = a as u64;
      let a_hi = (a >> 64) as u64;
      let b_lo = b as u64;
      let b_hi = (b >> 64) as u64;
      let zero = i64x2::from_array([0, 0]);

      // Karatsuba: 3 VGFM multiplies.
      let v0 = mul64(a_lo, b_lo);
      let v1 = mul64(a_hi, b_hi);
      let v2 = mul64(a_lo ^ a_hi, b_lo ^ b_hi);
      let mid = v2 ^ v0 ^ v1;

      // Assemble 256-bit product [lo_128 : hi_128].
      let lo_128 = v0 ^ vsldb::<8>(mid, zero);
      let hi_128 = v1 ^ vsldb::<8>(zero, mid);

      let result = mont_reduce(lo_128, hi_128);
      let arr = result.to_array();
      ((arr[0] as u64 as u128) << 64) | (arr[1] as u64 as u128)
    }
  }

  /// Combined 128×128 carryless multiply + Montgomery reduce using VGFM.
  ///
  /// Karatsuba decomposition with 3 VGFM multiplies (vs 4 schoolbook).
  ///
  /// # Safety
  /// Caller must ensure the z/Vector facility is available.
  #[target_feature(enable = "vector")]
  pub(super) unsafe fn clmul128_reduce(a: u128, b: u128) -> u128 {
    // SAFETY: target_feature gate guarantees z/Vector availability.
    unsafe { clmul128_reduce_core(a, b) }
  }

  /// 4-block wide Karatsuba multiply + single Montgomery reduction.
  ///
  /// 12 independent VGFM (4 × 3 Karatsuba), then one vector
  /// Montgomery reduction.
  #[cfg(any(feature = "aes-gcm", feature = "aes-gcm-siv"))]
  #[target_feature(enable = "vector")]
  #[inline]
  pub(super) unsafe fn aggregate_4blocks(acc: u128, h_powers_rev: &[u128; 4], blocks: &[u128; 4]) -> u128 {
    // SAFETY: target_feature gate guarantees z/Vector availability.
    unsafe {
      let b0 = acc ^ blocks[0];
      let b1 = blocks[1];
      let b2 = blocks[2];
      let b3 = blocks[3];
      let zero = i64x2::from_array([0, 0]);

      // 12 VGFM: 4 blocks × 3 Karatsuba multiplies.
      let v0_0 = mul64(b0 as u64, h_powers_rev[0] as u64);
      let v1_0 = mul64((b0 >> 64) as u64, (h_powers_rev[0] >> 64) as u64);
      let v2_0 = mul64(
        b0 as u64 ^ (b0 >> 64) as u64,
        h_powers_rev[0] as u64 ^ (h_powers_rev[0] >> 64) as u64,
      );

      let v0_1 = mul64(b1 as u64, h_powers_rev[1] as u64);
      let v1_1 = mul64((b1 >> 64) as u64, (h_powers_rev[1] >> 64) as u64);
      let v2_1 = mul64(
        b1 as u64 ^ (b1 >> 64) as u64,
        h_powers_rev[1] as u64 ^ (h_powers_rev[1] >> 64) as u64,
      );

      let v0_2 = mul64(b2 as u64, h_powers_rev[2] as u64);
      let v1_2 = mul64((b2 >> 64) as u64, (h_powers_rev[2] >> 64) as u64);
      let v2_2 = mul64(
        b2 as u64 ^ (b2 >> 64) as u64,
        h_powers_rev[2] as u64 ^ (h_powers_rev[2] >> 64) as u64,
      );

      let v0_3 = mul64(b3 as u64, h_powers_rev[3] as u64);
      let v1_3 = mul64((b3 >> 64) as u64, (h_powers_rev[3] >> 64) as u64);
      let v2_3 = mul64(
        b3 as u64 ^ (b3 >> 64) as u64,
        h_powers_rev[3] as u64 ^ (h_powers_rev[3] >> 64) as u64,
      );

      // XOR all 4 Karatsuba intermediates (i64x2 vector XOR).
      let v0 = v0_0 ^ v0_1 ^ v0_2 ^ v0_3;
      let v1 = v1_0 ^ v1_1 ^ v1_2 ^ v1_3;
      let v2 = v2_0 ^ v2_1 ^ v2_2 ^ v2_3;
      let mid = v2 ^ v0 ^ v1;

      let lo_128 = v0 ^ vsldb::<8>(mid, zero);
      let hi_128 = v1 ^ vsldb::<8>(zero, mid);

      let result = mont_reduce(lo_128, hi_128);
      let arr = result.to_array();
      ((arr[0] as u64 as u128) << 64) | (arr[1] as u64 as u128)
    }
  }
}

// ---------------------------------------------------------------------------
// powerpc64 VPMSUMD backend (polynomial multiply-sum doubleword)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "powerpc64")]
#[allow(unsafe_code)]
mod ppc_vpmsum {
  use core::{arch::asm, simd::i64x2};

  /// 64×64→128 carryless multiply using vpmsumd.
  ///
  /// Match the existing POWER checksum kernels' lane convention:
  /// lane 0 carries the low 64 bits, lane 1 the high 64 bits.
  #[inline]
  #[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
  unsafe fn mul64(a: u64, b: u64) -> (u64, u64) {
    let va = i64x2::from_array([a as i64, 0]);
    let vb = i64x2::from_array([b as i64, 0]);
    // SAFETY: Caller guarantees POWER8 crypto availability.
    unsafe {
      let out: i64x2;
      asm!(
        "vpmsumd {out}, {a}, {b}",
        out = lateout(vreg) out,
        a = in(vreg) va,
        b = in(vreg) vb,
        options(nomem, nostack, pure),
      );
      let [lo, hi] = out.to_array();
      (lo as u64, hi as u64)
    }
  }

  /// Core VPMSUMD multiply + reduce — `#[inline(always)]` for guaranteed inlining.
  #[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
  #[inline]
  pub(super) unsafe fn clmul128_reduce_core(a: u128, b: u128) -> u128 {
    // SAFETY: caller guarantees POWER8 crypto availability via target_feature chain.
    unsafe {
      let a_lo = a as u64;
      let a_hi = (a >> 64) as u64;
      let b_lo = b as u64;
      let b_hi = (b >> 64) as u64;

      // Karatsuba: 3 vpmsumd multiplies.
      let (v0_lo, v0_hi) = mul64(a_lo, b_lo);
      let (v1_lo, v1_hi) = mul64(a_hi, b_hi);
      let (v2_lo, v2_hi) = mul64(a_lo ^ a_hi, b_lo ^ b_hi);

      let mid_lo = v2_lo ^ v0_lo ^ v1_lo;
      let mid_hi = v2_hi ^ v0_hi ^ v1_hi;

      super::mont_reduce([v0_lo, v0_hi ^ mid_lo, v1_lo ^ mid_hi, v1_hi])
    }
  }

  /// Combined 128×128 carryless multiply + Montgomery reduce using vpmsumd.
  ///
  /// Karatsuba decomposition with 3 `vpmsumd` multiplies, then the same
  /// scalar Montgomery reduction as the portable path.
  ///
  /// # Safety
  /// Caller must ensure POWER8 crypto instructions are available.
  #[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
  pub(super) unsafe fn clmul128_reduce(a: u128, b: u128) -> u128 {
    // SAFETY: target_feature gate guarantees POWER8 crypto availability.
    unsafe { clmul128_reduce_core(a, b) }
  }

  /// 4-block wide Karatsuba multiply + single Montgomery reduction.
  ///
  /// 12 independent `vpmsumd` (4 × 3 Karatsuba), then one scalar
  /// Montgomery reduction.
  #[cfg(any(feature = "aes-gcm", feature = "aes-gcm-siv"))]
  #[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
  #[inline]
  pub(super) unsafe fn aggregate_4blocks(acc: u128, h_powers_rev: &[u128; 4], blocks: &[u128; 4]) -> u128 {
    // SAFETY: target_feature gate guarantees POWER8 crypto availability.
    unsafe {
      let b0 = acc ^ blocks[0];
      let b1 = blocks[1];
      let b2 = blocks[2];
      let b3 = blocks[3];

      // 12 vpmsumd: 4 blocks × 3 Karatsuba multiplies.
      let (z0_0l, z0_0h) = mul64(b0 as u64, h_powers_rev[0] as u64);
      let (z1_0l, z1_0h) = mul64((b0 >> 64) as u64, (h_powers_rev[0] >> 64) as u64);
      let (z2_0l, z2_0h) = mul64(
        b0 as u64 ^ (b0 >> 64) as u64,
        h_powers_rev[0] as u64 ^ (h_powers_rev[0] >> 64) as u64,
      );

      let (z0_1l, z0_1h) = mul64(b1 as u64, h_powers_rev[1] as u64);
      let (z1_1l, z1_1h) = mul64((b1 >> 64) as u64, (h_powers_rev[1] >> 64) as u64);
      let (z2_1l, z2_1h) = mul64(
        b1 as u64 ^ (b1 >> 64) as u64,
        h_powers_rev[1] as u64 ^ (h_powers_rev[1] >> 64) as u64,
      );

      let (z0_2l, z0_2h) = mul64(b2 as u64, h_powers_rev[2] as u64);
      let (z1_2l, z1_2h) = mul64((b2 >> 64) as u64, (h_powers_rev[2] >> 64) as u64);
      let (z2_2l, z2_2h) = mul64(
        b2 as u64 ^ (b2 >> 64) as u64,
        h_powers_rev[2] as u64 ^ (h_powers_rev[2] >> 64) as u64,
      );

      let (z0_3l, z0_3h) = mul64(b3 as u64, h_powers_rev[3] as u64);
      let (z1_3l, z1_3h) = mul64((b3 >> 64) as u64, (h_powers_rev[3] >> 64) as u64);
      let (z2_3l, z2_3h) = mul64(
        b3 as u64 ^ (b3 >> 64) as u64,
        h_powers_rev[3] as u64 ^ (h_powers_rev[3] >> 64) as u64,
      );

      // XOR all 4 Karatsuba intermediates.
      let z0_lo = z0_0l ^ z0_1l ^ z0_2l ^ z0_3l;
      let z0_hi = z0_0h ^ z0_1h ^ z0_2h ^ z0_3h;
      let z1_lo = z1_0l ^ z1_1l ^ z1_2l ^ z1_3l;
      let z1_hi = z1_0h ^ z1_1h ^ z1_2h ^ z1_3h;
      let z2_lo = z2_0l ^ z2_1l ^ z2_2l ^ z2_3l;
      let z2_hi = z2_0h ^ z2_1h ^ z2_2h ^ z2_3h;

      let mid_lo = z2_lo ^ z0_lo ^ z1_lo;
      let mid_hi = z2_hi ^ z0_hi ^ z1_hi;

      super::mont_reduce([z0_lo, z0_hi ^ mid_lo, z1_lo ^ mid_hi, z1_hi])
    }
  }
}

// ---------------------------------------------------------------------------
// riscv64 Zvbc backend (vector carryless multiply)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "riscv64")]
#[allow(unsafe_code)]
mod rv_clmul {
  use core::arch::asm;

  /// 64×64→128 carryless multiply using Zvbc (vclmul.vx + vclmulh.vx).
  ///
  /// Uses the clobber-only vreg workaround: data shuttled through GPRs
  /// and memory; vector registers referenced by explicit names.
  #[inline]
  #[target_feature(enable = "v", enable = "zvbc")]
  unsafe fn mul64(a: u64, b: u64) -> (u64, u64) {
    // SAFETY: Caller guarantees Zvbc availability.
    unsafe {
      let lo: u64;
      let hi: u64;
      asm!(
        "vsetivli zero, 1, e64, m1, ta, ma",
        "vmv.v.x v0, {a}",
        "vclmul.vx v1, v0, {b}",
        "vclmulh.vx v2, v0, {b}",
        "vmv.x.s {lo}, v1",
        "vmv.x.s {hi}, v2",
        a = in(reg) a,
        b = in(reg) b,
        lo = lateout(reg) lo,
        hi = lateout(reg) hi,
        out("v0") _,
        out("v1") _,
        out("v2") _,
        options(nostack),
      );
      (lo, hi)
    }
  }

  /// Combined 128×128 carryless multiply + Montgomery reduce using Zvbc.
  ///
  /// Karatsuba decomposition with 3 vclmul/vclmulh pairs, then scalar
  /// Montgomery reduction (reuses the portable `mont_reduce`).
  ///
  /// # Safety
  /// Caller must ensure Zvbc vector extension is available.
  #[target_feature(enable = "v", enable = "zvbc")]
  pub(super) unsafe fn clmul128_reduce(a: u128, b: u128) -> u128 {
    // SAFETY: target_feature gate guarantees Zvbc availability.
    unsafe {
      let a_lo = a as u64;
      let a_hi = (a >> 64) as u64;
      let b_lo = b as u64;
      let b_hi = (b >> 64) as u64;

      // Karatsuba: 3 multiplies.
      let (v0_lo, v0_hi) = mul64(a_lo, b_lo);
      let (v1_lo, v1_hi) = mul64(a_hi, b_hi);
      let (v2_lo, v2_hi) = mul64(a_lo ^ a_hi, b_lo ^ b_hi);

      // Cross term.
      let mid_lo = v2_lo ^ v0_lo ^ v1_lo;
      let mid_hi = v2_hi ^ v0_hi ^ v1_hi;

      // Assemble 256-bit product as [w0, w1, w2, w3] (low to high).
      let w0 = v0_lo;
      let w1 = v0_hi ^ mid_lo;
      let w2 = v1_lo ^ mid_hi;
      let w3 = v1_hi;

      // Reuse portable Montgomery reduction.
      super::mont_reduce([w0, w1, w2, w3])
    }
  }
}

// ---------------------------------------------------------------------------
// riscv64 Zbc backend (scalar carryless multiply)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "riscv64")]
#[allow(unsafe_code)]
mod rv_scalar_clmul {
  use core::arch::asm;

  /// 64×64→128 carryless multiply using scalar Zbc (clmul + clmulh).
  ///
  /// Identical encoding to Zbkc — dispatch checks either cap at runtime.
  #[inline]
  #[target_feature(enable = "zbc")]
  unsafe fn mul64(a: u64, b: u64) -> (u64, u64) {
    // SAFETY: Caller guarantees Zbc availability; clmul/clmulh are pure
    // register-to-register instructions with no memory side effects.
    unsafe {
      let lo: u64;
      let hi: u64;
      asm!(
        "clmul {lo}, {a}, {b}",
        "clmulh {hi}, {a}, {b}",
        a = in(reg) a,
        b = in(reg) b,
        lo = lateout(reg) lo,
        hi = lateout(reg) hi,
        options(nomem, nostack, pure),
      );
      (lo, hi)
    }
  }

  /// Combined 128×128 carryless multiply + Montgomery reduce using scalar Zbc.
  ///
  /// Karatsuba decomposition with 3 × clmul/clmulh pairs (6 instructions),
  /// then portable Montgomery reduction. ~100x faster than Pornin bmul64 on
  /// hardware with Zbc (e.g. SpacemiT K1).
  ///
  /// # Safety
  /// Caller must ensure Zbc or Zbkc scalar extension is available.
  #[target_feature(enable = "zbc")]
  pub(super) unsafe fn clmul128_reduce(a: u128, b: u128) -> u128 {
    // SAFETY: target_feature gate guarantees Zbc availability. mul64 calls
    // are safe within this target_feature scope.
    unsafe {
      let a_lo = a as u64;
      let a_hi = (a >> 64) as u64;
      let b_lo = b as u64;
      let b_hi = (b >> 64) as u64;

      // Karatsuba: 3 multiplies instead of 4.
      let (v0_lo, v0_hi) = mul64(a_lo, b_lo);
      let (v1_lo, v1_hi) = mul64(a_hi, b_hi);
      let (v2_lo, v2_hi) = mul64(a_lo ^ a_hi, b_lo ^ b_hi);

      // Cross term.
      let mid_lo = v2_lo ^ v0_lo ^ v1_lo;
      let mid_hi = v2_hi ^ v0_hi ^ v1_hi;

      super::mont_reduce([v0_lo, v0_hi ^ mid_lo, v1_lo ^ mid_hi, v1_hi])
    }
  }
}

// ---------------------------------------------------------------------------
// x86_64 VPCLMULQDQ wide backend (4-block aggregate)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod vpclmul {
  use core::arch::x86_64::*;

  /// Reduce four already packed POLYVAL-domain lanes into one accumulator.
  ///
  /// # Safety
  /// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
  /// VPCLMULQDQ + PCLMULQDQ + SSE2.
  #[cfg(any(feature = "aes-gcm", feature = "aes-gcm-siv"))]
  #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
  #[inline]
  unsafe fn aggregate_lanes(data: __m512i, h_powers_rev: &[u128; 4]) -> u128 {
    // SAFETY: x86 VPCLMUL aggregation because:
    // 1. This function's caller guarantees all required target features.
    // 2. `data` contains four initialized POLYVAL-domain lanes and `h_powers_rev` has four powers.
    unsafe {
      // Load H powers [H^4, H^3, H^2, H].
      let h_vec = _mm512_loadu_si512(h_powers_rev.as_ptr().cast());

      // 4-way parallel Karatsuba: 3 VPCLMULQDQ instructions.
      let lo = _mm512_clmulepi64_epi128(data, h_vec, 0x00); // a_lo * b_lo
      let hi = _mm512_clmulepi64_epi128(data, h_vec, 0x11); // a_hi * b_hi
      let data_mid = _mm512_xor_si512(data, _mm512_shuffle_epi32::<0x4e>(data));
      let h_mid = _mm512_xor_si512(h_vec, _mm512_shuffle_epi32::<0x4e>(h_vec));
      let mid = _mm512_xor_si512(
        _mm512_xor_si512(_mm512_clmulepi64_epi128(data_mid, h_mid, 0x00), lo),
        hi,
      );

      // Assemble per-lane 256-bit products [lo_128 : hi_128].
      let lo_128 = _mm512_xor_si512(lo, _mm512_bslli_epi128(mid, 8));
      let hi_128 = _mm512_xor_si512(hi, _mm512_bsrli_epi128(mid, 8));

      // Reduce 4 lanes -> 1 lane via cascading XOR.
      let lo_0 = _mm512_extracti64x2_epi64(lo_128, 0);
      let lo_1 = _mm512_extracti64x2_epi64(lo_128, 1);
      let lo_2 = _mm512_extracti64x2_epi64(lo_128, 2);
      let lo_3 = _mm512_extracti64x2_epi64(lo_128, 3);
      let lo_sum = _mm_xor_si128(_mm_xor_si128(lo_0, lo_1), _mm_xor_si128(lo_2, lo_3));

      let hi_0 = _mm512_extracti64x2_epi64(hi_128, 0);
      let hi_1 = _mm512_extracti64x2_epi64(hi_128, 1);
      let hi_2 = _mm512_extracti64x2_epi64(hi_128, 2);
      let hi_3 = _mm512_extracti64x2_epi64(hi_128, 3);
      let hi_sum = _mm_xor_si128(_mm_xor_si128(hi_0, hi_1), _mm_xor_si128(hi_2, hi_3));

      // Single Montgomery reduction.
      let result = super::pclmul::mont_reduce_sse2(lo_sum, hi_sum);

      let mut out = 0u128;
      _mm_storeu_si128((&mut out as *mut u128).cast(), result);
      out
    }
  }

  /// Reduce sixteen already packed POLYVAL-domain lanes into one accumulator.
  ///
  /// # Safety
  /// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
  /// VPCLMULQDQ + PCLMULQDQ + SSE2.
  #[cfg(any(feature = "aes-gcm", feature = "aes-gcm-siv"))]
  #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
  #[inline]
  unsafe fn aggregate_16_lanes(data: [__m512i; 4], h_powers_rev: &[u128; 16]) -> u128 {
    // SAFETY: x86 VPCLMUL 16-block aggregation because:
    // 1. This function's caller guarantees all required target features.
    // 2. `data` contains sixteen initialized POLYVAL-domain lanes in order.
    // 3. `h_powers_rev` contains exactly [H^16, H^15, ..., H], matching the lanes.
    unsafe {
      let mut lo_sum = _mm512_setzero_si512();
      let mut hi_sum = _mm512_setzero_si512();

      macro_rules! fold_lanes {
        ($data:expr, $power_offset:expr) => {{
          let h_vec = _mm512_loadu_si512(h_powers_rev.as_ptr().add($power_offset).cast());
          let lo = _mm512_clmulepi64_epi128($data, h_vec, 0x00);
          let hi = _mm512_clmulepi64_epi128($data, h_vec, 0x11);
          let data_mid = _mm512_xor_si512($data, _mm512_shuffle_epi32::<0x4e>($data));
          let h_mid = _mm512_xor_si512(h_vec, _mm512_shuffle_epi32::<0x4e>(h_vec));
          let mid = _mm512_xor_si512(
            _mm512_xor_si512(_mm512_clmulepi64_epi128(data_mid, h_mid, 0x00), lo),
            hi,
          );

          lo_sum = _mm512_xor_si512(lo_sum, _mm512_xor_si512(lo, _mm512_bslli_epi128(mid, 8)));
          hi_sum = _mm512_xor_si512(hi_sum, _mm512_xor_si512(hi, _mm512_bsrli_epi128(mid, 8)));
        }};
      }

      fold_lanes!(data[0], 0);
      fold_lanes!(data[1], 4);
      fold_lanes!(data[2], 8);
      fold_lanes!(data[3], 12);

      let lo_0 = _mm512_extracti64x2_epi64(lo_sum, 0);
      let lo_1 = _mm512_extracti64x2_epi64(lo_sum, 1);
      let lo_2 = _mm512_extracti64x2_epi64(lo_sum, 2);
      let lo_3 = _mm512_extracti64x2_epi64(lo_sum, 3);
      let lo = _mm_xor_si128(_mm_xor_si128(lo_0, lo_1), _mm_xor_si128(lo_2, lo_3));

      let hi_0 = _mm512_extracti64x2_epi64(hi_sum, 0);
      let hi_1 = _mm512_extracti64x2_epi64(hi_sum, 1);
      let hi_2 = _mm512_extracti64x2_epi64(hi_sum, 2);
      let hi_3 = _mm512_extracti64x2_epi64(hi_sum, 3);
      let hi = _mm_xor_si128(_mm_xor_si128(hi_0, hi_1), _mm_xor_si128(hi_2, hi_3));

      let result = super::pclmul::mont_reduce_sse2(lo, hi);
      let mut out = 0u128;
      _mm_storeu_si128((&mut out as *mut u128).cast(), result);
      out
    }
  }

  /// Reduce eight POLYVAL-domain lanes using 256-bit VPCLMULQDQ.
  ///
  /// # Safety
  /// Caller must ensure AVX2 + AVX-512F + AVX-512VL + VPCLMULQDQ +
  /// PCLMULQDQ + SSE2.
  #[cfg(feature = "aes-gcm")]
  #[target_feature(enable = "avx2,avx512f,avx512vl,vpclmulqdq,pclmulqdq,sse2")]
  #[inline]
  unsafe fn aggregate_8_lanes_256(data: [__m256i; 4], h_powers_rev: &[u128; 8]) -> u128 {
    // SAFETY: x86 VPCLMUL 8-block aggregation because:
    // 1. This function's caller guarantees all required target features.
    // 2. `data` contains eight initialized POLYVAL-domain lanes in order.
    // 3. `h_powers_rev` contains exactly [H^8, H^7, ..., H], matching the lanes.
    unsafe {
      let mut lo_sum = _mm256_setzero_si256();
      let mut hi_sum = _mm256_setzero_si256();

      macro_rules! fold_lanes {
        ($data:expr, $power_offset:expr) => {{
          let h_vec = _mm256_loadu_si256(h_powers_rev.as_ptr().add($power_offset).cast());
          let lo = _mm256_clmulepi64_epi128($data, h_vec, 0x00);
          let hi = _mm256_clmulepi64_epi128($data, h_vec, 0x11);
          let data_mid = _mm256_xor_si256($data, _mm256_shuffle_epi32::<0x4e>($data));
          let h_mid = _mm256_xor_si256(h_vec, _mm256_shuffle_epi32::<0x4e>(h_vec));
          let mid = _mm256_xor_si256(
            _mm256_xor_si256(_mm256_clmulepi64_epi128(data_mid, h_mid, 0x00), lo),
            hi,
          );

          lo_sum = _mm256_xor_si256(lo_sum, _mm256_xor_si256(lo, _mm256_bslli_epi128(mid, 8)));
          hi_sum = _mm256_xor_si256(hi_sum, _mm256_xor_si256(hi, _mm256_bsrli_epi128(mid, 8)));
        }};
      }

      fold_lanes!(data[0], 0);
      fold_lanes!(data[1], 2);
      fold_lanes!(data[2], 4);
      fold_lanes!(data[3], 6);

      let lo = _mm_xor_si128(_mm256_castsi256_si128(lo_sum), _mm256_extracti128_si256(lo_sum, 1));
      let hi = _mm_xor_si128(_mm256_castsi256_si128(hi_sum), _mm256_extracti128_si256(hi_sum, 1));

      let result = super::pclmul::mont_reduce_sse2(lo, hi);
      let mut out = 0u128;
      _mm_storeu_si128((&mut out as *mut u128).cast(), result);
      out
    }
  }

  /// Convert four big-endian GHASH lanes into POLYVAL-domain lanes.
  ///
  /// # Safety
  /// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ.
  #[cfg(feature = "aes-gcm")]
  #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
  #[inline]
  unsafe fn be_lanes(raw: __m512i) -> __m512i {
    let reverse_bytes = _mm512_set_epi32(
      0x0001_0203,
      0x0405_0607,
      0x0809_0a0b,
      0x0c0d_0e0f,
      0x0001_0203,
      0x0405_0607,
      0x0809_0a0b,
      0x0c0d_0e0f,
      0x0001_0203,
      0x0405_0607,
      0x0809_0a0b,
      0x0c0d_0e0f,
      0x0001_0203,
      0x0405_0607,
      0x0809_0a0b,
      0x0c0d_0e0f,
    );
    _mm512_shuffle_epi8(raw, reverse_bytes)
  }

  /// Convert two big-endian GHASH lanes into POLYVAL-domain lanes.
  ///
  /// # Safety
  /// Caller must ensure AVX2 is available.
  #[cfg(feature = "aes-gcm")]
  #[target_feature(enable = "avx2")]
  #[inline]
  unsafe fn be_lanes_256(raw: __m256i) -> __m256i {
    let reverse_bytes = _mm256_set_epi32(
      0x0001_0203,
      0x0405_0607,
      0x0809_0a0b,
      0x0c0d_0e0f,
      0x0001_0203,
      0x0405_0607,
      0x0809_0a0b,
      0x0c0d_0e0f,
    );
    _mm256_shuffle_epi8(raw, reverse_bytes)
  }

  /// Convert four big-endian GHASH lanes and XOR `acc` into the first lane.
  ///
  /// # Safety
  /// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + SSE2.
  #[cfg(feature = "aes-gcm")]
  #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
  #[inline]
  unsafe fn be_lanes_with_acc(acc: u128, raw: __m512i) -> __m512i {
    // SAFETY: direct GHASH lane conversion because:
    // 1. This function's caller guarantees the required x86 target features.
    // 2. `raw` contains four initialized big-endian GHASH lanes and `acc` is initialized.
    unsafe {
      let data = be_lanes(raw);
      let acc_lane = _mm512_zextsi128_si512(_mm_loadu_si128((&acc as *const u128).cast()));
      _mm512_xor_si512(data, acc_lane)
    }
  }

  /// Convert two big-endian GHASH lanes and XOR `acc` into the first lane.
  ///
  /// # Safety
  /// Caller must ensure AVX2 and SSE2 are available.
  #[cfg(feature = "aes-gcm")]
  #[target_feature(enable = "avx2,sse2")]
  #[inline]
  unsafe fn be_lanes_256_with_acc(acc: u128, raw: __m256i) -> __m256i {
    // SAFETY: direct GHASH lane conversion because:
    // 1. This function's caller guarantees the required x86 target features.
    // 2. `raw` contains two initialized big-endian GHASH lanes and `acc` is initialized.
    unsafe {
      let data = be_lanes_256(raw);
      let acc_lane = _mm256_castsi128_si256(_mm_loadu_si128((&acc as *const u128).cast()));
      _mm256_xor_si256(data, acc_lane)
    }
  }

  /// Load four big-endian GHASH blocks and XOR `acc` into the first lane.
  ///
  /// # Safety
  /// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + SSE2.
  #[cfg(feature = "aes-gcm")]
  #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
  #[inline]
  unsafe fn load_be_4blocks(acc: u128, block_bytes: &[u8; 64]) -> __m512i {
    // SAFETY: direct GHASH byte loading because:
    // 1. This function's caller guarantees the required x86 target features.
    // 2. `block_bytes` is exactly 64 initialized bytes and `acc` is an initialized field element.
    unsafe {
      let raw = _mm512_loadu_si512(block_bytes.as_ptr().cast());
      be_lanes_with_acc(acc, raw)
    }
  }

  /// Process 4 blocks in parallel using VPCLMULQDQ schoolbook-then-reduce.
  ///
  /// Computes: (acc ^ b0) * H4 ^ b1 * H3 ^ b2 * H2 ^ b3 * H
  ///
  /// The 4 schoolbook 128x128 products are computed in parallel with 4
  /// `_mm512_clmulepi64_epi128` instructions, XOR'd together into one
  /// 256-bit aggregate, then reduced with a single Montgomery reduction.
  ///
  /// # Safety
  /// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
  /// VPCLMULQDQ + PCLMULQDQ + SSE2.
  #[cfg(any(feature = "aes-gcm", feature = "aes-gcm-siv"))]
  #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
  pub(super) unsafe fn aggregate_4blocks(
    acc: u128,
    h_powers_rev: &[u128; 4], // [H^4, H^3, H^2, H]
    blocks: &[u128; 4],       // [b0, b1, b2, b3] in correct domain
  ) -> u128 {
    // SAFETY: x86 VPCLMUL aggregation because:
    // 1. This function's caller guarantees all required target features.
    // 2. `blocks` and `h_powers_rev` are fixed 4-lane initialized arrays.
    unsafe {
      // XOR accumulator into the first block.
      let mut block_data = *blocks;
      block_data[0] ^= acc;
      let data = _mm512_loadu_si512(block_data.as_ptr().cast());
      aggregate_lanes(data, h_powers_rev)
    }
  }

  /// Process 16 POLYVAL-domain blocks using VPCLMULQDQ and one shared reduction.
  ///
  /// # Safety
  /// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
  /// VPCLMULQDQ + PCLMULQDQ + SSE2.
  #[cfg(any(feature = "aes-gcm-siv", test))]
  #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
  pub(super) unsafe fn aggregate_16blocks(acc: u128, h_powers_rev: &[u128; 16], blocks: &[u128; 16]) -> u128 {
    // SAFETY: x86 VPCLMUL aggregation because:
    // 1. This function's caller guarantees all required target features.
    // 2. `blocks` and `h_powers_rev` are fixed 16-lane initialized arrays.
    // 3. The accumulator is folded into the first lane, matching sequential GHASH/POLYVAL.
    unsafe {
      let mut block_data = *blocks;
      block_data[0] ^= acc;
      aggregate_16_lanes(
        [
          _mm512_loadu_si512(block_data.as_ptr().cast()),
          _mm512_loadu_si512(block_data.as_ptr().add(4).cast()),
          _mm512_loadu_si512(block_data.as_ptr().add(8).cast()),
          _mm512_loadu_si512(block_data.as_ptr().add(12).cast()),
        ],
        h_powers_rev,
      )
    }
  }

  /// Process 4 little-endian POLYVAL blocks directly from bytes.
  ///
  /// # Safety
  /// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
  /// VPCLMULQDQ + PCLMULQDQ + SSE2, and `block_ptr` must point at at least
  /// 64 initialized bytes.
  #[cfg(feature = "aes-gcm-siv")]
  #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
  pub(super) unsafe fn aggregate_4blocks_le_bytes(acc: u128, h_powers_rev: &[u128; 4], block_ptr: *const u8) -> u128 {
    // SAFETY: direct-byte POLYVAL aggregation because:
    // 1. This function's caller guarantees all required x86 target features.
    // 2. `block_ptr` points at four initialized 16-byte POLYVAL blocks.
    // 3. POLYVAL uses little-endian field encoding, so byte memory order already matches the lane
    //    representation expected by `aggregate_lanes`.
    unsafe {
      let data = _mm512_loadu_si512(block_ptr.cast());
      let acc_lane = _mm512_zextsi128_si512(_mm_loadu_si128((&acc as *const u128).cast()));
      aggregate_lanes(_mm512_xor_si512(data, acc_lane), h_powers_rev)
    }
  }

  /// Process 16 little-endian POLYVAL blocks directly from bytes.
  ///
  /// # Safety
  /// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
  /// VPCLMULQDQ + PCLMULQDQ + SSE2, and `block_ptr` must point at at least
  /// 256 initialized bytes.
  #[cfg(feature = "aes-gcm-siv")]
  #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
  pub(super) unsafe fn aggregate_16blocks_le_bytes(acc: u128, h_powers_rev: &[u128; 16], block_ptr: *const u8) -> u128 {
    // SAFETY: direct-byte POLYVAL aggregation because:
    // 1. This function's caller guarantees all required x86 target features.
    // 2. `block_ptr` points at sixteen initialized 16-byte POLYVAL blocks.
    // 3. Only the first lane receives the incoming accumulator, matching the POLYVAL recurrence.
    unsafe {
      let data0 = _mm512_loadu_si512(block_ptr.cast());
      let acc_lane = _mm512_zextsi128_si512(_mm_loadu_si128((&acc as *const u128).cast()));
      aggregate_16_lanes(
        [
          _mm512_xor_si512(data0, acc_lane),
          _mm512_loadu_si512(block_ptr.add(64).cast()),
          _mm512_loadu_si512(block_ptr.add(128).cast()),
          _mm512_loadu_si512(block_ptr.add(192).cast()),
        ],
        h_powers_rev,
      )
    }
  }

  /// Process 4 big-endian GHASH blocks directly from bytes.
  ///
  /// Equivalent to `aggregate_4blocks(acc, h_powers_rev, blocks)` with
  /// `blocks[i] = u128::from_be_bytes(block_bytes[i])`, but keeps the
  /// byte-reversal and lane packing in SIMD.
  ///
  /// # Safety
  /// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
  /// VPCLMULQDQ + PCLMULQDQ + SSE2.
  #[cfg(feature = "aes-gcm")]
  #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
  pub(super) unsafe fn aggregate_4blocks_be_bytes(acc: u128, h_powers_rev: &[u128; 4], block_bytes: &[u8; 64]) -> u128 {
    // SAFETY: direct-byte GHASH aggregation because:
    // 1. This function's caller guarantees all required x86 target features.
    // 2. `block_bytes` is exactly four initialized 16-byte GHASH blocks.
    unsafe { aggregate_lanes(load_be_4blocks(acc, block_bytes), h_powers_rev) }
  }

  /// Process 4 big-endian GHASH lanes already resident in a SIMD register.
  ///
  /// # Safety
  /// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
  /// VPCLMULQDQ + PCLMULQDQ + SSE2.
  #[cfg(feature = "aes-gcm")]
  #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
  pub(super) unsafe fn aggregate_4blocks_be_lanes(acc: u128, h_powers_rev: &[u128; 4], raw: __m512i) -> u128 {
    // SAFETY: direct-lane GHASH aggregation because:
    // 1. This function's caller guarantees all required x86 target features.
    // 2. `raw` contains four initialized 16-byte ciphertext lanes in memory byte order.
    unsafe { aggregate_lanes(be_lanes_with_acc(acc, raw), h_powers_rev) }
  }

  /// Process 16 big-endian GHASH lanes already resident in SIMD registers.
  ///
  /// # Safety
  /// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
  /// VPCLMULQDQ + PCLMULQDQ + SSE2.
  #[cfg(feature = "aes-gcm")]
  #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
  pub(super) unsafe fn aggregate_16blocks_be_lanes(
    acc: u128,
    h_powers_rev: &[u128; 16],
    raw0: __m512i,
    raw1: __m512i,
    raw2: __m512i,
    raw3: __m512i,
  ) -> u128 {
    // SAFETY: direct-lane GHASH aggregation because:
    // 1. This function's caller guarantees all required x86 target features.
    // 2. `raw*` contain sixteen initialized 16-byte ciphertext lanes in memory byte order.
    // 3. Only the first lane receives the incoming accumulator, matching GHASH recurrence.
    unsafe {
      aggregate_16_lanes(
        [
          be_lanes_with_acc(acc, raw0),
          be_lanes(raw1),
          be_lanes(raw2),
          be_lanes(raw3),
        ],
        h_powers_rev,
      )
    }
  }

  /// Process 8 big-endian GHASH lanes already resident in 256-bit SIMD registers.
  ///
  /// # Safety
  /// Caller must ensure AVX2 + AVX-512F + AVX-512VL + VPCLMULQDQ +
  /// PCLMULQDQ + SSE2.
  #[cfg(feature = "aes-gcm")]
  #[target_feature(enable = "avx2,avx512f,avx512vl,vpclmulqdq,pclmulqdq,sse2")]
  pub(super) unsafe fn aggregate_8blocks_be_lanes_256(
    acc: u128,
    h_powers_rev: &[u128; 8],
    raw0: __m256i,
    raw1: __m256i,
    raw2: __m256i,
    raw3: __m256i,
  ) -> u128 {
    // SAFETY: direct-lane GHASH aggregation because:
    // 1. This function's caller guarantees all required x86 target features.
    // 2. `raw*` contain eight initialized 16-byte ciphertext lanes in memory byte order.
    // 3. Only the first lane receives the incoming accumulator, matching GHASH recurrence.
    unsafe {
      aggregate_8_lanes_256(
        [
          be_lanes_256_with_acc(acc, raw0),
          be_lanes_256(raw1),
          be_lanes_256(raw2),
          be_lanes_256(raw3),
        ],
        h_powers_rev,
      )
    }
  }
}

// ---------------------------------------------------------------------------
// x86_64: inline helpers for fused GCM paths
// ---------------------------------------------------------------------------

/// PCLMULQDQ-based 128×128 carryless multiply + Montgomery reduce.
///
/// # Safety
/// Caller must ensure PCLMULQDQ and SSE2 are available.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "pclmulqdq,sse2")]
#[inline]
pub(super) unsafe fn x86_clmul128_reduce_inline(a: u128, b: u128) -> u128 {
  // SAFETY: x86 carryless multiply because:
  // 1. Caller guarantees PCLMULQDQ and SSE2 availability.
  // 2. `a` and `b` are initialized POLYVAL-domain field elements.
  unsafe { pclmul::clmul128_reduce(a, b) }
}

/// PCLMULQDQ 4-block aggregate helper for already domain-converted blocks.
///
/// # Safety
/// Caller must ensure PCLMULQDQ and SSE2 are available.
#[cfg(all(target_arch = "x86_64", any(feature = "aes-gcm", feature = "aes-gcm-siv")))]
#[target_feature(enable = "pclmulqdq,sse2")]
#[inline]
pub(super) unsafe fn x86_pclmul_aggregate_4blocks_inline(
  acc: u128,
  h_powers_rev: &[u128; 4],
  blocks: &[u128; 4],
) -> u128 {
  // SAFETY: PCLMUL aggregate because:
  // 1. Caller guarantees PCLMULQDQ and SSE2 availability.
  // 2. `h_powers_rev` and `blocks` are initialized fixed-width inputs.
  unsafe { pclmul::aggregate_4blocks(acc, h_powers_rev, blocks) }
}

/// 4-block VPCLMULQDQ aggregate helper for little-endian POLYVAL bytes.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
/// VPCLMULQDQ + PCLMULQDQ + SSE2 are available, and `block_ptr` points at
/// at least 64 initialized bytes.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
#[inline]
pub(super) unsafe fn x86_aggregate_4blocks_le_bytes_inline(
  acc: u128,
  h_powers_rev: &[u128; 4],
  block_ptr: *const u8,
) -> u128 {
  // SAFETY: direct-byte VPCLMUL POLYVAL aggregation because:
  // 1. Caller guarantees all required x86 target features.
  // 2. `block_ptr` points at four initialized 16-byte POLYVAL blocks.
  unsafe { vpclmul::aggregate_4blocks_le_bytes(acc, h_powers_rev, block_ptr) }
}

/// 16-block VPCLMULQDQ aggregate helper for little-endian POLYVAL bytes.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
/// VPCLMULQDQ + PCLMULQDQ + SSE2 are available, and `block_ptr` points at
/// at least 256 initialized bytes.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
#[inline]
pub(super) unsafe fn x86_aggregate_16blocks_le_bytes_inline(
  acc: u128,
  h_powers_rev: &[u128; 16],
  block_ptr: *const u8,
) -> u128 {
  // SAFETY: direct-byte VPCLMUL POLYVAL aggregation because:
  // 1. Caller guarantees all required x86 target features.
  // 2. `block_ptr` points at sixteen initialized 16-byte POLYVAL blocks.
  // 3. The helper folds the incoming accumulator only into the first lane.
  unsafe { vpclmul::aggregate_16blocks_le_bytes(acc, h_powers_rev, block_ptr) }
}

/// PCLMULQDQ 4-block aggregate helper for big-endian GHASH bytes already in XMM registers.
///
/// # Safety
/// Caller must ensure PCLMULQDQ, SSE2, and SSSE3 are available.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "pclmulqdq,sse2,ssse3")]
#[inline]
pub(super) unsafe fn x86_pclmul_aggregate_4blocks_be_xmm_inline(
  acc: u128,
  h_powers_rev: &[u128; 4],
  raw0: core::arch::x86_64::__m128i,
  raw1: core::arch::x86_64::__m128i,
  raw2: core::arch::x86_64::__m128i,
  raw3: core::arch::x86_64::__m128i,
) -> u128 {
  // SAFETY: direct-register PCLMUL GHASH aggregation because:
  // 1. Caller guarantees PCLMULQDQ, SSE2, and SSSE3 availability.
  // 2. `raw*` contain initialized 16-byte GHASH blocks in memory byte order.
  unsafe { pclmul::aggregate_4blocks_be_xmm(acc, h_powers_rev, raw0, raw1, raw2, raw3) }
}

/// 4-block VPCLMULQDQ aggregate helper that loads big-endian GHASH bytes directly.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
/// VPCLMULQDQ + PCLMULQDQ + SSE2 are available.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
#[inline]
pub(super) unsafe fn x86_aggregate_4blocks_be_bytes_inline(
  acc: u128,
  h_powers_rev: &[u128; 4],
  block_bytes: &[u8; 64],
) -> u128 {
  // SAFETY: direct-byte VPCLMUL aggregation because:
  // 1. Caller guarantees all required x86 target features.
  // 2. `h_powers_rev` and `block_bytes` are fixed-size initialized inputs matching the backend
  //    contract.
  unsafe { vpclmul::aggregate_4blocks_be_bytes(acc, h_powers_rev, block_bytes) }
}

/// 4-block VPCLMULQDQ aggregate helper for big-endian GHASH lanes already in a register.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
/// VPCLMULQDQ + PCLMULQDQ + SSE2 are available.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
#[inline]
pub(super) unsafe fn x86_aggregate_4blocks_be_lanes_inline(
  acc: u128,
  h_powers_rev: &[u128; 4],
  raw: core::arch::x86_64::__m512i,
) -> u128 {
  // SAFETY: direct-lane VPCLMUL aggregation because:
  // 1. Caller guarantees all required x86 target features.
  // 2. `h_powers_rev` and `raw` are initialized inputs matching the backend contract.
  unsafe { vpclmul::aggregate_4blocks_be_lanes(acc, h_powers_rev, raw) }
}

/// 16-block VPCLMULQDQ aggregate helper for big-endian GHASH lanes already in registers.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512VL + AVX-512BW + AVX-512DQ +
/// VPCLMULQDQ + PCLMULQDQ + SSE2 are available.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
#[inline]
pub(super) unsafe fn x86_aggregate_16blocks_be_lanes_inline(
  acc: u128,
  h_powers_rev: &[u128; 16],
  raw0: core::arch::x86_64::__m512i,
  raw1: core::arch::x86_64::__m512i,
  raw2: core::arch::x86_64::__m512i,
  raw3: core::arch::x86_64::__m512i,
) -> u128 {
  // SAFETY: direct-lane VPCLMUL aggregation because:
  // 1. Caller guarantees all required x86 target features.
  // 2. `h_powers_rev` and `raw*` are initialized inputs matching the backend contract.
  // 3. The helper folds the incoming accumulator only into the first lane.
  unsafe { vpclmul::aggregate_16blocks_be_lanes(acc, h_powers_rev, raw0, raw1, raw2, raw3) }
}

/// 8-block VPCLMULQDQ aggregate helper for big-endian GHASH lanes already in 256-bit registers.
///
/// # Safety
/// Caller must ensure AVX2 + AVX-512F + AVX-512VL + VPCLMULQDQ +
/// PCLMULQDQ + SSE2 are available.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
#[target_feature(enable = "avx2,avx512f,avx512vl,vpclmulqdq,pclmulqdq,sse2")]
#[inline]
pub(super) unsafe fn x86_aggregate_8blocks_be_lanes_256_inline(
  acc: u128,
  h_powers_rev: &[u128; 8],
  raw0: core::arch::x86_64::__m256i,
  raw1: core::arch::x86_64::__m256i,
  raw2: core::arch::x86_64::__m256i,
  raw3: core::arch::x86_64::__m256i,
) -> u128 {
  // SAFETY: direct-lane VPCLMUL aggregation because:
  // 1. Caller guarantees all required target features.
  // 2. `h_powers_rev` and `raw*` are initialized inputs matching the backend contract.
  // 3. The helper folds the incoming accumulator only into the first lane.
  unsafe { vpclmul::aggregate_8blocks_be_lanes_256(acc, h_powers_rev, raw0, raw1, raw2, raw3) }
}

// ---------------------------------------------------------------------------
// aarch64: inline helper for fused paths (#[target_feature] + #[inline(always)])
// ---------------------------------------------------------------------------

/// PMULL-based 128×128 carryless multiply + Montgomery reduce, guaranteed
/// to inline.
///
/// Hot-path helper for fused `#[target_feature(enable = "aes,neon")]`
/// POLYVAL updates.
///
/// # Safety
/// Caller must ensure PMULL is available.
#[cfg(all(target_arch = "aarch64", any(feature = "aes-gcm", feature = "aes-gcm-siv")))]
#[target_feature(enable = "neon", enable = "aes")]
#[inline]
pub(super) unsafe fn aarch64_clmul128_reduce_inline(a: u128, b: u128) -> u128 {
  // SAFETY: PMULL reduction because:
  // 1. This helper is only callable from an `aes,neon` target-feature scope.
  // 2. The caller guarantees PMULL/AES availability on the current CPU.
  // 3. `a` and `b` are plain 128-bit field elements with no pointer aliasing contract.
  unsafe { pmull::clmul128_reduce_core(a, b) }
}

/// 4-block wide PMULL aggregate helper.
///
/// # Safety
/// Caller must ensure PMULL is available.
#[cfg(all(target_arch = "aarch64", any(feature = "aes-gcm", feature = "aes-gcm-siv")))]
#[target_feature(enable = "neon", enable = "aes")]
#[inline]
pub(super) unsafe fn aarch64_aggregate_4blocks_inline(acc: u128, h_powers_rev: &[u128; 4], blocks: &[u128; 4]) -> u128 {
  // SAFETY: PMULL aggregate call because:
  // 1. Caller guarantees PMULL availability before invoking this unsafe helper.
  // 2. `h_powers_rev` and `blocks` are fixed 4-lane arrays matching the backend contract.
  unsafe { pmull::aggregate_4blocks(acc, h_powers_rev, blocks) }
}

/// 8-block wide PMULL aggregate helper for big-endian GHASH lanes already in NEON registers.
///
/// # Safety
/// Caller must ensure PMULL is available.
#[cfg(all(target_arch = "aarch64", feature = "aes-gcm"))]
#[target_feature(enable = "neon", enable = "aes")]
#[inline]
pub(super) unsafe fn aarch64_aggregate_8blocks_be_lanes_inline(
  acc: u128,
  h_powers_rev: &[u128; 8],
  blocks: &[core::arch::aarch64::uint8x16_t; 8],
) -> u128 {
  // SAFETY: PMULL aggregate call because:
  // 1. Caller guarantees PMULL availability before invoking this unsafe helper.
  // 2. `h_powers_rev` and `blocks` are fixed 8-lane initialized inputs.
  unsafe { pmull::aggregate_8blocks_be_lanes(acc, h_powers_rev, blocks) }
}

/// POWER VPMSUMD multiply + reduce helper.
///
/// # Safety
/// Caller must ensure POWER8 crypto is available.
#[cfg(all(target_arch = "powerpc64", any(feature = "aes-gcm", feature = "aes-gcm-siv")))]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
#[inline]
pub(super) unsafe fn ppc_clmul128_reduce_inline(a: u128, b: u128) -> u128 {
  // SAFETY: POWER8 VPMSUMD reduction because:
  // 1. This helper is only callable from a POWER8 crypto target-feature scope.
  // 2. The caller guarantees POWER8 crypto availability on the current CPU.
  // 3. `a` and `b` are plain 128-bit field elements with no pointer aliasing contract.
  unsafe { ppc_vpmsum::clmul128_reduce_core(a, b) }
}

/// 4-block wide VPMSUMD aggregate helper.
///
/// # Safety
/// Caller must ensure POWER8 crypto is available.
#[cfg(all(target_arch = "powerpc64", any(feature = "aes-gcm", feature = "aes-gcm-siv")))]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
#[inline]
pub(super) unsafe fn ppc_aggregate_4blocks_inline(acc: u128, h_powers_rev: &[u128; 4], blocks: &[u128; 4]) -> u128 {
  // SAFETY: POWER8 aggregate call because:
  // 1. Caller guarantees POWER8 crypto availability before invoking this unsafe helper.
  // 2. `h_powers_rev` and `blocks` are fixed 4-lane arrays matching the backend contract.
  unsafe { ppc_vpmsum::aggregate_4blocks(acc, h_powers_rev, blocks) }
}

/// s390x VGFM multiply + reduce helper.
///
/// # Safety
/// Caller must ensure z/Vector is available.
#[cfg(all(target_arch = "s390x", feature = "aes-gcm-siv"))]
#[target_feature(enable = "vector")]
#[inline]
pub(super) unsafe fn s390x_clmul128_reduce_inline(a: u128, b: u128) -> u128 {
  // SAFETY: s390x VGFM reduction because:
  // 1. This helper is only callable from a z/Vector target-feature scope.
  // 2. The caller guarantees z/Vector availability on the current CPU.
  // 3. `a` and `b` are plain 128-bit field elements with no pointer aliasing contract.
  unsafe { s390x_vgfm::clmul128_reduce_core(a, b) }
}

/// 4-block wide s390x VGFM aggregate helper.
///
/// # Safety
/// Caller must ensure z/Vector is available.
#[cfg(all(target_arch = "s390x", any(feature = "aes-gcm", feature = "aes-gcm-siv")))]
#[target_feature(enable = "vector")]
#[inline]
pub(super) unsafe fn s390x_aggregate_4blocks_inline(acc: u128, h_powers_rev: &[u128; 4], blocks: &[u128; 4]) -> u128 {
  // SAFETY: z/Vector aggregate call because:
  // 1. Caller guarantees vector-facility availability before invoking this unsafe helper.
  // 2. `h_powers_rev` and `blocks` are fixed 4-lane arrays matching the backend contract.
  unsafe { s390x_vgfm::aggregate_4blocks(acc, h_powers_rev, blocks) }
}

#[cfg(all(target_arch = "riscv64", feature = "aes-gcm-siv"))]
/// Explicit portable carryless multiply + reduce.
///
/// Unlike `clmul128_reduce`, this never climbs into a hardware backend.
#[inline(always)]
pub(super) fn portable_clmul128_reduce_inline(a: u128, b: u128) -> u128 {
  mont_reduce(clmul128(a, b))
}

/// RISC-V Zvbc carryless multiply + reduce helper.
///
/// # Safety
/// Caller must ensure Zvbc is available.
#[cfg(all(target_arch = "riscv64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "v", enable = "zvbc")]
#[inline]
pub(super) unsafe fn riscv_vector_clmul128_reduce_inline(a: u128, b: u128) -> u128 {
  // SAFETY: RISC-V vector carryless reduction because:
  // 1. This helper is only callable from a `v,zvbc` target-feature scope.
  // 2. The caller guarantees Zvbc availability on the current CPU.
  // 3. `a` and `b` are plain 128-bit field elements with no pointer aliasing contract.
  unsafe { rv_clmul::clmul128_reduce(a, b) }
}

/// RISC-V Zbc/Zbkc carryless multiply + reduce helper.
///
/// # Safety
/// Caller must ensure Zbc or Zbkc is available.
#[cfg(all(target_arch = "riscv64", feature = "aes-gcm-siv"))]
#[target_feature(enable = "zbc")]
#[inline]
pub(super) unsafe fn riscv_scalar_clmul128_reduce_inline(a: u128, b: u128) -> u128 {
  // SAFETY: RISC-V scalar carryless reduction because:
  // 1. This helper is only callable from a `zbc` target-feature scope.
  // 2. The caller guarantees Zbc/Zbkc availability on the current CPU.
  // 3. `a` and `b` are plain 128-bit field elements with no pointer aliasing contract.
  unsafe { rv_scalar_clmul::clmul128_reduce(a, b) }
}

// ---------------------------------------------------------------------------
// Combined multiply + reduce with hardware dispatch
// ---------------------------------------------------------------------------

type Clmul128ReduceFn = fn(u128, u128) -> u128;

static CLMUL128_REDUCE_DISPATCH: OnceCache<Clmul128ReduceFn> = OnceCache::new();

#[inline]
fn current_caps() -> crate::platform::Caps {
  #[cfg(feature = "std")]
  {
    crate::platform::caps()
  }

  #[cfg(not(feature = "std"))]
  {
    crate::platform::caps_static()
  }
}

#[inline]
fn clmul128_reduce_portable(a: u128, b: u128) -> u128 {
  mont_reduce(clmul128(a, b))
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn clmul128_reduce_x86_pclmul(a: u128, b: u128) -> u128 {
  // SAFETY: resolver only selects this backend after verifying PCLMULQDQ support.
  unsafe { pclmul::clmul128_reduce(a, b) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn clmul128_reduce_aarch64_pmull(a: u128, b: u128) -> u128 {
  // SAFETY: resolver only selects this backend after verifying PMULL support.
  unsafe { pmull::clmul128_reduce(a, b) }
}

#[cfg(target_arch = "s390x")]
#[inline]
fn clmul128_reduce_s390x_vgfm(a: u128, b: u128) -> u128 {
  // SAFETY: resolver only selects this backend after verifying z/Vector support.
  unsafe { s390x_vgfm::clmul128_reduce(a, b) }
}

#[cfg(target_arch = "powerpc64")]
#[inline]
fn clmul128_reduce_power_vpmsum(a: u128, b: u128) -> u128 {
  // SAFETY: resolver only selects this backend after verifying POWER8 crypto support.
  unsafe { ppc_vpmsum::clmul128_reduce(a, b) }
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn clmul128_reduce_riscv_vector(a: u128, b: u128) -> u128 {
  // SAFETY: resolver only selects this backend after verifying Zvbc support.
  unsafe { rv_clmul::clmul128_reduce(a, b) }
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn clmul128_reduce_riscv_scalar(a: u128, b: u128) -> u128 {
  // SAFETY: resolver only selects this backend after verifying Zbc/Zbkc support.
  unsafe { rv_scalar_clmul::clmul128_reduce(a, b) }
}

#[inline]
fn resolve_clmul128_reduce() -> Clmul128ReduceFn {
  let _caps = current_caps();

  #[cfg(target_arch = "x86_64")]
  if _caps.has(crate::platform::caps::x86::PCLMULQDQ) {
    return clmul128_reduce_x86_pclmul;
  }

  #[cfg(target_arch = "aarch64")]
  if _caps.has(crate::platform::caps::aarch64::PMULL) {
    return clmul128_reduce_aarch64_pmull;
  }

  #[cfg(target_arch = "s390x")]
  if _caps.has(crate::platform::caps::s390x::VECTOR) {
    return clmul128_reduce_s390x_vgfm;
  }

  #[cfg(target_arch = "powerpc64")]
  if _caps.has(crate::platform::caps::power::POWER8_CRYPTO) {
    return clmul128_reduce_power_vpmsum;
  }

  #[cfg(target_arch = "riscv64")]
  {
    use crate::platform::caps::riscv;

    if _caps.has(riscv::ZVBC) {
      return clmul128_reduce_riscv_vector;
    }
    if _caps.has(riscv::ZBC) || _caps.has(riscv::ZBKC) {
      return clmul128_reduce_riscv_scalar;
    }
  }

  clmul128_reduce_portable
}

/// Combined 128×128 carryless multiply + Montgomery reduce.
///
/// Dispatches to PCLMULQDQ (x86_64) or PMULL (aarch64) when available,
/// otherwise uses the portable Pornin bmul64 + Karatsuba path.
pub(super) fn clmul128_reduce(a: u128, b: u128) -> u128 {
  let clmul = CLMUL128_REDUCE_DISPATCH.get_or_init(resolve_clmul128_reduce);
  clmul(a, b)
}

/// Precompute hash key powers [H, H^2, H^3, H^4] for 4-block wide processing.
///
/// Used by both GHASH and POLYVAL to enable the schoolbook-then-reduce
/// pattern: 4 parallel multiplies, 1 shared reduction.
#[cfg(any(
  test,
  all(
    feature = "aes-gcm-siv",
    any(
      target_arch = "x86_64",
      target_arch = "aarch64",
      target_arch = "powerpc64",
      target_arch = "s390x"
    )
  )
))]
pub(super) fn precompute_powers(h: u128) -> [u128; 4] {
  let h2 = clmul128_reduce(h, h);
  let h3 = clmul128_reduce(h2, h);
  let h4 = clmul128_reduce(h3, h);
  [h, h2, h3, h4]
}

/// Precompute hash key powers [H, H^2, ..., H^8] for 8-block GHASH windows.
#[cfg(feature = "aes-gcm")]
pub(super) fn precompute_powers_8(h: u128) -> [u128; 8] {
  let h2 = clmul128_reduce(h, h);
  let h3 = clmul128_reduce(h2, h);
  let h4 = clmul128_reduce(h3, h);
  let h5 = clmul128_reduce(h4, h);
  let h6 = clmul128_reduce(h5, h);
  let h7 = clmul128_reduce(h6, h);
  let h8 = clmul128_reduce(h7, h);
  [h, h2, h3, h4, h5, h6, h7, h8]
}

/// Precompute hash key powers [H, H^2, ..., H^16] for 16-block GHASH/POLYVAL windows.
#[cfg(any(
  all(feature = "aes-gcm", target_arch = "x86_64"),
  all(feature = "aes-gcm-siv", target_arch = "x86_64"),
  test
))]
pub(super) fn precompute_powers_16(h: u128) -> [u128; 16] {
  let mut powers = [0u128; 16];
  powers[0] = h;

  let mut i = 1usize;
  while i < powers.len() {
    powers[i] = clmul128_reduce(powers[i.strict_sub(1)], h);
    i = i.strict_add(1);
  }

  powers
}

/// Process 4 blocks through the hash accumulator in one shot.
///
/// Computes: (acc ^ b0) * H^4 ^ b1 * H^3 ^ b2 * H^2 ^ b3 * H
///
/// Dispatches to a target carryless-multiply aggregate when available,
/// otherwise falls back to 4 sequential `clmul128_reduce` calls.
#[cfg(any(
  feature = "aes-gcm",
  all(feature = "aes-gcm-siv", target_arch = "x86_64"),
  target_arch = "x86_64",
  test
))]
pub(super) fn accumulate_4blocks(
  acc: u128,
  h: u128,
  h_powers_rev: &[u128; 4], // [H^4, H^3, H^2, H]
  blocks: &[u128; 4],
) -> u128 {
  #[cfg(target_arch = "x86_64")]
  {
    if crate::platform::caps().has(crate::platform::caps::x86::VPCLMUL_READY) {
      // SAFETY: VPCLMULQDQ + AVX-512 availability verified via CPUID.
      return unsafe { vpclmul::aggregate_4blocks(acc, h_powers_rev, blocks) };
    }
    if crate::platform::caps().has(crate::platform::caps::x86::PCLMULQDQ) {
      // SAFETY: PCLMULQDQ availability verified via CPUID; x86_64 guarantees SSE2.
      return unsafe { x86_pclmul_aggregate_4blocks_inline(acc, h_powers_rev, blocks) };
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    if crate::platform::caps().has(crate::platform::caps::aarch64::PMULL) {
      // SAFETY: PMULL aggregate call because:
      // 1. Runtime caps confirmed PMULL support before entering the target-feature helper.
      // 2. `h_powers_rev` and `blocks` are fixed 4-lane arrays, matching the helper contract.
      return unsafe { aarch64_aggregate_4blocks_inline(acc, h_powers_rev, blocks) };
    }
  }

  #[cfg(target_arch = "powerpc64")]
  {
    if crate::platform::caps().has(crate::platform::caps::power::POWER8_CRYPTO) {
      // SAFETY: POWER8 crypto aggregate call because:
      // 1. Runtime caps confirmed POWER8_CRYPTO support before entering the target-feature helper.
      // 2. `h_powers_rev` and `blocks` are fixed 4-lane arrays, matching the helper contract.
      return unsafe { ppc_aggregate_4blocks_inline(acc, h_powers_rev, blocks) };
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    if crate::platform::caps().has(crate::platform::caps::s390x::VECTOR) {
      // SAFETY: z/Vector aggregate call because:
      // 1. Runtime caps confirmed vector-facility support before entering the target-feature helper.
      // 2. `h_powers_rev` and `blocks` are fixed 4-lane arrays, matching the helper contract.
      return unsafe { s390x_aggregate_4blocks_inline(acc, h_powers_rev, blocks) };
    }
  }

  #[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "s390x"
  )))]
  let _ = h_powers_rev;

  // Sequential fallback: equivalent to 4 individual block updates.
  let mut a = acc ^ blocks[0];
  a = clmul128_reduce(a, h);
  a ^= blocks[1];
  a = clmul128_reduce(a, h);
  a ^= blocks[2];
  a = clmul128_reduce(a, h);
  a ^= blocks[3];
  clmul128_reduce(a, h)
}

/// Process 16 blocks through the hash accumulator in one shot.
///
/// Computes `(acc ^ b0) * H^16 ^ b1 * H^15 ^ ... ^ b15 * H`.
#[cfg(any(all(feature = "aes-gcm-siv", target_arch = "x86_64"), test))]
pub(super) fn accumulate_16blocks(acc: u128, h: u128, h_powers_rev: &[u128; 16], blocks: &[u128; 16]) -> u128 {
  #[cfg(target_arch = "x86_64")]
  {
    if crate::platform::caps().has(crate::platform::caps::x86::VPCLMUL_READY) {
      // SAFETY: VPCLMUL aggregate call because:
      // 1. Runtime caps confirmed AVX-512 + VPCLMUL support before entering the target-feature helper.
      // 2. `h_powers_rev` and `blocks` are fixed 16-lane arrays, matching the helper contract.
      return unsafe { vpclmul::aggregate_16blocks(acc, h_powers_rev, blocks) };
    }
  }

  let _ = h;

  let mut a = clmul128_reduce(acc ^ blocks[0], h_powers_rev[0]);
  let mut i = 1usize;
  while i < blocks.len() {
    a ^= clmul128_reduce(blocks[i], h_powers_rev[i]);
    i = i.strict_add(1);
  }
  a
}

/// Process padded GCM-SIV POLYVAL input using x86 wide windows.
///
/// Full 16-byte blocks are folded with 16-block VPCLMUL windows when the caller
/// provides H powers through H^16, then 4-block windows, then scalar tails.
/// A final partial block is zero-padded as required by RFC 8452.
#[cfg(all(target_arch = "x86_64", feature = "aes-gcm-siv"))]
pub(super) fn accumulate_padded_x86(
  mut acc: u128,
  h: u128,
  h_powers_rev: &[u128; 4],
  h_powers_rev_16: Option<&[u128; 16]>,
  data: &[u8],
) -> u128 {
  let mut offset = 0usize;
  let has_vpclmul = crate::platform::caps().has(crate::platform::caps::x86::VPCLMUL_READY);

  if let Some(h16) = h_powers_rev_16 {
    while offset.strict_add(256) <= data.len() {
      if has_vpclmul {
        // SAFETY: x86 VPCLMUL byte aggregation because:
        // 1. Runtime caps confirmed AVX-512 + VPCLMUL support before entering the target-feature helper.
        // 2. `offset + 256 <= data.len()` proves the helper can read sixteen initialized blocks.
        acc = unsafe { x86_aggregate_16blocks_le_bytes_inline(acc, h16, data.as_ptr().add(offset)) };
      } else {
        let mut blocks = [0u128; 16];
        let mut i = 0usize;
        while i < 16 {
          let base = offset.strict_add(i.strict_mul(16));
          let mut block = [0u8; 16];
          block.copy_from_slice(&data[base..base.strict_add(16)]);
          blocks[i] = u128::from_le_bytes(block);
          i = i.strict_add(1);
        }
        acc = accumulate_16blocks(acc, h, h16, &blocks);
      }
      offset = offset.strict_add(256);
    }
  }

  while offset.strict_add(64) <= data.len() {
    if has_vpclmul {
      // SAFETY: x86 VPCLMUL byte aggregation because:
      // 1. Runtime caps confirmed AVX-512 + VPCLMUL support before entering the target-feature helper.
      // 2. `offset + 64 <= data.len()` proves the helper can read four initialized blocks.
      acc = unsafe { x86_aggregate_4blocks_le_bytes_inline(acc, h_powers_rev, data.as_ptr().add(offset)) };
    } else {
      let mut blocks = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&data[base..base.strict_add(16)]);
        blocks[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = accumulate_4blocks(acc, h, h_powers_rev, &blocks);
    }
    offset = offset.strict_add(64);
  }

  while offset.strict_add(16) <= data.len() {
    let mut block = [0u8; 16];
    block.copy_from_slice(&data[offset..offset.strict_add(16)]);
    acc ^= u128::from_le_bytes(block);
    acc = clmul128_reduce(acc, h);
    offset = offset.strict_add(16);
  }

  let remaining = data.len().strict_sub(offset);
  if remaining > 0 {
    let mut block = [0u8; 16];
    block[..remaining].copy_from_slice(&data[offset..]);
    acc ^= u128::from_le_bytes(block);
    acc = clmul128_reduce(acc, h);
  }

  acc
}

/// POLYVAL accumulator state.
///
/// The hash key H is stored as-is (no domain conversion). The "dot"
/// product uses a 2-pass fold-from-bottom reduction that is structurally
/// a Montgomery reduction, but for POLYVAL's polynomial and LE
/// representation this directly produces the correct field product.
#[cfg(feature = "aes-gcm-siv")]
pub(crate) struct Polyval {
  /// Hash key H (raw field element, no conversion).
  h: u128,
  /// Running accumulator.
  acc: u128,
}

#[cfg(feature = "aes-gcm-siv")]
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
    let (blocks, remainder) = data.as_chunks::<BLOCK_SIZE>();
    for block in blocks {
      self.update_block(block);
    }

    if !remainder.is_empty() {
      let mut block = [0u8; BLOCK_SIZE];
      block[..remainder.len()].copy_from_slice(remainder);
      self.update_block(&block);
    }
  }

  /// Finalize and return the 16-byte POLYVAL digest.
  #[inline]
  pub(crate) fn finalize(self) -> [u8; BLOCK_SIZE] {
    self.acc.to_le_bytes()
  }
}

#[cfg(feature = "aes-gcm-siv")]
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
  #[cfg(feature = "aes-gcm-siv")]
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
  #[cfg(feature = "aes-gcm-siv")]
  #[test]
  fn polyval_empty() {
    let h = [0x42u8; 16];
    let pv = Polyval::new(&h);
    assert_eq!(pv.finalize(), [0u8; 16]);
  }

  /// POLYVAL with zero key should always return zero.
  #[cfg(feature = "aes-gcm-siv")]
  #[test]
  fn polyval_zero_key() {
    let h = [0u8; 16];
    let x = [0xffu8; 16];
    let mut pv = Polyval::new(&h);
    pv.update_block(&x);
    assert_eq!(pv.finalize(), [0u8; 16]);
  }

  /// Verify that update_padded matches manual block-by-block.
  #[cfg(feature = "aes-gcm-siv")]
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

  /// Verify precompute_powers produces correct powers of H.
  #[test]
  fn precompute_powers_correct() {
    let h = u128::from_le_bytes(hex_to_16("25629347589242761d31f826ba4b757b"));
    let powers = precompute_powers(h);
    assert_eq!(powers[0], h, "powers[0] should be H");
    assert_eq!(powers[1], clmul128_reduce(h, h), "powers[1] should be H^2");
    assert_eq!(powers[2], clmul128_reduce(powers[1], h), "powers[2] should be H^3");
    assert_eq!(powers[3], clmul128_reduce(powers[2], h), "powers[3] should be H^4");
  }

  /// Verify precompute_powers_16 extends the same power chain to H^16.
  #[test]
  fn precompute_powers_16_correct() {
    let h = u128::from_le_bytes(hex_to_16("25629347589242761d31f826ba4b757b"));
    let powers = precompute_powers_16(h);
    assert_eq!(powers[0], h, "powers[0] should be H");

    let mut i = 1usize;
    while i < powers.len() {
      assert_eq!(
        powers[i],
        clmul128_reduce(powers[i.strict_sub(1)], h),
        "powers[{i}] should extend the H power chain"
      );
      i = i.strict_add(1);
    }
  }

  /// Verify accumulate_4blocks matches sequential block-by-block processing.
  #[test]
  fn accumulate_4blocks_matches_sequential() {
    let h_bytes = hex_to_16("25629347589242761d31f826ba4b757b");
    let h = u128::from_le_bytes(h_bytes);
    let powers = precompute_powers(h);
    let h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];

    let blocks = [
      u128::from_le_bytes(hex_to_16("4f4f95668c83dfb6401762bb2d01a262")),
      u128::from_le_bytes(hex_to_16("d1a24ddd2721d006bbe45f20d3c9f362")),
      u128::from_le_bytes(hex_to_16("0100000000000000000000000000000f")),
      u128::from_le_bytes(hex_to_16("abcdef0123456789abcdef0123456789")),
    ];
    let acc = 0x42u128;

    // Sequential: 4 individual updates.
    let mut seq = acc ^ blocks[0];
    seq = clmul128_reduce(seq, h);
    seq ^= blocks[1];
    seq = clmul128_reduce(seq, h);
    seq ^= blocks[2];
    seq = clmul128_reduce(seq, h);
    seq ^= blocks[3];
    seq = clmul128_reduce(seq, h);

    // Wide: one accumulate_4blocks call.
    let wide = accumulate_4blocks(acc, h, &h_powers_rev, &blocks);

    assert_eq!(wide, seq, "4-block aggregate must match sequential processing");
  }

  /// Verify accumulate_16blocks matches sequential block-by-block processing.
  #[test]
  fn accumulate_16blocks_matches_sequential() {
    let h_bytes = hex_to_16("25629347589242761d31f826ba4b757b");
    let h = u128::from_le_bytes(h_bytes);
    let powers = precompute_powers_16(h);
    let h_powers_rev = core::array::from_fn(|i| powers[15usize.strict_sub(i)]);
    let blocks = core::array::from_fn(|i| {
      let lane = (i as u128).wrapping_add(1);
      0x4f4f_9566_8c83_dfb6_4017_62bb_2d01_a262u128.wrapping_mul(lane)
        ^ 0xd1a2_4ddd_2721_d006_bbe4_5f20_d3c9_f362u128.rotate_left(i as u32)
    });
    let acc = 0x42u128;

    let mut seq = acc;
    let mut i = 0usize;
    while i < blocks.len() {
      seq ^= blocks[i];
      seq = clmul128_reduce(seq, h);
      i = i.strict_add(1);
    }

    let wide = accumulate_16blocks(acc, h, &h_powers_rev, &blocks);
    assert_eq!(wide, seq, "16-block aggregate must match sequential processing");
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  #[target_feature(enable = "avx2,avx512f,avx512vl,vpclmulqdq,pclmulqdq,sse2")]
  /// # Safety
  ///
  /// Caller must ensure AVX2 + AVX-512F + AVX-512VL + VPCLMULQDQ +
  /// PCLMULQDQ + SSE2 are available.
  unsafe fn x86_aggregate_8blocks_be_lanes_256_test_call(
    acc: u128,
    h_powers_rev: &[u128; 8],
    bytes: &[u8; 128],
  ) -> u128 {
    use core::arch::x86_64::*;

    // SAFETY: test-only x86 lane aggregation because:
    // 1. The caller verified the CPU features required by this target-feature helper.
    // 2. `bytes` is exactly 128 initialized bytes, so all four 32-byte loads are in bounds.
    // 3. The loaded lanes are passed directly to the helper under test.
    unsafe {
      let raw0 = _mm256_loadu_si256(bytes.as_ptr().cast());
      let raw1 = _mm256_loadu_si256(bytes.as_ptr().add(32).cast());
      let raw2 = _mm256_loadu_si256(bytes.as_ptr().add(64).cast());
      let raw3 = _mm256_loadu_si256(bytes.as_ptr().add(96).cast());
      x86_aggregate_8blocks_be_lanes_256_inline(acc, h_powers_rev, raw0, raw1, raw2, raw3)
    }
  }

  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm"))]
  #[test]
  fn x86_aggregate_8blocks_be_lanes_256_matches_sequential() {
    let required = crate::platform::caps::x86::VPCLMUL_READY | crate::platform::caps::x86::AVX2;
    if !crate::platform::caps().has(required) {
      return;
    }

    let h = u128::from_le_bytes(hex_to_16("25629347589242761d31f826ba4b757b"));
    let powers = precompute_powers_8(h);
    let h_powers_rev = core::array::from_fn(|i| powers[7usize.strict_sub(i)]);
    let acc = 0x1122_3344_5566_7788_99aa_bbcc_ddee_ff00u128;

    let mut bytes = [0u8; 128];
    let mut i = 0usize;
    while i < bytes.len() {
      bytes[i] = i.wrapping_mul(37).wrapping_add(19) as u8;
      i = i.strict_add(1);
    }

    let mut expected = acc;
    let mut offset = 0usize;
    while offset < bytes.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&bytes[offset..offset.strict_add(16)]);
      expected ^= u128::from_be_bytes(block);
      expected = clmul128_reduce(expected, h);
      offset = offset.strict_add(16);
    }

    // SAFETY: Runtime caps above confirmed AVX2 + VPCLMUL_READY before calling the target-feature
    // helper. The byte array and H-power table are fully initialized.
    let wide = unsafe { x86_aggregate_8blocks_be_lanes_256_test_call(acc, &h_powers_rev, &bytes) };
    assert_eq!(
      wide, expected,
      "8-block 256-bit VPCLMUL GHASH aggregate must match sequential fold"
    );
  }

  /// Verify the x86 padded wide path matches scalar POLYVAL over boundary sizes.
  #[cfg(all(target_arch = "x86_64", feature = "aes-gcm-siv"))]
  #[test]
  fn accumulate_padded_x86_matches_sequential_boundaries() {
    let h = u128::from_le_bytes(hex_to_16("25629347589242761d31f826ba4b757b"));
    let powers = precompute_powers_16(h);
    let h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    let h_powers_rev_16 = core::array::from_fn(|i| powers[15usize.strict_sub(i)]);
    let acc = 0x1122_3344_5566_7788_99aa_bbcc_ddee_ff00u128;
    let lengths = [
      0usize, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 257, 319, 320, 321,
    ];

    for len in lengths {
      let mut data = [0u8; 321];
      let mut i = 0usize;
      while i < len {
        data[i] = i.wrapping_mul(37).wrapping_add(19) as u8;
        i = i.strict_add(1);
      }

      let mut expected = acc;
      let mut offset = 0usize;
      while offset.strict_add(16) <= len {
        let mut block = [0u8; 16];
        block.copy_from_slice(&data[offset..offset.strict_add(16)]);
        expected ^= u128::from_le_bytes(block);
        expected = clmul128_reduce(expected, h);
        offset = offset.strict_add(16);
      }
      if offset < len {
        let mut block = [0u8; 16];
        block[..len.strict_sub(offset)].copy_from_slice(&data[offset..len]);
        expected ^= u128::from_le_bytes(block);
        expected = clmul128_reduce(expected, h);
      }

      let wide = accumulate_padded_x86(acc, h, &h_powers_rev, Some(&h_powers_rev_16), &data[..len]);
      assert_eq!(wide, expected, "padded x86 aggregate mismatch at len {len}");
    }
  }

  /// accumulate_4blocks with zero accumulator and zero blocks must produce zero.
  #[test]
  fn accumulate_4blocks_zeros() {
    let h = 0x42u128;
    let powers = precompute_powers(h);
    let h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    let blocks = [0u128; 4];
    let result = accumulate_4blocks(0, h, &h_powers_rev, &blocks);
    assert_eq!(result, 0);
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
