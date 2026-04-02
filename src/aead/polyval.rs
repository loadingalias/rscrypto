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

  /// Montgomery reduction of a 256-bit product [lo:hi] modulo POLYVAL's polynomial.
  ///
  /// Equivalent to the portable `mont_reduce` but uses SSE2 lane-parallel shifts.
  #[inline(always)]
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
    unsafe {
      let a_lo = a as u64;
      let a_hi = (a >> 64) as u64;
      let b_lo = b as u64;
      let b_hi = (b >> 64) as u64;

      // Schoolbook 128×128 → 256-bit product (4 PMULL instructions).
      let ll = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
      let hh = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));
      let lh = vreinterpretq_u64_p128(vmull_p64(a_lo, b_hi));
      let hl = vreinterpretq_u64_p128(vmull_p64(a_hi, b_lo));
      let mid = veorq_u64(lh, hl);

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

  /// Montgomery reduction of [lo:hi] modulo POLYVAL's polynomial.
  ///
  /// Polynomial: x^128 + x^127 + x^126 + x^121 + 1
  /// Shifts: 63, 62, 57 (complement of taps relative to 64-bit boundary)
  ///
  /// Uses NEON lane-parallel shifts (`vshlq_n_u64`, `vshrq_n_u64`) and
  /// `vextq_u64` for cross-lane propagation — structurally identical to
  /// the SSE2 `mont_reduce_sse2` path.
  #[inline(always)]
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

  /// Combined 128×128 carryless multiply + Montgomery reduce using VGFM.
  ///
  /// Karatsuba decomposition with 3 VGFM multiplies (vs 4 schoolbook).
  ///
  /// # Safety
  /// Caller must ensure the z/Vector facility is available.
  #[target_feature(enable = "vector")]
  pub(super) unsafe fn clmul128_reduce(a: u128, b: u128) -> u128 {
    // SAFETY: target_feature gate guarantees z/Vector availability.
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

      let w0 = v0_lo;
      let w1 = v0_hi ^ mid_lo;
      let w2 = v1_lo ^ mid_hi;
      let w3 = v1_hi;

      super::mont_reduce([w0, w1, w2, w3])
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
// x86_64 VPCLMULQDQ wide backend (4-block aggregate)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod vpclmul {
  use core::arch::x86_64::*;

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
  #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
  pub(super) unsafe fn aggregate_4blocks(
    acc: u128,
    h_powers_rev: &[u128; 4], // [H^4, H^3, H^2, H]
    blocks: &[u128; 4],       // [b0, b1, b2, b3] in correct domain
  ) -> u128 {
    // SAFETY: target_feature gate guarantees all required ISA extensions.
    unsafe {
      // XOR accumulator into the first block.
      let mut block_data = *blocks;
      block_data[0] ^= acc;
      let data = _mm512_loadu_si512(block_data.as_ptr().cast());

      // Load H powers [H^4, H^3, H^2, H].
      let h_vec = _mm512_loadu_si512(h_powers_rev.as_ptr().cast());

      // 4-way parallel schoolbook: 4 VPCLMULQDQ instructions.
      let lo = _mm512_clmulepi64_epi128(data, h_vec, 0x00); // a_lo * b_lo
      let hi = _mm512_clmulepi64_epi128(data, h_vec, 0x11); // a_hi * b_hi
      let m1 = _mm512_clmulepi64_epi128(data, h_vec, 0x10); // a_hi * b_lo
      let m2 = _mm512_clmulepi64_epi128(data, h_vec, 0x01); // a_lo * b_hi
      let mid = _mm512_xor_si512(m1, m2);

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
}

// ---------------------------------------------------------------------------
// Combined multiply + reduce with hardware dispatch
// ---------------------------------------------------------------------------

/// Combined 128×128 carryless multiply + Montgomery reduce.
///
/// Dispatches to PCLMULQDQ (x86_64) or PMULL (aarch64) when available,
/// otherwise uses the portable Pornin bmul64 + Karatsuba path.
pub(super) fn clmul128_reduce(a: u128, b: u128) -> u128 {
  #[cfg(target_arch = "x86_64")]
  {
    if crate::platform::caps().has(crate::platform::caps::x86::PCLMULQDQ) {
      // SAFETY: PCLMULQDQ availability verified via CPUID.
      return unsafe { pclmul::clmul128_reduce(a, b) };
    }
  }
  #[cfg(target_arch = "aarch64")]
  {
    if crate::platform::caps().has(crate::platform::caps::aarch64::PMULL) {
      // SAFETY: PMULL availability verified via HWCAP.
      return unsafe { pmull::clmul128_reduce(a, b) };
    }
  }
  #[cfg(target_arch = "s390x")]
  {
    if crate::platform::caps().has(crate::platform::caps::s390x::VECTOR) {
      // SAFETY: z/Vector availability verified via STFLE/HWCAP.
      return unsafe { s390x_vgfm::clmul128_reduce(a, b) };
    }
  }
  #[cfg(target_arch = "powerpc64")]
  {
    if crate::platform::caps().has(crate::platform::caps::power::POWER8_CRYPTO) {
      // SAFETY: POWER8 crypto availability verified via HWCAP.
      return unsafe { ppc_vpmsum::clmul128_reduce(a, b) };
    }
  }
  #[cfg(target_arch = "riscv64")]
  {
    if crate::platform::caps().has(crate::platform::caps::riscv::ZVBC) {
      // SAFETY: Zvbc availability verified via feature detection.
      return unsafe { rv_clmul::clmul128_reduce(a, b) };
    }
  }
  mont_reduce(clmul128(a, b))
}

/// Precompute hash key powers [H, H^2, H^3, H^4] for 4-block wide processing.
///
/// Used by both GHASH and POLYVAL to enable the schoolbook-then-reduce
/// pattern: 4 parallel multiplies, 1 shared reduction.
pub(super) fn precompute_powers(h: u128) -> [u128; 4] {
  let h2 = clmul128_reduce(h, h);
  let h3 = clmul128_reduce(h2, h);
  let h4 = clmul128_reduce(h3, h);
  [h, h2, h3, h4]
}

/// Process 4 blocks through the hash accumulator in one shot.
///
/// Computes: (acc ^ b0) * H^4 ^ b1 * H^3 ^ b2 * H^2 ^ b3 * H
///
/// Dispatches to VPCLMULQDQ (x86_64) when available, otherwise falls
/// back to 4 sequential `clmul128_reduce` calls.
#[cfg(any(target_arch = "x86_64", test))]
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
  }
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
