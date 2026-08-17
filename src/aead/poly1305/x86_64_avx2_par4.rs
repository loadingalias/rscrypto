use core::arch::x86_64::*;

use super::State;
use crate::{aead::AeadByteLengths, traits::ct};

/// Immediate byte for `_mm256_permute4x64_epi64`.
const fn imm8(x3: u8, x2: u8, x1: u8, x0: u8) -> i32 {
  (((x3) << 6) | ((x2) << 4) | ((x1) << 2) | (x0)) as i32
}

// ── Types ────────────────────────────────────────────────────────────────

/// Single 130-bit integer in five 26-bit limbs: `[_, _, _, l4, l3, l2, l1, l0]`.
#[derive(Clone, Copy)]
struct Aligned130(__m256i);

/// Precomputed multiplier: `a = [5r4, 5r3, 5r2, r4, r3, r2, r1, r0]`
/// and `a_5 = [5r1; 8]`.
#[derive(Clone, Copy)]
struct PrecomputedMultiplier {
  a: __m256i,
  a_5: __m256i,
}

/// Unreduced product of two 130-bit values (64-bit limbs).
/// `v1 = [_, _, _, t4]`, `v0 = [t3, t2, t1, t0]`.
#[derive(Clone, Copy)]
struct Unreduced130 {
  v0: __m256i,
  v1: __m256i,
}

/// Four 130-bit integers, 20 limbs across three `__m256i`.
#[derive(Clone, Copy)]
struct Aligned4x130 {
  v0: __m256i,
  v1: __m256i,
  v2: __m256i,
}

/// Unreduced product of four 130-bit multiplies (64-bit limbs).
#[derive(Clone, Copy)]
struct Unreduced4x130 {
  v0: __m256i,
  v1: __m256i,
  v2: __m256i,
  v3: __m256i,
  v4: __m256i,
}

/// Spaced multiplier `(R¹, R², R³, R⁴)` packed for lane-merge during finalization.
#[derive(Clone, Copy)]
struct SpacedMultiplier4x130 {
  v0: __m256i,
  v1: __m256i,
  r1: PrecomputedMultiplier,
}

// ── Aligned130 ───────────────────────────────────────────────────────────

impl Aligned130 {
  /// Pack five scalar 26-bit limbs into a `__m256i`.
  ///
  /// # Safety
  ///
  /// The caller must ensure that AVX2 is available.
  #[inline(always)]
  unsafe fn from_limbs(limbs: [u32; 5]) -> Self {
    // SAFETY: the caller guarantees AVX2 support.
    let lanes = unsafe {
      _mm256_setr_epi32(
        limbs[0].cast_signed(),
        limbs[1].cast_signed(),
        limbs[2].cast_signed(),
        limbs[3].cast_signed(),
        limbs[4].cast_signed(),
        0,
        0,
        0,
      )
    };
    Aligned130(lanes)
  }

  /// Extract five scalar 26-bit limbs.
  ///
  /// # Safety
  ///
  /// The caller must ensure that AVX2 is available.
  #[inline(always)]
  unsafe fn into_limbs(self) -> [u32; 5] {
    let mut buf = [0u32; 8];
    let dst = core::ptr::NonNull::from(&mut buf).cast::<__m256i>().as_ptr();
    // SAFETY: the caller guarantees AVX2 support. `dst` retains `buf`'s provenance and points to its entire writable
    // 32-byte range; the unaligned store has no stronger alignment requirement.
    unsafe { _mm256_storeu_si256(dst, self.0) };
    [buf[0], buf[1], buf[2], buf[3], buf[4]]
  }

  /// Load a full 16-byte block, split to 26-bit limbs, set hibit.
  ///
  /// AEAD-only: unconditionally sets the 2¹²⁸ high bit. Not suitable for raw
  /// Poly1305 where partial blocks omit the hibit.
  ///
  /// # Safety
  ///
  /// The caller must ensure that AVX2 is available.
  #[inline(always)]
  unsafe fn from_block(block: &[u8; 16]) -> Self {
    let src = core::ptr::NonNull::from(block).cast::<__m128i>().as_ptr();
    // SAFETY: the caller guarantees AVX2 support. `src` retains `block`'s provenance and exposes its complete
    // initialized 16-byte range; the load is unaligned. The nested split performs only AVX2 register operations.
    unsafe {
      Self::split_to_26bit(_mm256_or_si256(
        _mm256_and_si256(
          _mm256_castsi128_si256(_mm_loadu_si128(src)),
          _mm256_set_epi64x(0, 0, -1, -1),
        ),
        _mm256_set_epi64x(0, 1, 0, 0),
      ))
    }
  }

  /// Split a 130-bit integer (low 5 words) into 26-bit limbs.
  ///
  /// # Safety
  ///
  /// The caller must ensure that AVX2 is available.
  #[inline(always)]
  unsafe fn split_to_26bit(x: __m256i) -> Self {
    // SAFETY: the caller guarantees AVX2 support; all operations are register-only.
    unsafe {
      let xl = _mm256_sllv_epi32(x, _mm256_set_epi32(32, 32, 32, 24, 18, 12, 6, 0));
      let xh = _mm256_permutevar8x32_epi32(
        _mm256_srlv_epi32(x, _mm256_set_epi32(32, 32, 32, 2, 8, 14, 20, 26)),
        _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7),
      );
      Aligned130(_mm256_and_si256(
        _mm256_or_si256(xl, xh),
        _mm256_set_epi32(0, 0, 0, 0x3ff_ffff, 0x3ff_ffff, 0x3ff_ffff, 0x3ff_ffff, 0x3ff_ffff),
      ))
    }
  }

  /// Add two aligned values independently in each 32-bit lane.
  ///
  /// # Safety
  ///
  /// The caller must ensure that AVX2 is available.
  #[inline(always)]
  unsafe fn add(self, other: Aligned130) -> Aligned130 {
    // SAFETY: the caller guarantees AVX2 support.
    Aligned130(unsafe { _mm256_add_epi32(self.0, other.0) })
  }
}

// ── PrecomputedMultiplier ────────────────────────────────────────────────

impl PrecomputedMultiplier {
  /// Precompute the lane-packed multiplier from the kernel's bounded radix-2²⁶ representation.
  ///
  /// # Safety
  ///
  /// The caller must ensure that AVX2 is available.
  #[inline(always)]
  unsafe fn from_aligned(r: Aligned130) -> Self {
    // SAFETY: the caller guarantees AVX2 support; all operations are register-only.
    unsafe {
      // 5*R limbs: r + (r << 2) = r * 5
      let a_5 = _mm256_permutevar8x32_epi32(
        _mm256_add_epi32(r.0, _mm256_slli_epi32(r.0, 2)),
        _mm256_set_epi32(4, 3, 2, 1, 1, 1, 1, 1),
      );
      let a = _mm256_blend_epi32(r.0, a_5, 0b11100000);
      let a_5 = _mm256_permute2x128_si256(a_5, a_5, 0);
      PrecomputedMultiplier { a, a_5 }
    }
  }
}

// ── Single multiply: Aligned130 × PrecomputedMultiplier → Unreduced130 ──

/// Multiply one bounded radix-2²⁶ value by a precomputed multiplier.
///
/// The low five lanes of `x` must be below 2²⁸, and `r` must be derived from reduced multiplier limbs below 2²⁷.
/// These bounds keep the precomputed fivefold lanes below 2³⁰ and every five-product coefficient below 2⁶¹.
///
/// # Safety
///
/// The caller must ensure that AVX2 is available.
#[inline(always)]
unsafe fn mul_single(x: Aligned130, r: PrecomputedMultiplier) -> Unreduced130 {
  // SAFETY: the caller guarantees AVX2 support; the documented bounds keep every 64-bit lane sum below 2^61.
  unsafe {
    let x = x.0;
    let y = r.a;
    let z = r.a_5;

    // v0 = [t3, t2, t1, t0] — accumulate 5 products per limb.
    let mut v0 = _mm256_mul_epu32(
      _mm256_permutevar8x32_epi32(x, _mm256_set_epi64x(4, 3, 2, 1)),
      _mm256_permutevar8x32_epi32(y, _mm256_set_epi64x(7, 7, 7, 7)),
    );
    v0 = _mm256_add_epi64(
      v0,
      _mm256_mul_epu32(
        _mm256_permutevar8x32_epi32(x, _mm256_set_epi64x(3, 2, 1, 0)),
        _mm256_broadcastd_epi32(_mm256_castsi256_si128(y)),
      ),
    );
    v0 = _mm256_add_epi64(
      v0,
      _mm256_mul_epu32(
        _mm256_permutevar8x32_epi32(x, _mm256_set_epi64x(1, 1, 3, 3)),
        _mm256_permutevar8x32_epi32(y, _mm256_set_epi64x(2, 1, 6, 5)),
      ),
    );
    v0 = _mm256_add_epi64(
      v0,
      _mm256_mul_epu32(
        _mm256_permute4x64_epi64(x, imm8(1, 0, 0, 2)),
        _mm256_blend_epi32(_mm256_permutevar8x32_epi32(y, _mm256_set_epi64x(1, 2, 1, 1)), z, 0x03),
      ),
    );
    v0 = _mm256_add_epi64(
      v0,
      _mm256_mul_epu32(
        _mm256_permute4x64_epi64(x, imm8(0, 2, 2, 1)),
        _mm256_permutevar8x32_epi32(y, _mm256_set_epi64x(3, 6, 5, 6)),
      ),
    );

    // v1 = [_, _, _, t4]
    let mut v1 = _mm256_mul_epu32(
      _mm256_permutevar8x32_epi32(x, _mm256_set_epi64x(3, 2, 1, 0)),
      _mm256_permutevar8x32_epi32(y, _mm256_set_epi64x(1, 2, 3, 4)),
    );
    v1 = _mm256_add_epi64(v1, _mm256_permute4x64_epi64(v1, imm8(1, 0, 3, 2)));
    v1 = _mm256_add_epi64(v1, _mm256_permute4x64_epi64(v1, imm8(0, 0, 0, 1)));
    v1 = _mm256_add_epi64(v1, _mm256_mul_epu32(_mm256_permute4x64_epi64(x, imm8(0, 0, 0, 2)), y));

    Unreduced130 { v0, v1 }
  }
}

// ── Unreduced130 carry chain and reduction ───────────────────────────────

/// Carry: propagate bits >26 from v0 into v1.
///
/// The inputs must be bounded unreduced coefficients produced by this kernel's multiplication or preceding fold.
///
/// # Safety
///
/// The caller must ensure that AVX2 is available.
#[inline(always)]
unsafe fn adc_single(v1: __m256i, v0: __m256i) -> (__m256i, __m256i) {
  // SAFETY: the caller guarantees AVX2 support; the kernel bounds keep every propagated carry and lane sum in u64.
  unsafe {
    let v0 = _mm256_add_epi64(
      _mm256_and_si256(v0, _mm256_set_epi64x(-1, 0x3ff_ffff, 0x3ff_ffff, 0x3ff_ffff)),
      _mm256_permute4x64_epi64(
        _mm256_srlv_epi64(v0, _mm256_set_epi64x(64, 26, 26, 26)),
        imm8(2, 1, 0, 3),
      ),
    );
    let v1 = _mm256_add_epi64(
      v1,
      _mm256_permute4x64_epi64(_mm256_srli_epi64(v0, 26), imm8(2, 1, 0, 3)),
    );
    let chain = _mm256_and_si256(v0, _mm256_set_epi64x(0x3ff_ffff, -1, -1, -1));
    (v1, chain)
  }
}

/// Reduce modulo 2¹³⁰ − 5: fold top limb back into bottom.
///
/// The inputs must be the bounded carry-propagated representation returned by [`adc_single`].
///
/// # Safety
///
/// The caller must ensure that AVX2 is available.
#[inline(always)]
unsafe fn red_single(v1: __m256i, v0: __m256i) -> (__m256i, __m256i) {
  // SAFETY: the caller guarantees AVX2 support; the bounded top carry and its fivefold fold fit in every u64 lane.
  unsafe {
    let t = _mm256_srlv_epi64(v1, _mm256_set_epi64x(64, 64, 64, 26));
    let red_0 = _mm256_add_epi64(_mm256_add_epi64(v0, t), _mm256_slli_epi64(t, 2));
    let red_1 = _mm256_and_si256(v1, _mm256_set_epi64x(0, 0, 0, 0x3ff_ffff));
    (red_1, red_0)
  }
}

impl Unreduced130 {
  /// Carry-reduce this bounded convolution result into the kernel's radix-2²⁶ representation.
  ///
  /// # Safety
  ///
  /// The caller must ensure that AVX2 is available.
  #[inline(always)]
  unsafe fn reduce(self) -> Aligned130 {
    // SAFETY: the caller guarantees AVX2 support. `self` is a bounded kernel convolution, satisfying both nested
    // helpers' carry/fold contracts; the final operations only repack the resulting register lanes.
    unsafe {
      let (v1, v0) = adc_single(self.v1, self.v0);
      let (v1, v0) = red_single(v1, v0);
      let (v1, v0) = adc_single(v1, v0);
      // Switch from 64-bit to 32-bit limbs.
      Aligned130(_mm256_blend_epi32(
        _mm256_permutevar8x32_epi32(v0, _mm256_set_epi32(0, 6, 4, 0, 6, 4, 2, 0)),
        _mm256_permutevar8x32_epi32(v1, _mm256_set_epi32(0, 6, 4, 0, 6, 4, 2, 0)),
        0x90,
      ))
    }
  }
}

// ── Aligned4x130 ────────────────────────────────────────────────────────

impl Aligned4x130 {
  /// Load four consecutive full blocks for parallel processing.
  ///
  /// # Safety
  ///
  /// The caller must ensure that AVX2 is available.
  #[inline(always)]
  unsafe fn from_blocks(src: &[[u8; 16]; 4]) -> Self {
    let ptr = core::ptr::NonNull::from(src).cast::<__m256i>().as_ptr();
    // SAFETY: the caller guarantees AVX2 support. `ptr` retains `src`'s provenance; `src` contains 64 initialized
    // readable bytes, so the unaligned 32-byte loads at offsets 0 and 32 are valid. The nested helper is register-only.
    unsafe {
      let blocks_01 = _mm256_loadu_si256(ptr);
      let blocks_23 = _mm256_loadu_si256(ptr.add(1));
      Self::from_loaded_blocks(blocks_01, blocks_23)
    }
  }

  /// Interleave 4 blocks into 20 packed 26-bit limbs across 3 vectors.
  ///
  /// `blocks_01` and `blocks_23` must contain four consecutive full blocks in that order.
  ///
  /// # Safety
  ///
  /// The caller must ensure that AVX2 is available.
  #[inline(always)]
  unsafe fn from_loaded_blocks(blocks_01: __m256i, blocks_23: __m256i) -> Self {
    // SAFETY: the caller guarantees AVX2 support; all operations are register-only.
    unsafe {
      let mask_26 = _mm256_set1_epi32(0x3ff_ffff);
      let set_hibit = _mm256_set1_epi32(1 << 24);

      let a0 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi64(blocks_01, blocks_23), imm8(3, 1, 2, 0));
      let a1 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi64(blocks_01, blocks_23), imm8(3, 1, 2, 0));

      let v2 = _mm256_or_si256(_mm256_srli_epi64(a0, 40), set_hibit);
      let a2 = _mm256_or_si256(_mm256_srli_epi64(a1, 46), _mm256_slli_epi64(a0, 18));

      let v1 = _mm256_and_si256(_mm256_blend_epi32(_mm256_srli_epi64(a1, 26), a2, 0xAA), mask_26);
      let v0 = _mm256_and_si256(_mm256_blend_epi32(a1, _mm256_slli_epi64(a2, 26), 0xAA), mask_26);

      Aligned4x130 { v0, v1, v2 }
    }
  }

  /// Add four aligned values independently in each 32-bit lane.
  ///
  /// Both inputs must satisfy the kernel's loose-limb bounds, which keep every lane sum in `u32`.
  ///
  /// # Safety
  ///
  /// The caller must ensure that AVX2 is available.
  #[inline(always)]
  unsafe fn add(self, other: Aligned4x130) -> Aligned4x130 {
    // SAFETY: the caller guarantees AVX2 support; the documented bounds keep all three lane sums in u32.
    unsafe {
      Aligned4x130 {
        v0: _mm256_add_epi32(self.v0, other.v0),
        v1: _mm256_add_epi32(self.v1, other.v1),
        v2: _mm256_add_epi32(self.v2, other.v2),
      }
    }
  }
}

// ── 4-way parallel multiply ──────────────────────────────────────────────

/// Multiply 4 values by the same R: `(x0·R, x1·R, x2·R, x3·R)`.
///
/// `x` and `r` must satisfy the kernel's loose-radix bounds. Those bounds keep every five-product coefficient below
/// 2⁶¹.
///
/// # Safety
///
/// The caller must ensure that AVX2 is available.
#[inline(always)]
unsafe fn mul_4x130(x: &Aligned4x130, r: PrecomputedMultiplier) -> Unreduced4x130 {
  // SAFETY: the caller guarantees AVX2 support; the documented bounds keep every 64-bit lane sum below 2^61.
  unsafe {
    let mut x = *x;
    let y = r.a;
    let z = r.a_5;
    let ord = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);

    let mut t0 = _mm256_permute4x64_epi64(y, imm8(0, 0, 0, 0));
    let mut t1 = _mm256_permute4x64_epi64(y, imm8(1, 1, 1, 1));

    let mut v0 = _mm256_mul_epu32(x.v0, t0);
    let mut v1 = _mm256_mul_epu32(x.v1, t0);
    let mut v4 = _mm256_mul_epu32(x.v2, t0);
    let mut v2 = _mm256_mul_epu32(x.v0, t1);
    let mut v3 = _mm256_mul_epu32(x.v1, t1);

    t0 = _mm256_permutevar8x32_epi32(t0, ord);
    t1 = _mm256_permutevar8x32_epi32(t1, ord);

    v1 = _mm256_add_epi64(v1, _mm256_mul_epu32(x.v0, t0));
    v2 = _mm256_add_epi64(v2, _mm256_mul_epu32(x.v1, t0));
    v3 = _mm256_add_epi64(v3, _mm256_mul_epu32(x.v0, t1));
    v4 = _mm256_add_epi64(v4, _mm256_mul_epu32(x.v1, t1));

    let mut t2 = _mm256_permute4x64_epi64(y, imm8(2, 2, 2, 2));
    v4 = _mm256_add_epi64(v4, _mm256_mul_epu32(x.v0, t2));

    x.v0 = _mm256_permutevar8x32_epi32(x.v0, ord);
    x.v1 = _mm256_permutevar8x32_epi32(x.v1, ord);
    t2 = _mm256_permutevar8x32_epi32(t2, ord);

    v0 = _mm256_add_epi64(v0, _mm256_mul_epu32(x.v1, t2));
    v1 = _mm256_add_epi64(v1, _mm256_mul_epu32(x.v2, t2));
    v3 = _mm256_add_epi64(v3, _mm256_mul_epu32(x.v0, t0));
    v4 = _mm256_add_epi64(v4, _mm256_mul_epu32(x.v1, t0));

    t0 = _mm256_permutevar8x32_epi32(t0, ord);
    t1 = _mm256_permutevar8x32_epi32(t1, ord);

    v2 = _mm256_add_epi64(v2, _mm256_mul_epu32(x.v0, t0));
    v3 = _mm256_add_epi64(v3, _mm256_mul_epu32(x.v1, t0));
    v4 = _mm256_add_epi64(v4, _mm256_mul_epu32(x.v0, t1));

    t0 = _mm256_permute4x64_epi64(y, imm8(3, 3, 3, 3));

    v0 = _mm256_add_epi64(v0, _mm256_mul_epu32(x.v0, t0));
    v1 = _mm256_add_epi64(v1, _mm256_mul_epu32(x.v1, t0));
    v2 = _mm256_add_epi64(v2, _mm256_mul_epu32(x.v2, t0));

    t0 = _mm256_permutevar8x32_epi32(t0, ord);

    v1 = _mm256_add_epi64(v1, _mm256_mul_epu32(x.v0, t0));
    v2 = _mm256_add_epi64(v2, _mm256_mul_epu32(x.v1, t0));
    v3 = _mm256_add_epi64(v3, _mm256_mul_epu32(x.v2, t0));

    x.v1 = _mm256_permutevar8x32_epi32(x.v1, ord);

    v0 = _mm256_add_epi64(v0, _mm256_mul_epu32(x.v1, t0));
    v0 = _mm256_add_epi64(v0, _mm256_mul_epu32(x.v2, z));

    Unreduced4x130 { v0, v1, v2, v3, v4 }
  }
}

// ── Spaced multiply ─────────────────────────────────────────────────────

/// Multiply lane i by R^(4−i): `(x0·R⁴, x1·R³, x2·R², x3·R¹)`.
///
/// `x` and `m` must satisfy the kernel's loose-radix bounds and represent matching streams and powers of one key.
/// Those bounds keep every five-product coefficient below 2⁶¹.
///
/// # Safety
///
/// The caller must ensure that AVX2 is available.
#[inline(always)]
unsafe fn mul_spaced(x: Aligned4x130, m: SpacedMultiplier4x130) -> Unreduced4x130 {
  // SAFETY: the caller guarantees AVX2 support; the documented bounds keep every 64-bit lane sum below 2^61.
  unsafe {
    let mut x = x;
    let r1 = m.r1.a;

    let v0u = _mm256_unpacklo_epi32(m.v0, m.v1);
    let v1u = _mm256_unpackhi_epi32(m.v0, m.v1);

    let ord_a = _mm256_set_epi32(1, 0, 6, 7, 2, 0, 3, 1);
    let m_r_0 = _mm256_blend_epi32(
      _mm256_permutevar8x32_epi32(r1, ord_a),
      _mm256_permutevar8x32_epi32(v0u, ord_a),
      0b00111111,
    );
    let ord_b = _mm256_set_epi32(3, 2, 4, 5, 2, 0, 3, 1);
    let m_r_2 = _mm256_blend_epi32(
      _mm256_permutevar8x32_epi32(r1, ord_b),
      _mm256_permutevar8x32_epi32(v1u, ord_b),
      0b00111111,
    );
    let ord_c = _mm256_set_epi32(1, 4, 6, 6, 2, 4, 3, 5);
    let m_r_4 = _mm256_blend_epi32(
      _mm256_blend_epi32(
        _mm256_permutevar8x32_epi32(r1, ord_c),
        _mm256_permutevar8x32_epi32(v1u, ord_c),
        0b00010000,
      ),
      _mm256_permutevar8x32_epi32(v0u, ord_c),
      0b00101111,
    );

    let mut v0 = _mm256_mul_epu32(x.v0, m_r_0);
    let mut v1 = _mm256_mul_epu32(x.v1, m_r_0);
    let mut v2 = _mm256_mul_epu32(x.v0, m_r_2);
    let mut v3 = _mm256_mul_epu32(x.v1, m_r_2);
    let mut v4 = _mm256_mul_epu32(x.v0, m_r_4);

    let swap = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);
    let m_r_1 = _mm256_permutevar8x32_epi32(m_r_0, swap);
    let m_r_3 = _mm256_permutevar8x32_epi32(m_r_2, swap);

    v1 = _mm256_add_epi64(v1, _mm256_mul_epu32(x.v0, m_r_1));
    v2 = _mm256_add_epi64(v2, _mm256_mul_epu32(x.v1, m_r_1));
    v3 = _mm256_add_epi64(v3, _mm256_mul_epu32(x.v0, m_r_3));
    v4 = _mm256_add_epi64(v4, _mm256_mul_epu32(x.v1, m_r_3));
    v4 = _mm256_add_epi64(v4, _mm256_mul_epu32(x.v2, m_r_0));

    x.v0 = _mm256_permutevar8x32_epi32(x.v0, swap);

    v2 = _mm256_add_epi64(v2, _mm256_mul_epu32(x.v0, m_r_0));
    v3 = _mm256_add_epi64(v3, _mm256_mul_epu32(x.v0, m_r_1));
    v4 = _mm256_add_epi64(v4, _mm256_mul_epu32(x.v0, m_r_2));

    let m_5r_3 = _mm256_add_epi32(m_r_3, _mm256_slli_epi32(m_r_3, 2));
    let m_5r_4 = _mm256_add_epi32(m_r_4, _mm256_slli_epi32(m_r_4, 2));

    v0 = _mm256_add_epi64(v0, _mm256_mul_epu32(x.v0, m_5r_3));
    v0 = _mm256_add_epi64(v0, _mm256_mul_epu32(x.v1, m_5r_4));
    v1 = _mm256_add_epi64(v1, _mm256_mul_epu32(x.v0, m_5r_4));
    v2 = _mm256_add_epi64(v2, _mm256_mul_epu32(x.v2, m_5r_3));
    v3 = _mm256_add_epi64(v3, _mm256_mul_epu32(x.v2, m_5r_4));

    x.v1 = _mm256_permutevar8x32_epi32(x.v1, swap);

    v1 = _mm256_add_epi64(v1, _mm256_mul_epu32(x.v1, m_5r_3));
    v2 = _mm256_add_epi64(v2, _mm256_mul_epu32(x.v1, m_5r_4));
    v3 = _mm256_add_epi64(v3, _mm256_mul_epu32(x.v1, m_r_0));
    v4 = _mm256_add_epi64(v4, _mm256_mul_epu32(x.v1, m_r_1));

    let m_5r_1 = _mm256_permutevar8x32_epi32(m_5r_4, swap);
    let m_5r_2 = _mm256_permutevar8x32_epi32(m_5r_3, swap);

    v0 = _mm256_add_epi64(v0, _mm256_mul_epu32(x.v1, m_5r_2));
    v0 = _mm256_add_epi64(v0, _mm256_mul_epu32(x.v2, m_5r_1));
    v1 = _mm256_add_epi64(v1, _mm256_mul_epu32(x.v2, m_5r_2));

    Unreduced4x130 { v0, v1, v2, v3, v4 }
  }
}

// ── Unreduced4x130 ──────────────────────────────────────────────────────

impl Unreduced4x130 {
  /// Carry-reduce 4 values in parallel back to 26-bit limbs.
  ///
  /// `self` must contain bounded convolution coefficients produced by [`mul_4x130`] or [`mul_spaced`].
  ///
  /// # Safety
  ///
  /// The caller must ensure that AVX2 is available.
  #[inline(always)]
  unsafe fn reduce(self) -> Aligned4x130 {
    // SAFETY: the caller guarantees AVX2 support. The bounded coefficients keep every carry, fivefold fold, and lane
    // sum in u64; both closures and the final packing are register-only.
    unsafe {
      let mask_26 = _mm256_set1_epi64x(0x3ff_ffff);

      let adc = |x1: __m256i, x0: __m256i| -> (__m256i, __m256i) {
        let y1 = _mm256_add_epi64(x1, _mm256_srli_epi64(x0, 26));
        let y0 = _mm256_and_si256(x0, mask_26);
        (y1, y0)
      };
      let red = |x4: __m256i, x0: __m256i| -> (__m256i, __m256i) {
        let y0 = _mm256_add_epi64(x0, _mm256_mul_epu32(_mm256_srli_epi64(x4, 26), _mm256_set1_epi64x(5)));
        let y4 = _mm256_and_si256(x4, mask_26);
        (y4, y0)
      };

      let (r1, r0) = adc(self.v1, self.v0);
      let (r4, r3) = adc(self.v4, self.v3);
      let (r2, r1) = adc(self.v2, r1);
      let (r4, r0) = red(r4, r0);
      let (r3, r2) = adc(r3, r2);
      let (r1, r0) = adc(r1, r0);
      let (r4, r3) = adc(r4, r3);

      Aligned4x130 {
        v0: _mm256_blend_epi32(r0, _mm256_slli_epi64(r2, 32), 0b10101010),
        v1: _mm256_blend_epi32(r1, _mm256_slli_epi64(r3, 32), 0b10101010),
        v2: r4,
      }
    }
  }

  /// Horizontal sum of 4 lanes into a single `Unreduced130`.
  ///
  /// `self` must contain the bounded four-stream convolution produced by [`mul_spaced`]. The resulting coefficient
  /// sums remain below 2⁶³.
  ///
  /// # Safety
  ///
  /// The caller must ensure that AVX2 is available.
  #[inline(always)]
  unsafe fn sum(self) -> Unreduced130 {
    // SAFETY: the caller guarantees AVX2 support; the documented bounds keep every horizontal lane sum in u64.
    unsafe {
      let lo01 = _mm256_add_epi64(
        _mm256_unpackhi_epi64(self.v0, self.v1),
        _mm256_unpacklo_epi64(self.v0, self.v1),
      );
      let lo23 = _mm256_add_epi64(
        _mm256_unpackhi_epi64(self.v2, self.v3),
        _mm256_unpacklo_epi64(self.v2, self.v3),
      );
      let v0 = _mm256_add_epi64(
        _mm256_inserti128_si256(lo01, _mm256_castsi256_si128(lo23), 1),
        _mm256_inserti128_si256(lo23, _mm256_extracti128_si256(lo01, 1), 0),
      );
      let v4 = _mm256_add_epi64(self.v4, _mm256_permute4x64_epi64(self.v4, imm8(1, 0, 3, 2)));
      let v1 = _mm256_add_epi64(v4, _mm256_permute4x64_epi64(v4, imm8(0, 0, 0, 1)));
      Unreduced130 { v0, v1 }
    }
  }
}

// ── SpacedMultiplier4x130 ───────────────────────────────────────────────

impl SpacedMultiplier4x130 {
  /// Compute `(multiplier, R⁴)` from `(R¹, R²)`.
  ///
  /// `r1` and `r2` must be reduced consecutive powers of the same clamped Poly1305 key.
  ///
  /// # Safety
  ///
  /// The caller must ensure that AVX2 is available.
  #[inline(always)]
  unsafe fn new(r1: PrecomputedMultiplier, r2: PrecomputedMultiplier) -> (Self, PrecomputedMultiplier) {
    // SAFETY: the caller guarantees AVX2 support. Matching bounded powers satisfy both nested multiplication/reduction
    // contracts; the remaining operations only pack those register values.
    unsafe {
      let r3 = mul_single(Aligned130(r2.a), r1).reduce();
      let r4 = mul_single(Aligned130(r2.a), r2).reduce();

      let v0 = _mm256_blend_epi32(
        r3.0,
        _mm256_permutevar8x32_epi32(r2.a, _mm256_set_epi32(4, 3, 1, 0, 0, 0, 0, 0)),
        0b11100000,
      );
      let v1 = _mm256_blend_epi32(
        r4.0,
        _mm256_permutevar8x32_epi32(r2.a, _mm256_set_epi32(4, 2, 0, 0, 0, 0, 0, 0)),
        0b11100000,
      );

      let m = SpacedMultiplier4x130 { v0, v1, r1 };
      (m, PrecomputedMultiplier::from_aligned(r4))
    }
  }
}

// ── Top-level AEAD kernel ───────────────────────────────────────────────

/// Accumulated 4-way polynomial state.
#[derive(Clone, Copy)]
struct Par4State {
  poly: Aligned4x130,
  spaced: SpacedMultiplier4x130,
  r4: PrecomputedMultiplier,
}

/// Authenticate `(aad, ciphertext)` using the AVX2 4-way Poly1305 kernel.
///
/// Uses its own AVX2 kernel — ignores the per-block `ComputeBlockFn` dispatch.
/// `lengths` must encode the lengths of `aad` and `ciphertext`.
///
/// # Safety
///
/// The caller must ensure that AVX2 is available.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn authenticate_aead_par4(
  aad: &[u8],
  ciphertext: &[u8],
  key: &[u8; 32],
  lengths: AeadByteLengths,
) -> [u8; 16] {
  let state = State::new(key);

  // Precompute R¹, R² as AVX2 multipliers.
  // SAFETY: the caller guarantees AVX2. `State::new` produces a clamped multiplier, and each nested helper preserves
  // the bounded representation while deriving consecutive powers from that same key.
  let (r1, r2) = unsafe {
    let r = Aligned130::from_limbs(state.r);
    let r1 = PrecomputedMultiplier::from_aligned(r);
    let r2 = PrecomputedMultiplier::from_aligned(mul_single(Aligned130(r1.a), r1).reduce());
    (r1, r2)
  };

  // 4-block accumulator (initialized on first 4-block group).
  let mut acc: Option<Par4State> = None;
  let mut cached = [[0u8; 16]; 4];
  let mut num_cached = 0usize;

  // Reimplements padded-segment logic from `update_padded_segment` for 4-way batching.
  for segment in [aad, ciphertext] {
    let mut chunks = segment.chunks_exact(16);
    for chunk in &mut chunks {
      let mut block = [0u8; 16];
      block.copy_from_slice(chunk);
      // SAFETY: the caller guarantees AVX2. The loop produces consecutive full blocks; `push_block` preserves
      // `num_cached < 4` and keeps the cache, powers, and accumulator in the same key stream.
      num_cached = unsafe { push_block(block, &mut cached, num_cached, &mut acc, r1, r2) };
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
      let mut block = [0u8; 16];
      block[..rem.len()].copy_from_slice(rem);
      // SAFETY: the caller guarantees AVX2. Zero padding completes the segment's final full AEAD block, and the
      // established cache and matching-key invariants still hold.
      num_cached = unsafe { push_block(block, &mut cached, num_cached, &mut acc, r1, r2) };
    }
  }

  // Process lengths block.
  let length_block = lengths.to_le_bytes_block();
  // SAFETY: the caller guarantees AVX2. The length encoding is the final full AEAD block, and the established cache
  // and matching-key invariants still hold.
  num_cached = unsafe { push_block(length_block, &mut cached, num_cached, &mut acc, r1, r2) };

  // Finalize: merge 4 lanes, process remaining blocks.
  let mut p: Option<Aligned130> = acc.map(|s| {
    // SAFETY: the caller guarantees AVX2, and `accumulate_4_blocks` constructed `poly` and `spaced` from matching
    // bounded powers in this key stream.
    unsafe { mul_spaced(s.poly, s.spaced).sum().reduce() }
  });

  // 2-block tail.
  if num_cached >= 2 {
    // SAFETY: the caller guarantees AVX2. The cache invariant proves slots 0 and 1 are consecutive full blocks;
    // `p`, `r2`, and `r1` are bounded values from this key stream, and the two products preserve block order.
    unsafe {
      let mut c0 = Aligned130::from_block(&cached[0]);
      let c1 = Aligned130::from_block(&cached[1]);
      if let Some(pv) = p {
        c0 = c0.add(pv);
      }
      let a = mul_single(c0, r2);
      let b = mul_single(c1, r1);
      p = Some(
        Unreduced130 {
          v0: _mm256_add_epi64(a.v0, b.v0),
          v1: _mm256_add_epi64(a.v1, b.v1),
        }
        .reduce(),
      );
      cached[0] = cached[2];
      num_cached = num_cached.strict_sub(2);
    }
  }

  // 1-block tail.
  if num_cached == 1 {
    // SAFETY: the caller guarantees AVX2. The cache invariant proves slot 0 is the final full block, while `p` and
    // `r1` are bounded values from the same key stream.
    unsafe {
      let mut c = Aligned130::from_block(&cached[0]);
      if let Some(pv) = p {
        c = c.add(pv);
      }
      p = Some(mul_single(c, r1).reduce());
    }
  }

  // Convert AVX2 result back to scalar and finalize.
  let mut final_state = state;
  if let Some(pv) = p {
    // SAFETY: the caller guarantees AVX2, and `pv` is the bounded result of this kernel's reduction chain.
    final_state.h = unsafe { pv.into_limbs() };
  }
  let tag = final_state.finalize();
  ct::zeroize(cached.as_flattened_mut());
  tag
}

/// Cache one block; flush a 4-block group when full. Returns updated `num_cached`.
///
/// # Safety
///
/// The caller must ensure that AVX2 is available, `num_cached < 4`, the occupied cache prefix and `block` are
/// consecutive full AEAD-padded blocks, and `r1`, `r2`, and any accumulator belong to the same bounded key stream.
#[inline(always)]
unsafe fn push_block(
  block: [u8; 16],
  cached: &mut [[u8; 16]; 4],
  num_cached: usize,
  acc: &mut Option<Par4State>,
  r1: PrecomputedMultiplier,
  r2: PrecomputedMultiplier,
) -> usize {
  cached[num_cached] = block;
  let n = num_cached.strict_add(1);
  if n == 4 {
    // SAFETY: the caller guarantees AVX2 and a matching cache, multiplier, and accumulator. `n == 4` proves that
    // this write completed the four-block group before it is accumulated.
    unsafe { accumulate_4_blocks(cached, acc, r1, r2) };
    0
  } else {
    n
  }
}

/// Process a full 4-block group into the parallel accumulator.
///
/// # Safety
///
/// The caller must ensure that AVX2 is available, `cached` contains four consecutive full AEAD-padded blocks, and
/// `r1`, `r2`, and any existing accumulator are bounded powers and state derived from the same clamped Poly1305 key.
#[inline(always)]
unsafe fn accumulate_4_blocks(
  cached: &[[u8; 16]; 4],
  acc: &mut Option<Par4State>,
  r1: PrecomputedMultiplier,
  r2: PrecomputedMultiplier,
) {
  // SAFETY: the caller guarantees AVX2 and the matching-key block, multiplier, and accumulator invariants required
  // by each nested helper. All operations are fixed-width loads or register-only arithmetic.
  unsafe {
    let blocks = Aligned4x130::from_blocks(cached);
    if let Some(ref mut s) = *acc {
      s.poly = mul_4x130(&s.poly, s.r4).reduce().add(blocks);
    } else {
      let (spaced, r4) = SpacedMultiplier4x130::new(r1, r2);
      *acc = Some(Par4State {
        poly: blocks,
        spaced,
        r4,
      });
    }
  }
}
