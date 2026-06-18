#![allow(clippy::indexing_slicing)] // Fixed-size ML-KEM polynomial kernels.

use core::arch::x86_64::{
  __m128i, __m256i, __m512i, _mm_loadu_si128, _mm256_add_epi16, _mm256_add_epi32, _mm256_and_si256, _mm256_cmpgt_epi16,
  _mm256_cmpgt_epi32, _mm256_cvtepi16_epi32, _mm256_loadu_si256, _mm256_mulhi_epi16, _mm256_mullo_epi16,
  _mm256_mullo_epi32, _mm256_or_si256, _mm256_permutevar8x32_epi32, _mm256_set_epi16, _mm256_set_epi32,
  _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setzero_si256, _mm256_slli_epi32, _mm256_srai_epi32, _mm256_srli_epi32,
  _mm256_storeu_si256, _mm256_sub_epi16, _mm256_sub_epi32, _mm256_unpacklo_epi32, _mm512_add_epi32, _mm512_and_si512,
  _mm512_cmpgt_epi32_mask, _mm512_cvtepi16_epi32, _mm512_loadu_si512, _mm512_mask_add_epi32, _mm512_mask_sub_epi32,
  _mm512_mulhi_epi16, _mm512_mullo_epi16, _mm512_mullo_epi32, _mm512_or_si512, _mm512_set1_epi32, _mm512_setzero_si512,
  _mm512_slli_epi32, _mm512_srai_epi32, _mm512_srli_epi32, _mm512_storeu_si512, _mm512_sub_epi16,
};

use super::{GAMMAS_MONT, N, Poly, Q_I16, Q_I32, Q_MONT_INV_U16, Q_U32, ZETAS_MONT};

#[target_feature(enable = "avx2,sse4.1")]
pub(super) fn ntt_len_ge16_avx2(poly: &mut Poly, zeta_index: &mut usize) {
  let mut len = 128usize;
  while len >= 16 {
    let mut start = 0usize;
    while start < N {
      let zeta = _mm256_set1_epi16(ZETAS_MONT[*zeta_index]);
      *zeta_index = (*zeta_index).strict_add(1);
      let end = start.strict_add(len);
      let mut j = start;
      while j < end {
        // SAFETY: fixed-size AVX2 wide NTT butterfly chunk because:
        // 1. `len >= 16`, `j` advances by 16, and `j < start + len`, so `j..j + 16` is in the lower half.
        // 2. `start + (2 * len) <= N`, so `j + len..j + len + 16` is in the upper half.
        // 3. Each load/store touches exactly 16 u16 coefficients inside the fixed 256-coefficient
        //    polynomial.
        // 4. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`, and the caller proves
        //    AVX2 and SSE4.1 availability.
        unsafe {
          let u = _mm256_loadu_si256(poly.as_ptr().add(j).cast::<__m256i>());
          let t = mul_mont_mod_u16x16_avx2(
            _mm256_loadu_si256(poly.as_ptr().add(j.strict_add(len)).cast::<__m256i>()),
            zeta,
          );
          _mm256_storeu_si256(
            poly.as_mut_ptr().add(j.strict_add(len)).cast::<__m256i>(),
            sub_mod_u16x16_avx2(u, t),
          );
          _mm256_storeu_si256(poly.as_mut_ptr().add(j).cast::<__m256i>(), add_mod_u16x16_avx2(u, t));
        }
        j = j.strict_add(16);
      }
      start = start.strict_add(len.strict_mul(2));
    }
    len >>= 1;
  }
}

#[target_feature(enable = "avx2,sse4.1")]
pub(super) fn inverse_ntt_len_ge16_avx2(poly: &mut Poly, zeta_index: &mut usize) {
  let mut len = 16usize;
  while len <= 128 {
    let mut start = 0usize;
    while start < N {
      let zeta = _mm256_set1_epi16(ZETAS_MONT[*zeta_index]);
      *zeta_index = (*zeta_index).strict_sub(1);
      let end = start.strict_add(len);
      let mut j = start;
      while j < end {
        // SAFETY: fixed-size AVX2 wide inverse-NTT butterfly chunk because:
        // 1. `len >= 16`, `j` advances by 16, and `j < start + len`, so `j..j + 16` is in the lower half.
        // 2. `start + (2 * len) <= N`, so `j + len..j + len + 16` is in the upper half.
        // 3. Each load/store touches exactly 16 u16 coefficients inside the fixed 256-coefficient
        //    polynomial.
        // 4. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`, and the caller proves
        //    AVX2 and SSE4.1 availability.
        unsafe {
          let t = _mm256_loadu_si256(poly.as_ptr().add(j).cast::<__m256i>());
          let u = _mm256_loadu_si256(poly.as_ptr().add(j.strict_add(len)).cast::<__m256i>());
          _mm256_storeu_si256(poly.as_mut_ptr().add(j).cast::<__m256i>(), add_mod_u16x16_avx2(t, u));
          _mm256_storeu_si256(
            poly.as_mut_ptr().add(j.strict_add(len)).cast::<__m256i>(),
            mul_mont_mod_u16x16_avx2(sub_mod_u16x16_avx2(u, t), zeta),
          );
        }
        j = j.strict_add(16);
      }
      start = start.strict_add(len.strict_mul(2));
    }
    len <<= 1;
  }
}

#[target_feature(enable = "avx2,sse4.1")]
pub(super) fn ntt_len2_avx2(poly: &mut Poly, zeta_index: &mut usize) {
  let mut start = 0usize;
  while start < N {
    let zeta0 = ZETAS_MONT[*zeta_index];
    let zeta1 = ZETAS_MONT[(*zeta_index).strict_add(1)];
    let zeta2 = ZETAS_MONT[(*zeta_index).strict_add(2)];
    let zeta3 = ZETAS_MONT[(*zeta_index).strict_add(3)];
    *zeta_index = (*zeta_index).strict_add(4);

    // SAFETY: fixed-size AVX2 len-2 NTT butterfly quartet because:
    // 1. `start` advances by 16 while `start < N == 256`, so `start..start + 16` is in bounds.
    // 2. Each 16-coefficient load contains four public len-2 butterfly groups.
    // 3. The twiddle vector duplicates the four public factors as `[z0, z0, z1, z1, z2, z2, z3, z3]`.
    // 4. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`, and the caller proves
    //    AVX2 and SSE4.1 availability.
    unsafe {
      let values = _mm256_loadu_si256(poly.as_ptr().add(start).cast::<__m256i>());
      let lower = deinterleave_len2_lower_avx2(values);
      let upper = deinterleave_len2_upper_avx2(values);
      let twiddles = duplicate_i16_quartet_lanes_avx2(zeta0, zeta1, zeta2, zeta3);
      let t = mul_mont_mod_u16x16_avx2(upper, twiddles);
      let lower_out = add_mod_u16x16_avx2(lower, t);
      let upper_out = sub_mod_u16x16_avx2(lower, t);
      _mm256_storeu_si256(
        poly.as_mut_ptr().add(start).cast::<__m256i>(),
        _mm256_unpacklo_epi32(lower_out, upper_out),
      );
    }

    start = start.strict_add(16);
  }
}

#[target_feature(enable = "avx2,sse4.1")]
pub(super) fn inverse_ntt_len2_avx2(poly: &mut Poly, zeta_index: &mut usize) {
  let mut start = 0usize;
  while start < N {
    let zeta0 = ZETAS_MONT[*zeta_index];
    *zeta_index = (*zeta_index).strict_sub(1);
    let zeta1 = ZETAS_MONT[*zeta_index];
    *zeta_index = (*zeta_index).strict_sub(1);
    let zeta2 = ZETAS_MONT[*zeta_index];
    *zeta_index = (*zeta_index).strict_sub(1);
    let zeta3 = ZETAS_MONT[*zeta_index];
    *zeta_index = (*zeta_index).strict_sub(1);

    // SAFETY: fixed-size AVX2 len-2 inverse-NTT butterfly quartet because:
    // 1. `start` advances by 16 while `start < N == 256`, so `start..start + 16` is in bounds.
    // 2. Each 16-coefficient load contains four public len-2 butterfly groups.
    // 3. The twiddle vector duplicates the four public factors as `[z0, z0, z1, z1, z2, z2, z3, z3]`.
    // 4. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`, and the caller proves
    //    AVX2 and SSE4.1 availability.
    unsafe {
      let values = _mm256_loadu_si256(poly.as_ptr().add(start).cast::<__m256i>());
      let lower = deinterleave_len2_lower_avx2(values);
      let upper = deinterleave_len2_upper_avx2(values);
      let twiddles = duplicate_i16_quartet_lanes_avx2(zeta0, zeta1, zeta2, zeta3);
      let lower_out = add_mod_u16x16_avx2(lower, upper);
      let upper_out = mul_mont_mod_u16x16_avx2(sub_mod_u16x16_avx2(upper, lower), twiddles);
      _mm256_storeu_si256(
        poly.as_mut_ptr().add(start).cast::<__m256i>(),
        _mm256_unpacklo_epi32(lower_out, upper_out),
      );
    }

    start = start.strict_add(16);
  }
}

#[target_feature(enable = "avx2,sse4.1")]
pub(super) fn multiply_ntts_accumulate_k3_avx2(acc: &mut Poly, a: [&Poly; 3], b: [&Poly; 3]) {
  for i in (0..GAMMAS_MONT.len()).step_by(8) {
    let coeff_offset = i.strict_mul(2);

    // SAFETY: fixed-size AVX2 k=3 dot-product chunk because:
    // 1. `i` advances by 8 while `i < GAMMAS_MONT.len() == 128`; `coeff_offset` is at most 240.
    // 2. Every polynomial load/store touches `coeff_offset..coeff_offset + 16`, within each
    //    256-coefficient polynomial.
    // 3. The gamma load touches `i..i + 8`, within `GAMMAS_MONT`.
    // 4. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`, and the caller proves
    //    AVX2 and SSE4.1 availability.
    unsafe {
      let gamma = load_i16x8_as_i32x8_avx2(GAMMAS_MONT.as_ptr().add(i));
      let acc_pairs = _mm256_loadu_si256(acc.as_ptr().add(coeff_offset).cast::<__m256i>());
      let (p00, p01) = base_multiply_chunk_avx2(a[0], b[0], gamma, coeff_offset);
      let (p10, p11) = base_multiply_chunk_avx2(a[1], b[1], gamma, coeff_offset);
      let (p20, p21) = base_multiply_chunk_avx2(a[2], b[2], gamma, coeff_offset);

      let out0 = add_mod_u32x8_avx2(
        add_mod_u32x8_avx2(add_mod_u32x8_avx2(acc_lanes0_avx2(acc_pairs), p00), p10),
        p20,
      );
      let out1 = add_mod_u32x8_avx2(
        add_mod_u32x8_avx2(add_mod_u32x8_avx2(acc_lanes1_avx2(acc_pairs), p01), p11),
        p21,
      );
      store_u32_pair_lanes_as_u16_avx2(acc.as_mut_ptr().add(coeff_offset), out0, out1);
    }
  }
}

#[target_feature(enable = "avx2,sse4.1")]
pub(super) fn multiply_ntts_accumulate_k4_avx2(acc: &mut Poly, a: [&Poly; 4], b: [&Poly; 4]) {
  for i in (0..GAMMAS_MONT.len()).step_by(8) {
    let coeff_offset = i.strict_mul(2);

    // SAFETY: fixed-size AVX2 k=4 dot-product chunk because:
    // 1. `i` advances by 8 while `i < GAMMAS_MONT.len() == 128`; `coeff_offset` is at most 240.
    // 2. Every polynomial load/store touches `coeff_offset..coeff_offset + 16`, within each
    //    256-coefficient polynomial.
    // 3. The gamma load touches `i..i + 8`, within `GAMMAS_MONT`.
    // 4. The function is gated by `#[target_feature(enable = "avx2,sse4.1")]`, and the caller proves
    //    AVX2 and SSE4.1 availability.
    unsafe {
      let gamma = load_i16x8_as_i32x8_avx2(GAMMAS_MONT.as_ptr().add(i));
      let acc_pairs = _mm256_loadu_si256(acc.as_ptr().add(coeff_offset).cast::<__m256i>());
      let (p00, p01) = base_multiply_chunk_avx2(a[0], b[0], gamma, coeff_offset);
      let (p10, p11) = base_multiply_chunk_avx2(a[1], b[1], gamma, coeff_offset);
      let (p20, p21) = base_multiply_chunk_avx2(a[2], b[2], gamma, coeff_offset);
      let (p30, p31) = base_multiply_chunk_avx2(a[3], b[3], gamma, coeff_offset);

      let out0 = add_mod_u32x8_avx2(
        add_mod_u32x8_avx2(
          add_mod_u32x8_avx2(add_mod_u32x8_avx2(acc_lanes0_avx2(acc_pairs), p00), p10),
          p20,
        ),
        p30,
      );
      let out1 = add_mod_u32x8_avx2(
        add_mod_u32x8_avx2(
          add_mod_u32x8_avx2(add_mod_u32x8_avx2(acc_lanes1_avx2(acc_pairs), p01), p11),
          p21,
        ),
        p31,
      );
      store_u32_pair_lanes_as_u16_avx2(acc.as_mut_ptr().add(coeff_offset), out0, out1);
    }
  }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,sse4.1")]
pub(super) fn multiply_ntts_accumulate_k3_avx512(acc: &mut Poly, a: [&Poly; 3], b: [&Poly; 3]) {
  for i in (0..GAMMAS_MONT.len()).step_by(16) {
    let coeff_offset = i.strict_mul(2);

    // SAFETY: fixed-size AVX-512 k=3 dot-product chunk because:
    // 1. `i` advances by 16 while `i < GAMMAS_MONT.len() == 128`; `coeff_offset` is at most 224.
    // 2. Every polynomial load/store touches `coeff_offset..coeff_offset + 32`, within each
    //    256-coefficient polynomial.
    // 3. The gamma load touches `i..i + 16`, within `GAMMAS_MONT`.
    // 4. The function is gated by `#[target_feature(enable =
    //    "avx2,avx512f,avx512bw,avx512dq,sse4.1")]`, and the caller proves those features are
    //    available.
    unsafe {
      let gamma = load_i16x16_as_i32x16_avx512(GAMMAS_MONT.as_ptr().add(i));
      let acc_pairs = _mm512_loadu_si512(acc.as_ptr().add(coeff_offset).cast::<__m512i>());
      let (p00, p01) = base_multiply_chunk_avx512(a[0], b[0], gamma, coeff_offset);
      let (p10, p11) = base_multiply_chunk_avx512(a[1], b[1], gamma, coeff_offset);
      let (p20, p21) = base_multiply_chunk_avx512(a[2], b[2], gamma, coeff_offset);

      let out0 = add_mod_u32x16_avx512(
        add_mod_u32x16_avx512(add_mod_u32x16_avx512(acc_lanes0_avx512(acc_pairs), p00), p10),
        p20,
      );
      let out1 = add_mod_u32x16_avx512(
        add_mod_u32x16_avx512(add_mod_u32x16_avx512(acc_lanes1_avx512(acc_pairs), p01), p11),
        p21,
      );
      store_u32_pair_lanes_as_u16_avx512(acc.as_mut_ptr().add(coeff_offset), out0, out1);
    }
  }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,sse4.1")]
pub(super) fn multiply_ntts_accumulate_k4_avx512(acc: &mut Poly, a: [&Poly; 4], b: [&Poly; 4]) {
  for i in (0..GAMMAS_MONT.len()).step_by(16) {
    let coeff_offset = i.strict_mul(2);

    // SAFETY: fixed-size AVX-512 k=4 dot-product chunk because:
    // 1. `i` advances by 16 while `i < GAMMAS_MONT.len() == 128`; `coeff_offset` is at most 224.
    // 2. Every polynomial load/store touches `coeff_offset..coeff_offset + 32`, within each
    //    256-coefficient polynomial.
    // 3. The gamma load touches `i..i + 16`, within `GAMMAS_MONT`.
    // 4. The function is gated by `#[target_feature(enable =
    //    "avx2,avx512f,avx512bw,avx512dq,sse4.1")]`, and the caller proves those features are
    //    available.
    unsafe {
      let gamma = load_i16x16_as_i32x16_avx512(GAMMAS_MONT.as_ptr().add(i));
      let acc_pairs = _mm512_loadu_si512(acc.as_ptr().add(coeff_offset).cast::<__m512i>());
      let (p00, p01) = base_multiply_chunk_avx512(a[0], b[0], gamma, coeff_offset);
      let (p10, p11) = base_multiply_chunk_avx512(a[1], b[1], gamma, coeff_offset);
      let (p20, p21) = base_multiply_chunk_avx512(a[2], b[2], gamma, coeff_offset);
      let (p30, p31) = base_multiply_chunk_avx512(a[3], b[3], gamma, coeff_offset);

      let out0 = add_mod_u32x16_avx512(
        add_mod_u32x16_avx512(
          add_mod_u32x16_avx512(add_mod_u32x16_avx512(acc_lanes0_avx512(acc_pairs), p00), p10),
          p20,
        ),
        p30,
      );
      let out1 = add_mod_u32x16_avx512(
        add_mod_u32x16_avx512(
          add_mod_u32x16_avx512(add_mod_u32x16_avx512(acc_lanes1_avx512(acc_pairs), p01), p11),
          p21,
        ),
        p31,
      );
      store_u32_pair_lanes_as_u16_avx512(acc.as_mut_ptr().add(coeff_offset), out0, out1);
    }
  }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,sse4.1")]
pub(super) fn multiply_ntts_add_assign_avx512(acc: &mut Poly, a: &Poly, b: &Poly) {
  for i in (0..GAMMAS_MONT.len()).step_by(16) {
    let coeff_offset = i.strict_mul(2);

    // SAFETY: fixed-size AVX-512 polynomial multiply-add chunk because:
    // 1. `i` advances by 16 while `i < GAMMAS_MONT.len() == 128`; `coeff_offset` is at most 224.
    // 2. Every polynomial load/store touches `coeff_offset..coeff_offset + 32`, within each
    //    256-coefficient polynomial.
    // 3. The gamma load touches `i..i + 16`, within `GAMMAS_MONT`.
    // 4. The function is gated by `#[target_feature(enable =
    //    "avx2,avx512f,avx512bw,avx512dq,sse4.1")]`, and the caller proves those features are
    //    available.
    unsafe {
      let gamma = load_i16x16_as_i32x16_avx512(GAMMAS_MONT.as_ptr().add(i));
      let acc_pairs = _mm512_loadu_si512(acc.as_ptr().add(coeff_offset).cast::<__m512i>());
      let (product0, product1) = base_multiply_chunk_avx512(a, b, gamma, coeff_offset);

      let out0 = add_mod_u32x16_avx512(acc_lanes0_avx512(acc_pairs), product0);
      let out1 = add_mod_u32x16_avx512(acc_lanes1_avx512(acc_pairs), product1);
      store_u32_pair_lanes_as_u16_avx512(acc.as_mut_ptr().add(coeff_offset), out0, out1);
    }
  }
}

#[target_feature(enable = "avx2,sse4.1")]
fn deinterleave_len2_lower_avx2(values: __m256i) -> __m256i {
  _mm256_permutevar8x32_epi32(values, _mm256_set_epi32(0, 0, 6, 4, 0, 0, 2, 0))
}

#[target_feature(enable = "avx2,sse4.1")]
fn deinterleave_len2_upper_avx2(values: __m256i) -> __m256i {
  _mm256_permutevar8x32_epi32(values, _mm256_set_epi32(0, 0, 7, 5, 0, 0, 3, 1))
}

#[target_feature(enable = "avx2,sse4.1")]
fn duplicate_i16_quartet_lanes_avx2(a: i16, b: i16, c: i16, d: i16) -> __m256i {
  _mm256_set_epi16(0, 0, 0, 0, d, d, c, c, 0, 0, 0, 0, b, b, a, a)
}

#[target_feature(enable = "avx2,sse4.1")]
fn mul_mont_mod_u16x16_avx2(a: __m256i, b_mont: __m256i) -> __m256i {
  signed_to_mod_q_s16x16_avx2(montgomery_reduce_s16x16_avx2(
    _mm256_mullo_epi16(a, b_mont),
    _mm256_mulhi_epi16(a, b_mont),
  ))
}

#[target_feature(enable = "avx2,sse4.1")]
fn montgomery_reduce_s16x16_avx2(low: __m256i, high: __m256i) -> __m256i {
  let k = _mm256_mullo_epi16(low, _mm256_set1_epi16(Q_MONT_INV_U16 as i16));
  let c = _mm256_mulhi_epi16(k, _mm256_set1_epi16(Q_I16));
  _mm256_sub_epi16(high, c)
}

#[target_feature(enable = "avx2,sse4.1")]
fn signed_to_mod_q_s16x16_avx2(value: __m256i) -> __m256i {
  let negative = _mm256_cmpgt_epi16(_mm256_setzero_si256(), value);
  _mm256_add_epi16(value, _mm256_and_si256(negative, _mm256_set1_epi16(Q_I16)))
}

#[target_feature(enable = "avx2,sse4.1")]
fn add_mod_u16x16_avx2(a: __m256i, b: __m256i) -> __m256i {
  let sum = _mm256_add_epi16(a, b);
  let ge_q = _mm256_cmpgt_epi16(sum, _mm256_set1_epi16(Q_I16 - 1));
  _mm256_sub_epi16(sum, _mm256_and_si256(ge_q, _mm256_set1_epi16(Q_I16)))
}

#[target_feature(enable = "avx2,sse4.1")]
fn sub_mod_u16x16_avx2(a: __m256i, b: __m256i) -> __m256i {
  let diff = _mm256_sub_epi16(a, b);
  let borrowed = _mm256_cmpgt_epi16(b, a);
  _mm256_add_epi16(diff, _mm256_and_si256(borrowed, _mm256_set1_epi16(Q_I16)))
}

#[target_feature(enable = "avx2,sse4.1")]
fn base_multiply_chunk_avx2(a: &Poly, b: &Poly, gamma: __m256i, coeff_offset: usize) -> (__m256i, __m256i) {
  let mask = _mm256_set1_epi32(0xffff);
  // SAFETY: fixed-size AVX2 polynomial chunk loads because:
  // 1. The caller passes `coeff_offset` from `0..=240` in 16-coefficient steps.
  // 2. Each load touches `coeff_offset..coeff_offset + 16`, within each 256-coefficient polynomial.
  // 3. `_mm256_loadu_si256` accepts arbitrary alignment.
  unsafe {
    let a_pairs = _mm256_loadu_si256(a.as_ptr().add(coeff_offset).cast::<__m256i>());
    let b_pairs = _mm256_loadu_si256(b.as_ptr().add(coeff_offset).cast::<__m256i>());
    let a0 = _mm256_and_si256(a_pairs, mask);
    let a1 = _mm256_srli_epi32::<16>(a_pairs);
    let b0 = _mm256_and_si256(b_pairs, mask);
    let b1 = _mm256_srli_epi32::<16>(b_pairs);

    let a0b0 = _mm256_mullo_epi32(a0, b0);
    let a1b1 = montgomery_reduce_i32x8_avx2(_mm256_mullo_epi32(a1, b1));
    let a1b1_gamma = _mm256_mullo_epi32(a1b1, gamma);
    let c0 = signed_to_mod_q_i32x8_avx2(montgomery_reduce_i32x8_avx2(_mm256_add_epi32(a0b0, a1b1_gamma)));
    let c1 = signed_to_mod_q_i32x8_avx2(montgomery_reduce_i32x8_avx2(_mm256_add_epi32(
      _mm256_mullo_epi32(a0, b1),
      _mm256_mullo_epi32(a1, b0),
    )));
    (c0, c1)
  }
}

#[target_feature(enable = "avx2,sse4.1")]
fn acc_lanes0_avx2(acc_pairs: __m256i) -> __m256i {
  _mm256_and_si256(acc_pairs, _mm256_set1_epi32(0xffff))
}

#[target_feature(enable = "avx2,sse4.1")]
fn acc_lanes1_avx2(acc_pairs: __m256i) -> __m256i {
  _mm256_srli_epi32::<16>(acc_pairs)
}

#[target_feature(enable = "avx2,sse4.1")]
fn store_u32_pair_lanes_as_u16_avx2(ptr: *mut u16, lo: __m256i, hi: __m256i) {
  let packed = _mm256_or_si256(lo, _mm256_slli_epi32::<16>(hi));
  // SAFETY: fixed-size AVX2 polynomial chunk store because:
  // 1. The caller passes a pointer to the start of a 16-coefficient in-bounds chunk.
  // 2. `lo` and `hi` are reduced modulo Q, so each lane fits in u16 before packing.
  // 3. `_mm256_storeu_si256` accepts arbitrary alignment.
  unsafe { _mm256_storeu_si256(ptr.cast::<__m256i>(), packed) };
}

#[target_feature(enable = "avx2,sse4.1")]
fn load_i16x8_as_i32x8_avx2(ptr: *const i16) -> __m256i {
  // SAFETY: unaligned 8-coefficient input load because:
  // 1. The caller proves `ptr..ptr + 8` is readable.
  // 2. `_mm_loadu_si128` accepts arbitrary alignment.
  // 3. `_mm256_cvtepi16_epi32` sign-extends the 8 i16 lanes into 8 i32 lanes.
  let packed = unsafe { _mm_loadu_si128(ptr.cast::<__m128i>()) };
  _mm256_cvtepi16_epi32(packed)
}

#[target_feature(enable = "avx2,sse4.1")]
fn montgomery_reduce_i32x8_avx2(value: __m256i) -> __m256i {
  let k = _mm256_mullo_epi16(value, _mm256_set1_epi32(i32::from(Q_MONT_INV_U16)));
  let c = _mm256_mulhi_epi16(k, _mm256_set1_epi32(Q_I32));
  let value_high = _mm256_srli_epi32::<16>(value);
  let reduced = _mm256_sub_epi16(value_high, c);
  _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(reduced))
}

#[target_feature(enable = "avx2,sse4.1")]
fn signed_to_mod_q_i32x8_avx2(value: __m256i) -> __m256i {
  let negative = _mm256_cmpgt_epi32(_mm256_setzero_si256(), value);
  _mm256_add_epi32(value, _mm256_and_si256(negative, _mm256_set1_epi32(Q_I32)))
}

#[target_feature(enable = "avx2,sse4.1")]
fn add_mod_u32x8_avx2(a: __m256i, b: __m256i) -> __m256i {
  let sum = _mm256_add_epi32(a, b);
  let ge_q = _mm256_cmpgt_epi32(sum, _mm256_set1_epi32((Q_U32 - 1) as i32));
  _mm256_sub_epi32(sum, _mm256_and_si256(ge_q, _mm256_set1_epi32(Q_I32)))
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,sse4.1")]
fn base_multiply_chunk_avx512(a: &Poly, b: &Poly, gamma: __m512i, coeff_offset: usize) -> (__m512i, __m512i) {
  let mask = _mm512_set1_epi32(0xffff);
  // SAFETY: fixed-size AVX-512 polynomial chunk loads because:
  // 1. The caller passes `coeff_offset` from `0..=224` in 32-coefficient steps.
  // 2. Each load touches `coeff_offset..coeff_offset + 32`, within each 256-coefficient polynomial.
  // 3. `_mm512_loadu_si512` accepts arbitrary alignment.
  unsafe {
    let a_pairs = _mm512_loadu_si512(a.as_ptr().add(coeff_offset).cast::<__m512i>());
    let b_pairs = _mm512_loadu_si512(b.as_ptr().add(coeff_offset).cast::<__m512i>());
    let a0 = _mm512_and_si512(a_pairs, mask);
    let a1 = _mm512_srli_epi32::<16>(a_pairs);
    let b0 = _mm512_and_si512(b_pairs, mask);
    let b1 = _mm512_srli_epi32::<16>(b_pairs);

    let a0b0 = _mm512_mullo_epi32(a0, b0);
    let a1b1 = montgomery_reduce_i32x16_avx512(_mm512_mullo_epi32(a1, b1));
    let a1b1_gamma = _mm512_mullo_epi32(a1b1, gamma);
    let c0 = signed_to_mod_q_i32x16_avx512(montgomery_reduce_i32x16_avx512(_mm512_add_epi32(a0b0, a1b1_gamma)));
    let c1 = signed_to_mod_q_i32x16_avx512(montgomery_reduce_i32x16_avx512(_mm512_add_epi32(
      _mm512_mullo_epi32(a0, b1),
      _mm512_mullo_epi32(a1, b0),
    )));
    (c0, c1)
  }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,sse4.1")]
fn acc_lanes0_avx512(acc_pairs: __m512i) -> __m512i {
  _mm512_and_si512(acc_pairs, _mm512_set1_epi32(0xffff))
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,sse4.1")]
fn acc_lanes1_avx512(acc_pairs: __m512i) -> __m512i {
  _mm512_srli_epi32::<16>(acc_pairs)
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,sse4.1")]
fn store_u32_pair_lanes_as_u16_avx512(ptr: *mut u16, lo: __m512i, hi: __m512i) {
  let packed = _mm512_or_si512(lo, _mm512_slli_epi32::<16>(hi));
  // SAFETY: fixed-size AVX-512 polynomial chunk store because:
  // 1. The caller passes a pointer to the start of a 32-coefficient in-bounds chunk.
  // 2. `lo` and `hi` are reduced modulo Q, so each lane fits in u16 before packing.
  // 3. `_mm512_storeu_si512` accepts arbitrary alignment.
  unsafe { _mm512_storeu_si512(ptr.cast::<__m512i>(), packed) };
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,sse4.1")]
fn load_i16x16_as_i32x16_avx512(ptr: *const i16) -> __m512i {
  // SAFETY: unaligned 16-coefficient input load because:
  // 1. The caller proves `ptr..ptr + 16` is readable.
  // 2. `_mm256_loadu_si256` accepts arbitrary alignment.
  // 3. `_mm512_cvtepi16_epi32` sign-extends the 16 i16 lanes into 16 i32 lanes.
  let packed = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
  _mm512_cvtepi16_epi32(packed)
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,sse4.1")]
fn montgomery_reduce_i32x16_avx512(value: __m512i) -> __m512i {
  let k = _mm512_mullo_epi16(value, _mm512_set1_epi32(i32::from(Q_MONT_INV_U16)));
  let c = _mm512_mulhi_epi16(k, _mm512_set1_epi32(Q_I32));
  let value_high = _mm512_srli_epi32::<16>(value);
  let reduced = _mm512_sub_epi16(value_high, c);
  _mm512_srai_epi32::<16>(_mm512_slli_epi32::<16>(reduced))
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,sse4.1")]
fn signed_to_mod_q_i32x16_avx512(value: __m512i) -> __m512i {
  let negative = _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), value);
  _mm512_mask_add_epi32(value, negative, value, _mm512_set1_epi32(Q_I32))
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,sse4.1")]
fn add_mod_u32x16_avx512(a: __m512i, b: __m512i) -> __m512i {
  let sum = _mm512_add_epi32(a, b);
  let ge_q = _mm512_cmpgt_epi32_mask(sum, _mm512_set1_epi32((Q_U32 - 1) as i32));
  _mm512_mask_sub_epi32(sum, ge_q, sum, _mm512_set1_epi32(Q_I32))
}
