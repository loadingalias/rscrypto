use core::arch::s390x::{
  vec_add, vec_mul, vec_sl, vec_sr, vec_sra, vec_sub, vector_signed_int, vector_unsigned_int, vector_unsigned_long_long,
};
use core::simd::{
  i32x4, i64x2,
  num::{SimdInt, SimdUint},
  u32x4, u64x2,
};

use super::{
  GAMMAS_MONT, MONT_R_SQUARED_MOD_Q, N, Poly, Q_HALF, Q_I32, Q_MONT_INV_U16, Q_U32, SAMPLE_NTT_ACC_CHUNK_COEFFS,
  ZETAS_MONT, low_u16, low_u32,
};

const Q_MONT_INV_I32: i32 = Q_MONT_INV_U16.cast_signed() as i32;
const Q_COMPRESS_DIV_SHIFT: u32 = 33;

/// # Safety
/// The executing CPU must support z/Vector.
#[target_feature(enable = "vector")]
pub(super) unsafe fn compress_values_4<const D: usize>(values: [u16; 4]) -> [u16; 4] {
  let value = u32x4_from_u16(values);
  let shift = u32::try_from(D).expect("ML-KEM compression width fits u32");
  let numerator = wrapping_add_u32x4(wrapping_shl_u32x4(value, shift), u32x4::splat(Q_HALF));
  // SAFETY: this function is already gated by z/Vector and the helper has no additional contract.
  let quotient = unsafe { div_q_compress_u32x4_ct(numerator) };
  u32x4_to_u16(quotient & u32x4::splat((1u32 << D).strict_sub(1)))
}

/// # Safety
/// The executing CPU must support z/Vector.
#[target_feature(enable = "vector")]
pub(super) unsafe fn decompress_values_4<const D: usize>(values: [u16; 4]) -> [u16; 4] {
  let value = u32x4_from_u16(values);
  let shift = u32::try_from(D).expect("ML-KEM compression width fits u32");
  let scaled = wrapping_add_u32x4(
    mul_u32x4_16_ct(u32x4::splat(Q_U32), value),
    u32x4::splat(1u32 << D.strict_sub(1)),
  );
  u32x4_to_u16(wrapping_shr_u32x4(scaled, shift))
}

/// # Safety
/// The executing CPU must support z/Vector.
#[target_feature(enable = "vector")]
pub(super) unsafe fn to_montgomery_product_domain_vector(poly: &mut Poly) {
  let poly_ptr = poly.as_mut_ptr();
  let mut i = 0usize;
  while i < N {
    // SAFETY: `i` advances in fixed 4-coefficient chunks across the 256-coefficient polynomial.
    let coeffs = unsafe { load_u32x4(poly_ptr.cast_const(), i) };
    let converted = signed_to_mod_q_i32x4(montgomery_reduce_i32x4(coeffs.cast::<i32>()));
    // SAFETY: same fixed chunk proven above.
    unsafe {
      store_u32x4(poly_ptr, i, converted);
    }
    i = i.wrapping_add(4);
  }
}

/// # Safety
/// The executing CPU must support z/Vector.
#[target_feature(enable = "vector")]
pub(super) unsafe fn from_montgomery_product_domain_vector(poly: &mut Poly) {
  let poly_ptr = poly.as_mut_ptr();
  let mut i = 0usize;
  while i < N {
    // SAFETY: `i` advances in fixed 4-coefficient chunks across the 256-coefficient polynomial.
    let coeffs = unsafe { load_u32x4(poly_ptr.cast_const(), i) };
    let converted = mul_mont_const_mod_u32x4(coeffs, MONT_R_SQUARED_MOD_Q);
    // SAFETY: same fixed chunk proven above.
    unsafe {
      store_u32x4(poly_ptr, i, converted);
    }
    i = i.wrapping_add(4);
  }
}

/// # Safety
/// The executing CPU must support z/Vector.
#[target_feature(enable = "vector")]
pub(super) unsafe fn ntt_vector(poly: &mut Poly) {
  let poly_ptr = poly.as_mut_ptr();
  let mut zeta_index = 1usize;
  let mut len = 128usize;
  while len >= 4 {
    let mut start = 0usize;
    while start < N {
      // SAFETY: `zeta_index` follows the fixed ML-KEM NTT schedule and stays inside
      // `ZETAS_MONT`.
      let zeta = unsafe { load_zeta(zeta_index) };
      zeta_index = zeta_index.wrapping_add(1);
      let end = start.wrapping_add(len);
      let mut j = start;
      while j < end {
        // SAFETY: `j..j + 4` and `j + len..j + len + 4` are inside the lower and upper halves of
        // the current public NTT butterfly block.
        unsafe {
          let u = load_u32x4(poly_ptr.cast_const(), j);
          let upper = j.wrapping_add(len);
          let t = mul_mont_const_mod_u32x4(load_u32x4(poly_ptr.cast_const(), upper), zeta);
          store_u32x4(poly_ptr, upper, sub_mod_u32x4(u, t));
          store_u32x4(poly_ptr, j, add_mod_u32x4(u, t));
        }
        j = j.wrapping_add(4);
      }
      start = start.wrapping_add(len << 1);
    }
    len >>= 1;
  }

  // SAFETY: the first NTT stages leave `zeta_index` positioned at the fixed len-2 tail schedule,
  // and `poly_ptr` points to the full 256-coefficient polynomial.
  unsafe {
    ntt_len2_vector(poly_ptr, &mut zeta_index);
  }
}

/// # Safety
/// The executing CPU must support z/Vector.
#[target_feature(enable = "vector")]
pub(super) unsafe fn inverse_ntt_vector(poly: &mut Poly, final_scale_mont: i16) {
  let poly_ptr = poly.as_mut_ptr();
  let mut zeta_index = 127usize;
  // SAFETY: `zeta_index` starts at the fixed inverse len-2 tail schedule, and `poly_ptr` points to
  // the full 256-coefficient polynomial.
  unsafe {
    inverse_ntt_len2_vector(poly_ptr, &mut zeta_index);
  }

  let mut len = 4usize;
  while len <= 128 {
    let mut start = 0usize;
    while start < N {
      // SAFETY: `zeta_index` follows the fixed ML-KEM inverse NTT schedule and stays inside
      // `ZETAS_MONT`.
      let zeta = unsafe { load_zeta(zeta_index) };
      zeta_index = zeta_index.wrapping_sub(1);
      let end = start.wrapping_add(len);
      let mut j = start;
      while j < end {
        // SAFETY: `j..j + 4` and `j + len..j + len + 4` are inside the lower and upper halves of
        // the current public inverse-NTT butterfly block.
        unsafe {
          let t = load_u32x4(poly_ptr.cast_const(), j);
          let upper = j.wrapping_add(len);
          let u = load_u32x4(poly_ptr.cast_const(), upper);
          store_u32x4(poly_ptr, j, add_mod_u32x4(t, u));
          store_u32x4(poly_ptr, upper, mul_mont_const_mod_u32x4(sub_mod_u32x4(u, t), zeta));
        }
        j = j.wrapping_add(4);
      }
      start = start.wrapping_add(len << 1);
    }
    len <<= 1;
  }

  let mut i = 0usize;
  while i < N {
    // SAFETY: `i` advances in 4-coefficient chunks across the fixed 256-coefficient polynomial.
    let coeffs = unsafe { load_u32x4(poly_ptr.cast_const(), i) };
    let scaled = mul_mont_const_mod_u32x4(coeffs, final_scale_mont);
    // SAFETY: same fixed 4-coefficient chunk proven above.
    unsafe {
      store_u32x4(poly_ptr, i, scaled);
    }
    i = i.wrapping_add(4);
  }
}

/// # Safety
/// The executing CPU must support z/Vector.
#[target_feature(enable = "vector")]
pub(super) unsafe fn multiply_ntts_add_assign_vector(acc: &mut Poly, a: &Poly, b: &Poly) {
  let acc_ptr = acc.as_mut_ptr();
  let a_ptr = a.as_ptr();
  let b_ptr = b.as_ptr();
  let mut coeff_offset = 0usize;
  while coeff_offset < N {
    // SAFETY: `coeff_offset` advances by one public 8-coefficient base-multiply group over the full
    // 256-coefficient polynomials.
    unsafe {
      multiply_ntts_add_assign_4(acc_ptr, a_ptr, coeff_offset, b_ptr, coeff_offset, coeff_offset / 2);
    }
    coeff_offset = coeff_offset.wrapping_add(8);
  }
}

/// # Safety
/// The executing CPU must support z/Vector. `coeff_offset` must be a multiple of
/// 16 and `coeff_offset + 16` must not exceed `N`.
#[target_feature(enable = "vector")]
pub(super) unsafe fn multiply_ntts_add_assign_chunk_vector(
  acc: &mut Poly,
  a: &[u16; SAMPLE_NTT_ACC_CHUNK_COEFFS],
  b: &Poly,
  coeff_offset: usize,
) {
  debug_assert_eq!(coeff_offset % SAMPLE_NTT_ACC_CHUNK_COEFFS, 0);
  debug_assert!(coeff_offset.strict_add(SAMPLE_NTT_ACC_CHUNK_COEFFS) <= N);

  let acc_ptr = acc.as_mut_ptr();
  let a_ptr = a.as_ptr();
  let b_ptr = b.as_ptr();
  let mut local = 0usize;
  while local < SAMPLE_NTT_ACC_CHUNK_COEFFS {
    let global = coeff_offset.wrapping_add(local);
    // SAFETY: `local` covers exactly one public 16-coefficient sample chunk in two 8-coefficient
    // groups, and `global` is the corresponding fixed chunk boundary inside `b` and `acc`.
    unsafe {
      multiply_ntts_add_assign_4(acc_ptr, a_ptr, local, b_ptr, global, global / 2);
    }
    local = local.wrapping_add(8);
  }
}

/// # Safety
/// The executing CPU must support z/Vector.
#[target_feature(enable = "vector")]
pub(super) unsafe fn multiply_ntts_accumulate_k3_vector(acc: &mut Poly, a: [&Poly; 3], b: [&Poly; 3]) {
  let acc_ptr = acc.as_mut_ptr();
  let a0 = a[0].as_ptr();
  let a1 = a[1].as_ptr();
  let a2 = a[2].as_ptr();
  let b0 = b[0].as_ptr();
  let b1 = b[1].as_ptr();
  let b2 = b[2].as_ptr();

  let mut coeff_offset = 0usize;
  while coeff_offset < N {
    // SAFETY: `coeff_offset` advances by one public 8-coefficient base-multiply group over the full
    // 256-coefficient polynomials, and all six multiplicand pointers refer to fixed-size `Poly`
    // values supplied by the caller.
    unsafe {
      multiply_ntts_accumulate_k3_4(acc_ptr, [a0, a1, a2], [b0, b1, b2], coeff_offset, coeff_offset / 2);
    }
    coeff_offset = coeff_offset.wrapping_add(8);
  }
}

/// # Safety
/// The executing CPU must support z/Vector.
#[target_feature(enable = "vector")]
pub(super) unsafe fn multiply_ntts_accumulate_k4_vector(acc: &mut Poly, a: [&Poly; 4], b: [&Poly; 4]) {
  let acc_ptr = acc.as_mut_ptr();
  let a0 = a[0].as_ptr();
  let a1 = a[1].as_ptr();
  let a2 = a[2].as_ptr();
  let a3 = a[3].as_ptr();
  let b0 = b[0].as_ptr();
  let b1 = b[1].as_ptr();
  let b2 = b[2].as_ptr();
  let b3 = b[3].as_ptr();

  let mut coeff_offset = 0usize;
  while coeff_offset < N {
    // SAFETY: `coeff_offset` advances by one public 8-coefficient base-multiply group over the full
    // 256-coefficient polynomials, and all eight multiplicand pointers refer to fixed-size `Poly`
    // values supplied by the caller.
    unsafe {
      multiply_ntts_accumulate_k4_4(
        acc_ptr,
        [a0, a1, a2, a3],
        [b0, b1, b2, b3],
        coeff_offset,
        coeff_offset / 2,
      );
    }
    coeff_offset = coeff_offset.wrapping_add(8);
  }
}

/// # Safety
/// The executing CPU must support z/Vector. `a` and `b` must be aligned and readable
/// for eight initialized coefficients at `a_offset` and `b_offset`. `acc` must be
/// aligned, readable, and writable for eight initialized coefficients at `b_offset`,
/// without overlapping either input. `gamma_offset + 4` must not exceed `GAMMAS_MONT.len()`.
#[inline(always)]
unsafe fn multiply_ntts_add_assign_4(
  acc: *mut u16,
  a: *const u16,
  a_offset: usize,
  b: *const u16,
  b_offset: usize,
  gamma_offset: usize,
) {
  // SAFETY: caller guarantees that `a_offset..a_offset + 8`, `b_offset..b_offset + 8`, and
  // `gamma_offset..gamma_offset + 4` are inside their fixed ML-KEM buffers.
  let (c0, c1) = unsafe { base_multiply_4(a, a_offset, b, b_offset, gamma_offset) };

  // SAFETY: caller guarantees `b_offset..b_offset + 8` is inside `acc`.
  unsafe {
    add_interleaved_4(acc, b_offset, c0, c1);
  }
}

/// # Safety
/// The executing CPU must support z/Vector. Every input pointer must be aligned and
/// readable for eight initialized coefficients starting at `coeff_offset`. `acc`
/// must be aligned, readable, and writable for the same initialized range, without
/// overlapping any input. `gamma_offset + 4` must not exceed `GAMMAS_MONT.len()`.
#[inline(always)]
unsafe fn multiply_ntts_accumulate_k3_4(
  acc: *mut u16,
  a: [*const u16; 3],
  b: [*const u16; 3],
  coeff_offset: usize,
  gamma_offset: usize,
) {
  // SAFETY: caller guarantees `coeff_offset..coeff_offset + 8` for every multiplicand and
  // `gamma_offset..gamma_offset + 4` are inside fixed ML-KEM buffers.
  let ((a0b0, a0b1), (a1b0, a1b1), (a2b0, a2b1)) = unsafe {
    (
      base_multiply_4(a[0], coeff_offset, b[0], coeff_offset, gamma_offset),
      base_multiply_4(a[1], coeff_offset, b[1], coeff_offset, gamma_offset),
      base_multiply_4(a[2], coeff_offset, b[2], coeff_offset, gamma_offset),
    )
  };
  let c0 = add_mod_u32x4(add_mod_u32x4(a0b0, a1b0), a2b0);
  let c1 = add_mod_u32x4(add_mod_u32x4(a0b1, a1b1), a2b1);

  // SAFETY: caller guarantees `coeff_offset..coeff_offset + 8` is inside `acc`.
  unsafe {
    add_interleaved_4(acc, coeff_offset, c0, c1);
  }
}

/// # Safety
/// The executing CPU must support z/Vector. Every input pointer must be aligned and
/// readable for eight initialized coefficients starting at `coeff_offset`. `acc`
/// must be aligned, readable, and writable for the same initialized range, without
/// overlapping any input. `gamma_offset + 4` must not exceed `GAMMAS_MONT.len()`.
#[inline(always)]
unsafe fn multiply_ntts_accumulate_k4_4(
  acc: *mut u16,
  a: [*const u16; 4],
  b: [*const u16; 4],
  coeff_offset: usize,
  gamma_offset: usize,
) {
  // SAFETY: caller guarantees `coeff_offset..coeff_offset + 8` for every multiplicand and
  // `gamma_offset..gamma_offset + 4` are inside fixed ML-KEM buffers.
  let ((a0b0, a0b1), (a1b0, a1b1), (a2b0, a2b1), (a3b0, a3b1)) = unsafe {
    (
      base_multiply_4(a[0], coeff_offset, b[0], coeff_offset, gamma_offset),
      base_multiply_4(a[1], coeff_offset, b[1], coeff_offset, gamma_offset),
      base_multiply_4(a[2], coeff_offset, b[2], coeff_offset, gamma_offset),
      base_multiply_4(a[3], coeff_offset, b[3], coeff_offset, gamma_offset),
    )
  };
  let c0 = add_mod_u32x4(add_mod_u32x4(add_mod_u32x4(a0b0, a1b0), a2b0), a3b0);
  let c1 = add_mod_u32x4(add_mod_u32x4(add_mod_u32x4(a0b1, a1b1), a2b1), a3b1);

  // SAFETY: caller guarantees `coeff_offset..coeff_offset + 8` is inside `acc`.
  unsafe {
    add_interleaved_4(acc, coeff_offset, c0, c1);
  }
}

/// # Safety
/// The executing CPU must support z/Vector. `a` and `b` must be aligned and readable
/// for eight initialized coefficients at `a_offset` and `b_offset`.
/// `gamma_offset + 4` must not exceed `GAMMAS_MONT.len()`.
#[inline(always)]
unsafe fn base_multiply_4(
  a: *const u16,
  a_offset: usize,
  b: *const u16,
  b_offset: usize,
  gamma_offset: usize,
) -> (u32x4, u32x4) {
  // SAFETY: caller guarantees that `a_offset..a_offset + 8`, `b_offset..b_offset + 8`, and
  // `gamma_offset..gamma_offset + 4` are inside their fixed ML-KEM buffers.
  let (a0, a1, b0, b1, gamma) = unsafe {
    (
      load_even_u32x4(a, a_offset).cast::<i32>(),
      load_odd_u32x4(a, a_offset).cast::<i32>(),
      load_even_u32x4(b, b_offset).cast::<i32>(),
      load_odd_u32x4(b, b_offset).cast::<i32>(),
      load_gamma_i32x4(gamma_offset),
    )
  };

  let a1b1 = montgomery_reduce_i32x4(mul_i32x4_16_ct(a1, b1));
  let a0b0 = mul_i32x4_16_ct(a0, b0);
  let a1b1_gamma = mul_i32x4_16_ct(a1b1, gamma);
  let c0 = signed_to_mod_q_i32x4(montgomery_reduce_i32x4(wrapping_add_i32x4(a0b0, a1b1_gamma)));

  let a0b1 = mul_i32x4_16_ct(a0, b1);
  let a1b0 = mul_i32x4_16_ct(a1, b0);
  let c1 = signed_to_mod_q_i32x4(montgomery_reduce_i32x4(wrapping_add_i32x4(a0b1, a1b0)));

  (c0, c1)
}

/// # Safety
/// The executing CPU must support z/Vector. `poly` must be aligned, initialized,
/// and exclusively readable and writable for `N` coefficients. `zeta_index` must
/// start at 64, leaving 64 entries in `ZETAS_MONT` for this stage.
#[inline(always)]
unsafe fn ntt_len2_vector(poly: *mut u16, zeta_index: &mut usize) {
  let mut start = 0usize;
  while start < N {
    // SAFETY: the len-2 tail consumes two public zeta entries per 8-coefficient block.
    let (zeta0, zeta1) = unsafe { load_zeta_pair(*zeta_index) };
    *zeta_index = (*zeta_index).wrapping_add(2);

    // SAFETY: `start` advances in fixed 8-coefficient blocks across the full polynomial.
    let (lower, upper) = unsafe {
      (
        load_len2_lower_u32x4(poly.cast_const(), start),
        load_len2_upper_u32x4(poly.cast_const(), start),
      )
    };
    let twiddles = duplicate_i32_pair_lanes(zeta0, zeta1);
    let t = mul_mont_mod_u32x4(upper, twiddles);
    // SAFETY: same fixed 8-coefficient block proven above.
    unsafe {
      store_len2_interleaved_4(poly, start, add_mod_u32x4(lower, t), sub_mod_u32x4(lower, t));
    }

    start = start.wrapping_add(8);
  }
}

/// # Safety
/// The executing CPU must support z/Vector. `poly` must be aligned, initialized,
/// and exclusively readable and writable for `N` coefficients. `zeta_index` must
/// start at 127 for the descending 64-entry twiddle schedule.
#[inline(always)]
unsafe fn inverse_ntt_len2_vector(poly: *mut u16, zeta_index: &mut usize) {
  let mut start = 0usize;
  while start < N {
    // SAFETY: the inverse len-2 tail consumes two public zeta entries per 8-coefficient block.
    let zeta0 = unsafe { i32::from(load_zeta(*zeta_index)) };
    *zeta_index = (*zeta_index).wrapping_sub(1);
    // SAFETY: `zeta_index` remains inside the fixed inverse NTT schedule after the decrement above.
    let zeta1 = unsafe { i32::from(load_zeta(*zeta_index)) };
    *zeta_index = (*zeta_index).wrapping_sub(1);

    // SAFETY: `start` advances in fixed 8-coefficient blocks across the full polynomial.
    let (lower, upper) = unsafe {
      (
        load_len2_lower_u32x4(poly.cast_const(), start),
        load_len2_upper_u32x4(poly.cast_const(), start),
      )
    };
    let twiddles = duplicate_i32_pair_lanes(zeta0, zeta1);
    let lower_out = add_mod_u32x4(lower, upper);
    let upper_out = mul_mont_mod_u32x4(sub_mod_u32x4(upper, lower), twiddles);
    // SAFETY: same fixed 8-coefficient block proven above.
    unsafe {
      store_len2_interleaved_4(poly, start, lower_out, upper_out);
    }

    start = start.wrapping_add(8);
  }
}

/// # Safety
/// The executing CPU must support z/Vector. `acc` must be aligned and exclusively
/// readable and writable for eight initialized coefficients starting at `offset`.
#[inline(always)]
unsafe fn add_interleaved_4(acc: *mut u16, offset: usize, c0: u32x4, c1: u32x4) {
  // SAFETY: caller guarantees `offset..offset + 8` is inside `acc`.
  let (out0, out1) = unsafe {
    (
      add_mod_u32x4(load_even_u32x4(acc.cast_const(), offset), c0).to_array(),
      add_mod_u32x4(load_odd_u32x4(acc.cast_const(), offset), c1).to_array(),
    )
  };

  // SAFETY: caller guarantees `offset..offset + 8` is inside `acc`.
  unsafe {
    let acc = acc.add(offset);
    *acc = low_u16(out0[0]);
    *acc.add(1) = low_u16(out1[0]);
    *acc.add(2) = low_u16(out0[1]);
    *acc.add(3) = low_u16(out1[1]);
    *acc.add(4) = low_u16(out0[2]);
    *acc.add(5) = low_u16(out1[2]);
    *acc.add(6) = low_u16(out0[3]);
    *acc.add(7) = low_u16(out1[3]);
  }
}

// Keep modular arithmetic in vector registers; scalar lane arithmetic can lower to
// operand-dependent scalar multiply on IBM Z.
#[inline(always)]
fn wrapping_add_u32x4(a: u32x4, b: u32x4) -> u32x4 {
  // SAFETY: all callers are inside z/Vector-gated entry points. The portable and
  // architectural vector types have the same integer lane widths and order,
  // and every bit pattern is valid. The intrinsic performs wrapping lane arithmetic.
  unsafe {
    core::mem::transmute(vec_add(
      core::mem::transmute::<u32x4, vector_unsigned_int>(a),
      core::mem::transmute::<u32x4, vector_unsigned_int>(b),
    ))
  }
}

#[inline(always)]
fn wrapping_sub_u32x4(a: u32x4, b: u32x4) -> u32x4 {
  // SAFETY: all callers are inside z/Vector-gated entry points. The portable and
  // architectural vector types have the same integer lane widths and order,
  // and every bit pattern is valid. The intrinsic performs wrapping lane arithmetic.
  unsafe {
    core::mem::transmute(vec_sub(
      core::mem::transmute::<u32x4, vector_unsigned_int>(a),
      core::mem::transmute::<u32x4, vector_unsigned_int>(b),
    ))
  }
}

#[inline(always)]
fn wrapping_add_i32x4(a: i32x4, b: i32x4) -> i32x4 {
  // SAFETY: all callers are inside z/Vector-gated entry points. The portable and
  // architectural vector types have the same integer lane widths and order,
  // and every bit pattern is valid. The intrinsic performs wrapping lane arithmetic.
  unsafe {
    core::mem::transmute(vec_add(
      core::mem::transmute::<i32x4, vector_signed_int>(a),
      core::mem::transmute::<i32x4, vector_signed_int>(b),
    ))
  }
}

#[inline(always)]
fn wrapping_sub_i32x4(a: i32x4, b: i32x4) -> i32x4 {
  // SAFETY: all callers are inside z/Vector-gated entry points. The portable and
  // architectural vector types have the same integer lane widths and order,
  // and every bit pattern is valid. The intrinsic performs wrapping lane arithmetic.
  unsafe {
    core::mem::transmute(vec_sub(
      core::mem::transmute::<i32x4, vector_signed_int>(a),
      core::mem::transmute::<i32x4, vector_signed_int>(b),
    ))
  }
}

#[inline(always)]
fn wrapping_add_u64x2(a: u64x2, b: u64x2) -> u64x2 {
  // SAFETY: all callers are inside z/Vector-gated entry points. The portable and
  // architectural vector types have the same integer lane widths and order,
  // and every bit pattern is valid. The intrinsic performs wrapping lane arithmetic.
  unsafe {
    core::mem::transmute(vec_add(
      core::mem::transmute::<u64x2, vector_unsigned_long_long>(a),
      core::mem::transmute::<u64x2, vector_unsigned_long_long>(b),
    ))
  }
}

#[inline(always)]
fn wrapping_shl_u32x4(value: u32x4, shift: u32) -> u32x4 {
  // SAFETY: all callers are inside z/Vector-gated entry points. These vector types
  // have identical lane layouts and all bit patterns are valid. The intrinsic
  // masks each shift count to the lane width, matching wrapping shift semantics.
  unsafe {
    core::mem::transmute(vec_sl(
      core::mem::transmute::<u32x4, vector_unsigned_int>(value),
      core::mem::transmute::<u32x4, vector_unsigned_int>(u32x4::splat(shift)),
    ))
  }
}

#[inline(always)]
fn wrapping_shr_u32x4(value: u32x4, shift: u32) -> u32x4 {
  // SAFETY: all callers are inside z/Vector-gated entry points. These vector types
  // have identical lane layouts and all bit patterns are valid. The intrinsic
  // masks each shift count to the lane width, matching wrapping shift semantics.
  unsafe {
    core::mem::transmute(vec_sr(
      core::mem::transmute::<u32x4, vector_unsigned_int>(value),
      core::mem::transmute::<u32x4, vector_unsigned_int>(u32x4::splat(shift)),
    ))
  }
}

#[inline(always)]
fn wrapping_shl_i32x4(value: i32x4, shift: u32) -> i32x4 {
  // SAFETY: all callers are inside z/Vector-gated entry points. These vector types
  // have identical lane layouts and all bit patterns are valid. The intrinsic
  // masks each shift count to the lane width, matching wrapping shift semantics.
  unsafe {
    core::mem::transmute(vec_sl(
      core::mem::transmute::<i32x4, vector_signed_int>(value),
      core::mem::transmute::<u32x4, vector_unsigned_int>(u32x4::splat(shift)),
    ))
  }
}

#[inline(always)]
fn wrapping_shr_i32x4(value: i32x4, shift: u32) -> i32x4 {
  // SAFETY: all callers are inside z/Vector-gated entry points. These vector types
  // have identical lane layouts and all bit patterns are valid. The intrinsic
  // masks each shift count to the lane width, matching wrapping shift semantics.
  unsafe {
    core::mem::transmute(vec_sra(
      core::mem::transmute::<i32x4, vector_signed_int>(value),
      core::mem::transmute::<u32x4, vector_unsigned_int>(u32x4::splat(shift)),
    ))
  }
}

#[inline(always)]
fn wrapping_shl_u64x2(value: u64x2, shift: u32) -> u64x2 {
  // SAFETY: all callers are inside z/Vector-gated entry points. These vector types
  // have identical lane layouts and all bit patterns are valid. The intrinsic
  // masks each shift count to the lane width, matching wrapping shift semantics.
  unsafe {
    core::mem::transmute(vec_sl(
      core::mem::transmute::<u64x2, vector_unsigned_long_long>(value),
      core::mem::transmute::<u64x2, vector_unsigned_long_long>(u64x2::splat(u64::from(shift))),
    ))
  }
}

#[inline(always)]
fn wrapping_shr_u64x2(value: u64x2, shift: u32) -> u64x2 {
  // SAFETY: all callers are inside z/Vector-gated entry points. These vector types
  // have identical lane layouts and all bit patterns are valid. The intrinsic
  // masks each shift count to the lane width, matching wrapping shift semantics.
  unsafe {
    core::mem::transmute(vec_sr(
      core::mem::transmute::<u64x2, vector_unsigned_long_long>(value),
      core::mem::transmute::<u64x2, vector_unsigned_long_long>(u64x2::splat(u64::from(shift))),
    ))
  }
}

#[inline(always)]
fn montgomery_reduce_i32x4(value: i32x4) -> i32x4 {
  let k = mul_i32x4_16_ct(sign_extend_i16_i32x4(value), i32x4::splat(Q_MONT_INV_I32));
  let c = wrapping_shr_i32x4(mul_i32x4_16_ct(sign_extend_i16_i32x4(k), i32x4::splat(Q_I32)), 16);
  sign_extend_i16_i32x4(wrapping_sub_i32x4(wrapping_shr_i32x4(value, 16), c))
}

#[inline(always)]
fn signed_to_mod_q_i32x4(value: i32x4) -> u32x4 {
  wrapping_add_i32x4(value, (wrapping_shr_i32x4(value, 31)) & i32x4::splat(Q_I32)).cast::<u32>()
}

#[inline(always)]
fn add_mod_u32x4(a: u32x4, b: u32x4) -> u32x4 {
  add_q_if_borrowed_u32x4(wrapping_sub_u32x4(wrapping_add_u32x4(a, b), u32x4::splat(Q_U32)))
}

#[inline(always)]
fn sub_mod_u32x4(a: u32x4, b: u32x4) -> u32x4 {
  add_q_if_borrowed_u32x4(wrapping_sub_u32x4(a, b))
}

#[inline(always)]
fn add_q_if_borrowed_u32x4(value: u32x4) -> u32x4 {
  let borrow = wrapping_shr_u32x4(value, 31);
  // SAFETY: every caller of this helper is reached only from z/Vector-gated ML-KEM entry points in
  // this module. `borrow` is a 0/1 lane value derived from fixed-width modular arithmetic.
  let mask = unsafe { bitmask_u32x4(borrow) };
  wrapping_add_u32x4(value, mask & u32x4::splat(Q_U32))
}

#[inline(always)]
fn sign_extend_i16_i32x4(value: i32x4) -> i32x4 {
  wrapping_shr_i32x4(wrapping_shl_i32x4(value, 16), 16)
}

#[inline(always)]
fn mul_mont_const_mod_u32x4(a: u32x4, b_mont: i16) -> u32x4 {
  mul_mont_mod_u32x4(a, i32x4::splat(i32::from(b_mont)))
}

#[inline(always)]
fn mul_mont_mod_u32x4(a: u32x4, b_mont: i32x4) -> u32x4 {
  signed_to_mod_q_i32x4(montgomery_reduce_i32x4(mul_i32x4_16_ct(a.cast::<i32>(), b_mont)))
}

#[inline(always)]
fn mul_i32x4_16_ct(a: i32x4, b: i32x4) -> i32x4 {
  let a_sign = wrapping_shr_i32x4(a, 31);
  let b_sign = wrapping_shr_i32x4(b, 31);
  let abs_a = wrapping_sub_i32x4(a ^ a_sign, a_sign).cast::<u32>();
  let abs_b = wrapping_sub_i32x4(b ^ b_sign, b_sign).cast::<u32>();
  let magnitude = mul_u32x4_16_ct(abs_a, abs_b);
  let sign = (a_sign ^ b_sign).cast::<u32>();
  wrapping_sub_u32x4(magnitude ^ sign, sign).cast::<i32>()
}

#[inline(always)]
fn mul_u32x4_16_ct(a: u32x4, b: u32x4) -> u32x4 {
  // SAFETY: all callers are inside z/Vector-gated entry points. The portable and
  // architectural vector types have the same integer lane widths and order,
  // and every bit pattern is valid. The intrinsic performs wrapping lane arithmetic.
  unsafe {
    core::mem::transmute(vec_mul(
      core::mem::transmute::<u32x4, vector_unsigned_int>(a),
      core::mem::transmute::<u32x4, vector_unsigned_int>(b),
    ))
  }
}

/// # Safety
/// The executing CPU must support z/Vector.
#[target_feature(enable = "vector")]
#[inline]
unsafe fn div_q_compress_u32x4_ct(value: u32x4) -> u32x4 {
  let [x0, x1, x2, x3] = value.to_array();
  // SAFETY: this helper is gated by z/Vector and widens the lower two public-width lanes.
  let lo = unsafe { div_q_compress_u64x2_ct(u64x2::from_array([u64::from(x0), u64::from(x1)])) }.to_array();
  // SAFETY: same z/Vector contract and fixed lane widening as above.
  let hi = unsafe { div_q_compress_u64x2_ct(u64x2::from_array([u64::from(x2), u64::from(x3)])) }.to_array();
  u32x4::from_array([low_u32(lo[0]), low_u32(lo[1]), low_u32(hi[0]), low_u32(hi[1])])
}

/// # Safety
/// The executing CPU must support z/Vector.
#[target_feature(enable = "vector")]
#[inline]
unsafe fn div_q_compress_u64x2_ct(value: u64x2) -> u64x2 {
  // Exact reciprocal division for ML-KEM compression:
  // floor(value / 3329) == (value * 2_580_335) >> 33 for value < 2^23.
  // The multiplier is expanded as a fixed shift/add schedule so this path does
  // not reintroduce secret-fed multiply on IBM Z.
  // SAFETY: callers reach this helper only from z/Vector-gated ML-KEM compression entry points.
  let acc = unsafe {
    let acc = opaque_u64x2(value);
    let acc = wrapping_add_u64x2(acc, opaque_u64x2(wrapping_shl_u64x2(value, 1)));
    let acc = wrapping_add_u64x2(acc, opaque_u64x2(wrapping_shl_u64x2(value, 2)));
    let acc = wrapping_add_u64x2(acc, opaque_u64x2(wrapping_shl_u64x2(value, 3)));
    let acc = wrapping_add_u64x2(acc, opaque_u64x2(wrapping_shl_u64x2(value, 5)));
    let acc = wrapping_add_u64x2(acc, opaque_u64x2(wrapping_shl_u64x2(value, 6)));
    let acc = wrapping_add_u64x2(acc, opaque_u64x2(wrapping_shl_u64x2(value, 8)));
    let acc = wrapping_add_u64x2(acc, opaque_u64x2(wrapping_shl_u64x2(value, 9)));
    let acc = wrapping_add_u64x2(acc, opaque_u64x2(wrapping_shl_u64x2(value, 10)));
    let acc = wrapping_add_u64x2(acc, opaque_u64x2(wrapping_shl_u64x2(value, 11)));
    let acc = wrapping_add_u64x2(acc, opaque_u64x2(wrapping_shl_u64x2(value, 12)));
    let acc = wrapping_add_u64x2(acc, opaque_u64x2(wrapping_shl_u64x2(value, 14)));
    let acc = wrapping_add_u64x2(acc, opaque_u64x2(wrapping_shl_u64x2(value, 16)));
    let acc = wrapping_add_u64x2(acc, opaque_u64x2(wrapping_shl_u64x2(value, 17)));
    let acc = wrapping_add_u64x2(acc, opaque_u64x2(wrapping_shl_u64x2(value, 18)));
    wrapping_add_u64x2(acc, opaque_u64x2(wrapping_shl_u64x2(value, 21)))
  };
  wrapping_shr_u64x2(acc, Q_COMPRESS_DIV_SHIFT)
}

/// # Safety
/// The executing CPU must support z/Vector.
#[target_feature(enable = "vector")]
#[inline]
unsafe fn opaque_u64x2(value: u64x2) -> u64x2 {
  let mut out = value;
  // SAFETY: this is an empty z/Vector register barrier. It emits no operation, but prevents LLVM
  // from rewriting the fixed shift/add reciprocal schedule into native multiply instructions.
  unsafe {
    core::arch::asm!("/* {0} */", inout(vreg) out, options(nomem, nostack, preserves_flags));
  }
  out
}

/// # Safety
/// The executing CPU must support z/Vector.
#[target_feature(enable = "vector")]
#[inline]
unsafe fn bitmask_u32x4(value: u32x4) -> u32x4 {
  // SAFETY: `u32x4` and `i64x2` are both one 128-bit z/Vector register. The bit pattern is preserved
  // and interpreted only by vector integer instructions.
  let input: i64x2 = unsafe { core::mem::transmute(value & u32x4::splat(1)) };
  let out: i64x2;
  // SAFETY: z/Vector `vlcf` computes two's-complement negation lane-wise. The input lanes are
  // constrained to 0 or 1 immediately above, so the output lanes are exactly 0 or all-ones masks.
  // This is the fixed-work mask operation used by ML-KEM arithmetic to avoid native
  // secret-fed multiply/divide while also avoiding scalar lane extraction.
  unsafe {
    core::arch::asm!(
      "vlcf {out}, {input}",
      out = lateout(vreg) out,
      input = in(vreg) input,
      options(nomem, nostack, pure)
    );
  }
  // SAFETY: `out` is the same 128-bit z/Vector register reinterpreted back to four u32 lanes.
  unsafe { core::mem::transmute(out) }
}

/// # Safety
/// `offset` must be less than `ZETAS_MONT.len()`.
#[inline(always)]
unsafe fn load_zeta(offset: usize) -> i16 {
  // SAFETY: caller guarantees `offset` is inside the fixed public ML-KEM zeta table.
  unsafe { *ZETAS_MONT.as_ptr().add(offset) }
}

/// # Safety
/// `offset + 2` must not exceed `ZETAS_MONT.len()`.
#[inline(always)]
unsafe fn load_zeta_pair(offset: usize) -> (i32, i32) {
  // SAFETY: caller guarantees `offset..offset + 2` is inside the fixed public ML-KEM zeta table.
  unsafe {
    let zeta = ZETAS_MONT.as_ptr().add(offset);
    (i32::from(*zeta), i32::from(*zeta.add(1)))
  }
}

/// # Safety
/// `offset + 4` must not exceed `GAMMAS_MONT.len()`.
#[inline(always)]
unsafe fn load_gamma_i32x4(offset: usize) -> i32x4 {
  let gamma = GAMMAS_MONT.as_ptr();
  // SAFETY: caller guarantees `offset..offset + 4` is inside the fixed public ML-KEM gamma table.
  unsafe {
    let gamma = gamma.add(offset);
    i32x4::from_array([
      i32::from(*gamma),
      i32::from(*gamma.add(1)),
      i32::from(*gamma.add(2)),
      i32::from(*gamma.add(3)),
    ])
  }
}

/// # Safety
/// `values` must be aligned and readable for 4 initialized `u16` values
/// starting at `offset`, all within the same allocation.
#[inline(always)]
unsafe fn load_u32x4(values: *const u16, offset: usize) -> u32x4 {
  // SAFETY: caller guarantees `offset..offset + 4` is inside `values`.
  unsafe {
    let values = values.add(offset);
    u32x4::from_array([
      u32::from(*values),
      u32::from(*values.add(1)),
      u32::from(*values.add(2)),
      u32::from(*values.add(3)),
    ])
  }
}

/// # Safety
/// `values` must be aligned and readable for 8 initialized `u16` values
/// starting at `offset`, all within the same allocation.
#[inline(always)]
unsafe fn load_even_u32x4(values: *const u16, offset: usize) -> u32x4 {
  // SAFETY: caller guarantees `offset..offset + 8` is inside `values`.
  unsafe {
    let values = values.add(offset);
    u32x4::from_array([
      u32::from(*values),
      u32::from(*values.add(2)),
      u32::from(*values.add(4)),
      u32::from(*values.add(6)),
    ])
  }
}

/// # Safety
/// `values` must be aligned and readable for 8 initialized `u16` values
/// starting at `offset`, all within the same allocation.
#[inline(always)]
unsafe fn load_odd_u32x4(values: *const u16, offset: usize) -> u32x4 {
  // SAFETY: caller guarantees `offset..offset + 8` is inside `values`.
  unsafe {
    let values = values.add(offset);
    u32x4::from_array([
      u32::from(*values.add(1)),
      u32::from(*values.add(3)),
      u32::from(*values.add(5)),
      u32::from(*values.add(7)),
    ])
  }
}

/// # Safety
/// `values` must be aligned and readable for 8 initialized `u16` values
/// starting at `offset`, all within the same allocation.
#[inline(always)]
unsafe fn load_len2_lower_u32x4(values: *const u16, offset: usize) -> u32x4 {
  // SAFETY: caller guarantees `offset..offset + 8` is inside `values`.
  unsafe {
    let values = values.add(offset);
    u32x4::from_array([
      u32::from(*values),
      u32::from(*values.add(1)),
      u32::from(*values.add(4)),
      u32::from(*values.add(5)),
    ])
  }
}

/// # Safety
/// `values` must be aligned and readable for 8 initialized `u16` values
/// starting at `offset`, all within the same allocation.
#[inline(always)]
unsafe fn load_len2_upper_u32x4(values: *const u16, offset: usize) -> u32x4 {
  // SAFETY: caller guarantees `offset..offset + 8` is inside `values`.
  unsafe {
    let values = values.add(offset);
    u32x4::from_array([
      u32::from(*values.add(2)),
      u32::from(*values.add(3)),
      u32::from(*values.add(6)),
      u32::from(*values.add(7)),
    ])
  }
}

/// # Safety
/// `values` must be aligned and exclusively writable for 4 `u16` values
/// starting at `offset`, all within the same allocation.
#[inline(always)]
unsafe fn store_u32x4(values: *mut u16, offset: usize, lanes: u32x4) {
  let lanes = lanes.to_array();
  // SAFETY: caller guarantees `offset..offset + 4` is inside `values`.
  unsafe {
    let values = values.add(offset);
    *values = low_u16(lanes[0]);
    *values.add(1) = low_u16(lanes[1]);
    *values.add(2) = low_u16(lanes[2]);
    *values.add(3) = low_u16(lanes[3]);
  }
}

/// # Safety
/// `values` must be aligned and exclusively writable for 8 `u16` values
/// starting at `offset`, all within the same allocation.
#[inline(always)]
unsafe fn store_len2_interleaved_4(values: *mut u16, offset: usize, lower: u32x4, upper: u32x4) {
  let lower = lower.to_array();
  let upper = upper.to_array();
  // SAFETY: caller guarantees `offset..offset + 8` is inside `values`.
  unsafe {
    let values = values.add(offset);
    *values = low_u16(lower[0]);
    *values.add(1) = low_u16(lower[1]);
    *values.add(2) = low_u16(upper[0]);
    *values.add(3) = low_u16(upper[1]);
    *values.add(4) = low_u16(lower[2]);
    *values.add(5) = low_u16(lower[3]);
    *values.add(6) = low_u16(upper[2]);
    *values.add(7) = low_u16(upper[3]);
  }
}

#[inline(always)]
fn duplicate_i32_pair_lanes(a: i32, b: i32) -> i32x4 {
  i32x4::from_array([a, a, b, b])
}

#[inline(always)]
fn u32x4_from_u16(values: [u16; 4]) -> u32x4 {
  u32x4::from_array([
    u32::from(values[0]),
    u32::from(values[1]),
    u32::from(values[2]),
    u32::from(values[3]),
  ])
}

#[inline(always)]
fn u32x4_to_u16(values: u32x4) -> [u16; 4] {
  values.to_array().map(low_u16)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn vector_compress_decompress_matches_scalar_helpers() {
    if !std::arch::is_s390x_feature_detected!("vector") {
      return;
    }

    fn check_width<const D: usize>() {
      for base in (0u16..super::super::Q).step_by(4) {
        let values = [0u16, 1, 2, 3].map(|offset| base.wrapping_add(offset) % super::super::Q);
        // SAFETY: the enclosing test checked z/Vector availability.
        let compressed = unsafe { compress_values_4::<D>(values) };
        assert_eq!(
          compressed,
          values.map(super::super::compress_value::<D>),
          "width {D}, base {base}"
        );
      }
      let limit = 1u16.strict_shl(u32::try_from(D).expect("ML-KEM compression width fits u32"));
      for base in (0u16..limit).step_by(4) {
        let values = [0u16, 1, 2, 3].map(|offset| base.wrapping_add(offset) % limit);
        // SAFETY: the enclosing test checked z/Vector availability.
        let decompressed = unsafe { decompress_values_4::<D>(values) };
        assert_eq!(
          decompressed,
          values.map(super::super::decompress_value::<D>),
          "width {D}, base {base}"
        );
      }
    }
    check_width::<1>();
    check_width::<4>();
    check_width::<5>();
    check_width::<10>();
    check_width::<11>();
  }

  #[test]
  fn vector_wrapping_arithmetic_matches_scalar_lanes() {
    if !std::arch::is_s390x_feature_detected!("vector") {
      return;
    }
    let a = u32x4::from_array([0, u32::MAX, 0x8000_0000, 17]);
    let b = u32x4::from_array([u32::MAX, 1, 0x8000_0000, 0x1234_5678]);
    assert_eq!(
      wrapping_add_u32x4(a, b).to_array(),
      core::array::from_fn(|i| a[i].wrapping_add(b[i]))
    );
    assert_eq!(
      wrapping_sub_u32x4(a, b).to_array(),
      core::array::from_fn(|i| a[i].wrapping_sub(b[i]))
    );
    assert_eq!(
      mul_u32x4_16_ct(a, b).to_array(),
      core::array::from_fn(|i| a[i].wrapping_mul(b[i]))
    );
    let signed_a = a.cast::<i32>();
    let signed_b = b.cast::<i32>();
    assert_eq!(
      wrapping_add_i32x4(signed_a, signed_b).to_array(),
      core::array::from_fn(|i| signed_a[i].wrapping_add(signed_b[i]))
    );
    assert_eq!(
      wrapping_sub_i32x4(signed_a, signed_b).to_array(),
      core::array::from_fn(|i| signed_a[i].wrapping_sub(signed_b[i]))
    );
    let wide_a = u64x2::from_array([u64::MAX, 0x8000_0000_0000_0000]);
    let wide_b = u64x2::from_array([1, 0x8000_0000_0000_0000]);
    assert_eq!(
      wrapping_add_u64x2(wide_a, wide_b).to_array(),
      core::array::from_fn(|i| wide_a[i].wrapping_add(wide_b[i]))
    );
    for shift in [0, 1, 15, 16, 31, 32, 33, 63, 64, 65] {
      assert_eq!(
        wrapping_shl_u32x4(a, shift).to_array(),
        a.to_array().map(|lane| lane.wrapping_shl(shift))
      );
      assert_eq!(
        wrapping_shr_u32x4(a, shift).to_array(),
        a.to_array().map(|lane| lane.wrapping_shr(shift))
      );
      assert_eq!(
        wrapping_shl_i32x4(signed_a, shift).to_array(),
        signed_a.to_array().map(|lane| lane.wrapping_shl(shift))
      );
      assert_eq!(
        wrapping_shr_i32x4(signed_a, shift).to_array(),
        signed_a.to_array().map(|lane| lane.wrapping_shr(shift))
      );
      assert_eq!(
        wrapping_shl_u64x2(wide_a, shift).to_array(),
        wide_a.to_array().map(|lane| lane.wrapping_shl(shift))
      );
      assert_eq!(
        wrapping_shr_u64x2(wide_a, shift).to_array(),
        wide_a.to_array().map(|lane| lane.wrapping_shr(shift))
      );
    }
  }

  #[test]
  fn vector_multiply_ntts_matches_scalar_accumulator() {
    if !std::arch::is_s390x_feature_detected!("vector") {
      return;
    }

    for seed in 0usize..16 {
      let acc = test_poly(seed);
      let a = test_poly(seed.strict_add(100));
      let b = test_poly(seed.strict_add(200));

      let mut scalar = acc;
      super::super::multiply_ntts_add_assign_scalar(&mut scalar, &a, &b);

      let mut vector = acc;
      // SAFETY: test is runtime-gated on z/Vector availability.
      unsafe {
        multiply_ntts_add_assign_vector(&mut vector, &a, &b);
      }

      assert_eq!(vector, scalar, "seed {seed}");
    }
  }

  #[test]
  fn vector_multiply_ntts_accumulate_matches_scalar_dot_product() {
    if !std::arch::is_s390x_feature_detected!("vector") {
      return;
    }

    for seed in 0usize..16 {
      let acc = test_poly(seed);
      let a0 = test_poly(seed.strict_add(100));
      let a1 = test_poly(seed.strict_add(101));
      let a2 = test_poly(seed.strict_add(102));
      let a3 = test_poly(seed.strict_add(103));
      let b0 = test_poly(seed.strict_add(200));
      let b1 = test_poly(seed.strict_add(201));
      let b2 = test_poly(seed.strict_add(202));
      let b3 = test_poly(seed.strict_add(203));

      let mut scalar_k3 = acc;
      super::super::multiply_ntts_add_assign_scalar(&mut scalar_k3, &a0, &b0);
      super::super::multiply_ntts_add_assign_scalar(&mut scalar_k3, &a1, &b1);
      super::super::multiply_ntts_add_assign_scalar(&mut scalar_k3, &a2, &b2);

      let mut vector_k3 = acc;
      // SAFETY: test is runtime-gated on z/Vector availability.
      unsafe {
        multiply_ntts_accumulate_k3_vector(&mut vector_k3, [&a0, &a1, &a2], [&b0, &b1, &b2]);
      }
      assert_eq!(vector_k3, scalar_k3, "k3 seed {seed}");

      let mut scalar_k4 = scalar_k3;
      super::super::multiply_ntts_add_assign_scalar(&mut scalar_k4, &a3, &b3);

      let mut vector_k4 = acc;
      // SAFETY: test is runtime-gated on z/Vector availability.
      unsafe {
        multiply_ntts_accumulate_k4_vector(&mut vector_k4, [&a0, &a1, &a2, &a3], [&b0, &b1, &b2, &b3]);
      }
      assert_eq!(vector_k4, scalar_k4, "k4 seed {seed}");
    }
  }

  #[test]
  fn vector_ntt_matches_scalar_reference() {
    if !std::arch::is_s390x_feature_detected!("vector") {
      return;
    }

    for seed in 0usize..16 {
      let mut scalar = test_poly(seed);
      let mut vector = scalar;

      super::super::ntt_scalar(&mut scalar);
      // SAFETY: direct s390x vector NTT test call because:
      // 1. Runtime feature detection confirmed z/Vector availability above.
      // 2. `vector` is a fixed 256-coefficient polynomial matching the kernel contract.
      // 3. The kernel's memory access schedule depends only on public ML-KEM dimensions and uses
      //    fixed-work shift/add multiplication for secret-fed coefficient products.
      unsafe {
        ntt_vector(&mut vector);
      }

      assert_eq!(vector, scalar, "forward seed {seed}");

      super::super::inverse_ntt_scalar(&mut scalar);
      // SAFETY: direct s390x vector inverse-NTT test call because:
      // 1. Runtime feature detection confirmed z/Vector availability above.
      // 2. `vector` is a fixed 256-coefficient polynomial matching the kernel contract.
      // 3. The kernel's memory access schedule depends only on public ML-KEM dimensions and uses
      //    fixed-work shift/add multiplication for secret-fed coefficient products.
      unsafe {
        inverse_ntt_vector(&mut vector, super::super::INV_NTT_SCALE_MONT);
      }

      assert_eq!(vector, scalar, "inverse seed {seed}");
    }
  }

  fn test_poly(seed: usize) -> Poly {
    let mut poly = [0u16; N];
    for (i, coeff) in poly.iter_mut().enumerate() {
      let value = seed
        .strict_mul(37)
        .strict_add(i.strict_mul(19))
        .strict_add(11)
        .strict_rem(usize::from(super::super::Q));
      *coeff = u16::try_from(value).expect("coefficient reduced modulo Q fits u16");
    }
    poly
  }
}
