use core::simd::{
  i32x4,
  num::{SimdInt, SimdUint},
  u32x4,
};

use super::{GAMMAS_MONT, N, Poly, Q_HALF, Q_I32, Q_MONT_INV_U16, Q_U32, SAMPLE_NTT_ACC_CHUNK_COEFFS};

const Q_MONT_INV_I32: i32 = Q_MONT_INV_U16 as i16 as i32;

#[target_feature(enable = "vector")]
pub(super) unsafe fn compress_values_4<const D: usize>(values: [u16; 4]) -> [u16; 4] {
  let value = u32x4_from_u16(values);
  let numerator = (value << (D as u32)) + u32x4::splat(Q_HALF);
  u32x4_to_u16(div_q_compress_u32x4_ct(numerator) & u32x4::splat((1u32 << D) - 1))
}

#[target_feature(enable = "vector")]
pub(super) unsafe fn decompress_values_4<const D: usize>(values: [u16; 4]) -> [u16; 4] {
  let value = u32x4_from_u16(values);
  let scaled = mul_u32x4_16_ct(u32x4::splat(Q_U32), value) + u32x4::splat(1u32 << (D - 1));
  u32x4_to_u16(scaled >> (D as u32))
}

#[target_feature(enable = "vector")]
pub(super) unsafe fn multiply_ntts_add_assign_vector(acc: &mut Poly, a: &Poly, b: &Poly) {
  let mut coeff_offset = 0usize;
  while coeff_offset < N {
    multiply_ntts_add_assign_4(acc, a, coeff_offset, b, coeff_offset, coeff_offset / 2);
    coeff_offset = coeff_offset.strict_add(8);
  }
}

#[target_feature(enable = "vector")]
pub(super) unsafe fn multiply_ntts_add_assign_chunk_vector(
  acc: &mut Poly,
  a: &[u16; SAMPLE_NTT_ACC_CHUNK_COEFFS],
  b: &Poly,
  coeff_offset: usize,
) {
  debug_assert_eq!(coeff_offset % SAMPLE_NTT_ACC_CHUNK_COEFFS, 0);
  debug_assert!(coeff_offset.strict_add(SAMPLE_NTT_ACC_CHUNK_COEFFS) <= N);

  let mut local = 0usize;
  while local < SAMPLE_NTT_ACC_CHUNK_COEFFS {
    multiply_ntts_add_assign_4(
      acc,
      a,
      local,
      b,
      coeff_offset.strict_add(local),
      coeff_offset.strict_add(local) / 2,
    );
    local = local.strict_add(8);
  }
}

#[inline(always)]
fn multiply_ntts_add_assign_4(
  acc: &mut Poly,
  a: &[u16],
  a_offset: usize,
  b: &Poly,
  b_offset: usize,
  gamma_offset: usize,
) {
  let a0 = load_even_u32x4(a, a_offset).cast::<i32>();
  let a1 = load_odd_u32x4(a, a_offset).cast::<i32>();
  let b0 = load_even_u32x4(b, b_offset).cast::<i32>();
  let b1 = load_odd_u32x4(b, b_offset).cast::<i32>();
  let gamma = i32x4::from_array([
    i32::from(GAMMAS_MONT[gamma_offset]),
    i32::from(GAMMAS_MONT[gamma_offset.strict_add(1)]),
    i32::from(GAMMAS_MONT[gamma_offset.strict_add(2)]),
    i32::from(GAMMAS_MONT[gamma_offset.strict_add(3)]),
  ]);

  let a1b1 = montgomery_reduce_i32x4(mul_i32x4_16_ct(a1, b1));
  let a0b0 = mul_i32x4_16_ct(a0, b0);
  let a1b1_gamma = mul_i32x4_16_ct(a1b1, gamma);
  let c0 = signed_to_mod_q_i32x4(montgomery_reduce_i32x4(a0b0 + a1b1_gamma));

  let a0b1 = mul_i32x4_16_ct(a0, b1);
  let a1b0 = mul_i32x4_16_ct(a1, b0);
  let c1 = signed_to_mod_q_i32x4(montgomery_reduce_i32x4(a0b1 + a1b0));

  add_interleaved_4(acc, b_offset, c0, c1);
}

#[inline(always)]
fn add_interleaved_4(acc: &mut Poly, offset: usize, c0: u32x4, c1: u32x4) {
  let out0 = add_mod_u32x4(load_even_u32x4(acc, offset), c0).to_array();
  let out1 = add_mod_u32x4(load_odd_u32x4(acc, offset), c1).to_array();

  acc[offset] = out0[0] as u16;
  acc[offset.strict_add(1)] = out1[0] as u16;
  acc[offset.strict_add(2)] = out0[1] as u16;
  acc[offset.strict_add(3)] = out1[1] as u16;
  acc[offset.strict_add(4)] = out0[2] as u16;
  acc[offset.strict_add(5)] = out1[2] as u16;
  acc[offset.strict_add(6)] = out0[3] as u16;
  acc[offset.strict_add(7)] = out1[3] as u16;
}

#[inline(always)]
fn montgomery_reduce_i32x4(value: i32x4) -> i32x4 {
  let k = mul_i32x4_16_ct(sign_extend_i16_i32x4(value), i32x4::splat(Q_MONT_INV_I32));
  let c = mul_i32x4_16_ct(sign_extend_i16_i32x4(k), i32x4::splat(Q_I32)) >> 16;
  sign_extend_i16_i32x4((value >> 16) - c)
}

#[inline(always)]
fn signed_to_mod_q_i32x4(value: i32x4) -> u32x4 {
  (value + ((value >> 31) & i32x4::splat(Q_I32))).cast::<u32>()
}

#[inline(always)]
fn add_mod_u32x4(a: u32x4, b: u32x4) -> u32x4 {
  add_q_if_borrowed_u32x4((a + b) - u32x4::splat(Q_U32))
}

#[inline(always)]
fn add_q_if_borrowed_u32x4(value: u32x4) -> u32x4 {
  let borrow = value >> 31;
  value + (opaque_bit_u32x4(borrow).wrapping_neg() & u32x4::splat(Q_U32))
}

#[inline(always)]
fn sign_extend_i16_i32x4(value: i32x4) -> i32x4 {
  (value << 16) >> 16
}

#[inline(always)]
fn mul_i32x4_16_ct(a: i32x4, b: i32x4) -> i32x4 {
  let a_sign = a >> 31;
  let b_sign = b >> 31;
  let abs_a = ((a ^ a_sign) - a_sign).cast::<u32>();
  let abs_b = ((b ^ b_sign) - b_sign).cast::<u32>();
  let magnitude = mul_u32x4_16_ct(abs_a, abs_b);
  let sign = (a_sign ^ b_sign).cast::<u32>();
  ((magnitude ^ sign) - sign).cast::<i32>()
}

#[inline(always)]
fn mul_u32x4_16_ct(a: u32x4, b: u32x4) -> u32x4 {
  let mut acc = u32x4::splat(0);
  let mut bit = 0u32;
  while bit < 16 {
    let mask = opaque_bit_u32x4((b >> bit) & u32x4::splat(1)).wrapping_neg();
    acc = acc + ((a << bit) & mask);
    bit = bit.strict_add(1);
  }
  acc
}

#[inline(always)]
fn div_q_compress_u32x4_ct(value: u32x4) -> u32x4 {
  let mut quotient = u32x4::splat(0);
  let mut remainder = u32x4::splat(0);
  let mut bit = 23u32;
  while bit > 0 {
    bit = bit.strict_sub(1);
    remainder = (remainder << 1) | ((value >> bit) & u32x4::splat(1));
    let reduced = remainder - u32x4::splat(Q_U32);
    let borrow = reduced >> 31;
    let ge = opaque_bit_u32x4(borrow ^ u32x4::splat(1));
    remainder = add_q_if_borrowed_u32x4(reduced);
    quotient = quotient | (ge << bit);
  }
  quotient
}

#[inline(always)]
fn opaque_bit_u32x4(value: u32x4) -> u32x4 {
  let mut out = value & u32x4::splat(1);
  // SAFETY: this empty z/Vector asm block is intentionally a register-only compiler barrier.
  // The enclosing public helpers are `#[target_feature(enable = "vector")]`, and this module is
  // only dispatched after runtime z/Vector detection. It emits no instructions with timing behavior;
  // it only prevents LLVM from folding the fixed-work shift/add product back into native multiply.
  unsafe {
    core::arch::asm!("/* {out} */", out = inout(vreg) out, options(nomem, nostack));
  }
  out
}

#[inline(always)]
fn load_even_u32x4(values: &[u16], offset: usize) -> u32x4 {
  u32x4::from_array([
    u32::from(values[offset]),
    u32::from(values[offset.strict_add(2)]),
    u32::from(values[offset.strict_add(4)]),
    u32::from(values[offset.strict_add(6)]),
  ])
}

#[inline(always)]
fn load_odd_u32x4(values: &[u16], offset: usize) -> u32x4 {
  u32x4::from_array([
    u32::from(values[offset.strict_add(1)]),
    u32::from(values[offset.strict_add(3)]),
    u32::from(values[offset.strict_add(5)]),
    u32::from(values[offset.strict_add(7)]),
  ])
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
  let values = values.to_array();
  [values[0] as u16, values[1] as u16, values[2] as u16, values[3] as u16]
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn vector_compress_decompress_matches_scalar_helpers() {
    if !std::arch::is_s390x_feature_detected!("vector") {
      return;
    }

    for seed in 0u16..128 {
      let values = [
        seed % super::super::Q,
        seed.wrapping_mul(17).wrapping_add(3) % super::super::Q,
        seed.wrapping_mul(29).wrapping_add(11) % super::super::Q,
        seed.wrapping_mul(43).wrapping_add(19) % super::super::Q,
      ];

      // SAFETY: test is runtime-gated on z/Vector availability.
      let compressed = unsafe { compress_values_4::<10>(values) };
      let expected_compressed = [
        super::super::compress_value::<10>(values[0]),
        super::super::compress_value::<10>(values[1]),
        super::super::compress_value::<10>(values[2]),
        super::super::compress_value::<10>(values[3]),
      ];
      assert_eq!(compressed, expected_compressed, "compress seed {seed}");

      let decoded = [
        seed & 0x03ff,
        seed.wrapping_mul(5).wrapping_add(7) & 0x03ff,
        seed.wrapping_mul(9).wrapping_add(13) & 0x03ff,
        seed.wrapping_mul(13).wrapping_add(17) & 0x03ff,
      ];
      // SAFETY: test is runtime-gated on z/Vector availability.
      let decompressed = unsafe { decompress_values_4::<10>(decoded) };
      let expected_decompressed = [
        super::super::decompress_value::<10>(decoded[0]),
        super::super::decompress_value::<10>(decoded[1]),
        super::super::decompress_value::<10>(decoded[2]),
        super::super::decompress_value::<10>(decoded[3]),
      ];
      assert_eq!(decompressed, expected_decompressed, "decompress seed {seed}");
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

  fn test_poly(seed: usize) -> Poly {
    let mut poly = [0u16; N];
    for (i, coeff) in poly.iter_mut().enumerate() {
      *coeff =
        ((seed.strict_mul(37).strict_add(i.strict_mul(19)).strict_add(11)) % usize::from(super::super::Q)) as u16;
    }
    poly
  }
}
