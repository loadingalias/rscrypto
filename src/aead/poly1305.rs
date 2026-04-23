#![allow(clippy::indexing_slicing)]

//! Portable Poly1305 core.

#[cfg(feature = "std")]
use crate::backend::cache::OnceCache;
use crate::{
  aead::{
    LengthOverflow,
    targets::{AeadPrimitive, select_backend},
  },
  platform::{Arch, Caps},
  traits::ct,
};

const LIMB_MASK: u32 = 0x03ff_ffff;
const FULL_BLOCK_HIBIT: u32 = 1 << 24;

type ComputeBlockFn = fn(&mut State, &[u8; 16], bool);

#[cfg(feature = "std")]
static XCHACHA20POLY1305_COMPUTE_BLOCK_DISPATCH: OnceCache<ComputeBlockFn> = OnceCache::new();
#[cfg(feature = "std")]
static CHACHA20POLY1305_COMPUTE_BLOCK_DISPATCH: OnceCache<ComputeBlockFn> = OnceCache::new();

#[inline]
fn load_u32_le(input: &[u8]) -> u32 {
  let mut bytes = [0u8; 4];
  bytes.copy_from_slice(input);
  u32::from_le_bytes(bytes)
}

#[cfg(any(
  all(target_arch = "powerpc64", target_endian = "little"),
  target_arch = "riscv64",
  target_arch = "s390x",
))]
#[inline]
fn compute_block_scalar_reduction(
  state: &mut State,
  block: &[u8; 16],
  partial: bool,
  mut sum4_mul: impl FnMut([u32; 4], [u32; 4]) -> u64,
) {
  let hibit = if partial { 0 } else { FULL_BLOCK_HIBIT };

  let r0 = state.r[0];
  let r1 = state.r[1];
  let r2 = state.r[2];
  let r3 = state.r[3];
  let r4 = state.r[4];

  let s1 = r1 * 5;
  let s2 = r2 * 5;
  let s3 = r3 * 5;
  let s4 = r4 * 5;

  let mut h0 = state.h[0];
  let mut h1 = state.h[1];
  let mut h2 = state.h[2];
  let mut h3 = state.h[3];
  let mut h4 = state.h[4];

  h0 = h0.wrapping_add(load_u32_le(&block[0..4]) & LIMB_MASK);
  h1 = h1.wrapping_add((load_u32_le(&block[3..7]) >> 2) & LIMB_MASK);
  h2 = h2.wrapping_add((load_u32_le(&block[6..10]) >> 4) & LIMB_MASK);
  h3 = h3.wrapping_add((load_u32_le(&block[9..13]) >> 6) & LIMB_MASK);
  h4 = h4.wrapping_add((load_u32_le(&block[12..16]) >> 8) | hibit);

  let d0 = sum4_mul([h0, h1, h2, h3], [r0, s4, s3, s2]) + (u64::from(h4) * u64::from(s1));
  let mut d1 = sum4_mul([h0, h1, h2, h3], [r1, r0, s4, s3]) + (u64::from(h4) * u64::from(s2));
  let mut d2 = sum4_mul([h0, h1, h2, h3], [r2, r1, r0, s4]) + (u64::from(h4) * u64::from(s3));
  let mut d3 = sum4_mul([h0, h1, h2, h3], [r3, r2, r1, r0]) + (u64::from(h4) * u64::from(s4));
  let mut d4 = sum4_mul([h0, h1, h2, h3], [r4, r3, r2, r1]) + (u64::from(h4) * u64::from(r0));

  let mut c = (d0 >> 26) as u32;
  h0 = (d0 as u32) & LIMB_MASK;
  d1 += u64::from(c);

  c = (d1 >> 26) as u32;
  h1 = (d1 as u32) & LIMB_MASK;
  d2 += u64::from(c);

  c = (d2 >> 26) as u32;
  h2 = (d2 as u32) & LIMB_MASK;
  d3 += u64::from(c);

  c = (d3 >> 26) as u32;
  h3 = (d3 as u32) & LIMB_MASK;
  d4 += u64::from(c);

  c = (d4 >> 26) as u32;
  h4 = (d4 as u32) & LIMB_MASK;
  h0 = h0.wrapping_add(c * 5);

  c = h0 >> 26;
  h0 &= LIMB_MASK;
  h1 = h1.wrapping_add(c);

  state.h = [h0, h1, h2, h3, h4];
}

#[inline]
fn current_caps() -> Caps {
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
fn compute_block_resolved(primitive: AeadPrimitive) -> ComputeBlockFn {
  #[cfg(feature = "std")]
  {
    match primitive {
      AeadPrimitive::XChaCha20Poly1305 => {
        XCHACHA20POLY1305_COMPUTE_BLOCK_DISPATCH.get_or_init(|| resolve_compute_block(primitive))
      }
      AeadPrimitive::ChaCha20Poly1305 => {
        CHACHA20POLY1305_COMPUTE_BLOCK_DISPATCH.get_or_init(|| resolve_compute_block(primitive))
      }
      _ => resolve_compute_block(primitive),
    }
  }

  #[cfg(not(feature = "std"))]
  {
    resolve_compute_block(primitive)
  }
}

#[inline]
fn resolve_compute_block(primitive: AeadPrimitive) -> ComputeBlockFn {
  match select_backend(primitive, Arch::current(), current_caps()) {
    #[cfg(target_arch = "wasm32")]
    crate::aead::targets::AeadBackend::WasmSimd128 => wasm_simd128::compute_block,
    #[cfg(target_arch = "x86_64")]
    crate::aead::targets::AeadBackend::X86Avx512 => x86_avx512::compute_block,
    #[cfg(target_arch = "x86_64")]
    crate::aead::targets::AeadBackend::X86Avx2 => x86_avx2::compute_block,
    #[cfg(target_arch = "aarch64")]
    crate::aead::targets::AeadBackend::Aarch64Neon => aarch64_neon::compute_block,
    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    crate::aead::targets::AeadBackend::PowerVector => power_vsx::compute_block,
    #[cfg(target_arch = "s390x")]
    crate::aead::targets::AeadBackend::S390xVector => s390x_vector::compute_block,
    #[cfg(target_arch = "riscv64")]
    crate::aead::targets::AeadBackend::Riscv64Vector => riscv64_vector::compute_block,
    _ => State::compute_block_portable,
  }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_block_x86_avx2(state: &mut State, block: &[u8; 16], partial: bool) {
  use core::arch::x86_64::{__m256i, _mm256_mul_epu32, _mm256_setr_epi32, _mm256_storeu_si256};

  #[inline(always)]
  fn sum4_mul(lhs: [u32; 4], rhs: [u32; 4]) -> u64 {
    // SAFETY: the enclosing kernel enables AVX2 and the destination array is a valid
    // unaligned store target for one `__m256i`.
    unsafe {
      let a = _mm256_setr_epi32(lhs[0] as i32, 0, lhs[1] as i32, 0, lhs[2] as i32, 0, lhs[3] as i32, 0);
      let b = _mm256_setr_epi32(rhs[0] as i32, 0, rhs[1] as i32, 0, rhs[2] as i32, 0, rhs[3] as i32, 0);
      let products = _mm256_mul_epu32(a, b);
      let mut lanes = [0u64; 4];
      _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, products);
      lanes[0] + lanes[1] + lanes[2] + lanes[3]
    }
  }

  let hibit = if partial { 0 } else { FULL_BLOCK_HIBIT };

  let r0 = state.r[0];
  let r1 = state.r[1];
  let r2 = state.r[2];
  let r3 = state.r[3];
  let r4 = state.r[4];

  let s1 = r1 * 5;
  let s2 = r2 * 5;
  let s3 = r3 * 5;
  let s4 = r4 * 5;

  let mut h0 = state.h[0];
  let mut h1 = state.h[1];
  let mut h2 = state.h[2];
  let mut h3 = state.h[3];
  let mut h4 = state.h[4];

  h0 = h0.wrapping_add(load_u32_le(&block[0..4]) & LIMB_MASK);
  h1 = h1.wrapping_add((load_u32_le(&block[3..7]) >> 2) & LIMB_MASK);
  h2 = h2.wrapping_add((load_u32_le(&block[6..10]) >> 4) & LIMB_MASK);
  h3 = h3.wrapping_add((load_u32_le(&block[9..13]) >> 6) & LIMB_MASK);
  h4 = h4.wrapping_add((load_u32_le(&block[12..16]) >> 8) | hibit);

  let d0 = sum4_mul([h0, h1, h2, h3], [r0, s4, s3, s2]) + (u64::from(h4) * u64::from(s1));
  let mut d1 = sum4_mul([h0, h1, h2, h3], [r1, r0, s4, s3]) + (u64::from(h4) * u64::from(s2));
  let mut d2 = sum4_mul([h0, h1, h2, h3], [r2, r1, r0, s4]) + (u64::from(h4) * u64::from(s3));
  let mut d3 = sum4_mul([h0, h1, h2, h3], [r3, r2, r1, r0]) + (u64::from(h4) * u64::from(s4));
  let mut d4 = sum4_mul([h0, h1, h2, h3], [r4, r3, r2, r1]) + (u64::from(h4) * u64::from(r0));

  let mut c = (d0 >> 26) as u32;
  h0 = (d0 as u32) & LIMB_MASK;
  d1 += u64::from(c);

  c = (d1 >> 26) as u32;
  h1 = (d1 as u32) & LIMB_MASK;
  d2 += u64::from(c);

  c = (d2 >> 26) as u32;
  h2 = (d2 as u32) & LIMB_MASK;
  d3 += u64::from(c);

  c = (d3 >> 26) as u32;
  h3 = (d3 as u32) & LIMB_MASK;
  d4 += u64::from(c);

  c = (d4 >> 26) as u32;
  h4 = (d4 as u32) & LIMB_MASK;
  h0 = h0.wrapping_add(c * 5);

  c = h0 >> 26;
  h0 &= LIMB_MASK;
  h1 = h1.wrapping_add(c);

  state.h = [h0, h1, h2, h3, h4];
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
unsafe fn compute_block_x86_avx512(state: &mut State, block: &[u8; 16], partial: bool) {
  use core::arch::x86_64::{__m512i, _mm512_mul_epu32, _mm512_setr_epi32, _mm512_storeu_si512};

  #[inline(always)]
  fn pair_sum4_mul(lhs: [u32; 4], rhs_lo: [u32; 4], rhs_hi: [u32; 4]) -> (u64, u64) {
    // SAFETY: the enclosing kernel enables AVX-512F, and the destination array is a valid
    // unaligned store target for one `__m512i`.
    unsafe {
      let a = _mm512_setr_epi32(
        lhs[0] as i32,
        0,
        lhs[1] as i32,
        0,
        lhs[2] as i32,
        0,
        lhs[3] as i32,
        0,
        lhs[0] as i32,
        0,
        lhs[1] as i32,
        0,
        lhs[2] as i32,
        0,
        lhs[3] as i32,
        0,
      );
      let b = _mm512_setr_epi32(
        rhs_lo[0] as i32,
        0,
        rhs_lo[1] as i32,
        0,
        rhs_lo[2] as i32,
        0,
        rhs_lo[3] as i32,
        0,
        rhs_hi[0] as i32,
        0,
        rhs_hi[1] as i32,
        0,
        rhs_hi[2] as i32,
        0,
        rhs_hi[3] as i32,
        0,
      );
      let products = _mm512_mul_epu32(a, b);
      let mut lanes = [0u64; 8];
      _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, products);
      (
        lanes[0] + lanes[1] + lanes[2] + lanes[3],
        lanes[4] + lanes[5] + lanes[6] + lanes[7],
      )
    }
  }

  #[inline(always)]
  fn single_sum4_mul(lhs: [u32; 4], rhs: [u32; 4]) -> u64 {
    pair_sum4_mul(lhs, rhs, [0; 4]).0
  }

  let hibit = if partial { 0 } else { FULL_BLOCK_HIBIT };

  let r0 = state.r[0];
  let r1 = state.r[1];
  let r2 = state.r[2];
  let r3 = state.r[3];
  let r4 = state.r[4];

  let s1 = r1 * 5;
  let s2 = r2 * 5;
  let s3 = r3 * 5;
  let s4 = r4 * 5;

  let mut h0 = state.h[0];
  let mut h1 = state.h[1];
  let mut h2 = state.h[2];
  let mut h3 = state.h[3];
  let mut h4 = state.h[4];

  h0 = h0.wrapping_add(load_u32_le(&block[0..4]) & LIMB_MASK);
  h1 = h1.wrapping_add((load_u32_le(&block[3..7]) >> 2) & LIMB_MASK);
  h2 = h2.wrapping_add((load_u32_le(&block[6..10]) >> 4) & LIMB_MASK);
  h3 = h3.wrapping_add((load_u32_le(&block[9..13]) >> 6) & LIMB_MASK);
  h4 = h4.wrapping_add((load_u32_le(&block[12..16]) >> 8) | hibit);

  let (d0_base, d1_base) = pair_sum4_mul([h0, h1, h2, h3], [r0, s4, s3, s2], [r1, r0, s4, s3]);
  let (d2_base, d3_base) = pair_sum4_mul([h0, h1, h2, h3], [r2, r1, r0, s4], [r3, r2, r1, r0]);
  let d4_base = single_sum4_mul([h0, h1, h2, h3], [r4, r3, r2, r1]);

  let d0 = d0_base + (u64::from(h4) * u64::from(s1));
  let mut d1 = d1_base + (u64::from(h4) * u64::from(s2));
  let mut d2 = d2_base + (u64::from(h4) * u64::from(s3));
  let mut d3 = d3_base + (u64::from(h4) * u64::from(s4));
  let mut d4 = d4_base + (u64::from(h4) * u64::from(r0));

  let mut c = (d0 >> 26) as u32;
  h0 = (d0 as u32) & LIMB_MASK;
  d1 += u64::from(c);

  c = (d1 >> 26) as u32;
  h1 = (d1 as u32) & LIMB_MASK;
  d2 += u64::from(c);

  c = (d2 >> 26) as u32;
  h2 = (d2 as u32) & LIMB_MASK;
  d3 += u64::from(c);

  c = (d3 >> 26) as u32;
  h3 = (d3 as u32) & LIMB_MASK;
  d4 += u64::from(c);

  c = (d4 >> 26) as u32;
  h4 = (d4 as u32) & LIMB_MASK;
  h0 = h0.wrapping_add(c * 5);

  c = h0 >> 26;
  h0 &= LIMB_MASK;
  h1 = h1.wrapping_add(c);

  state.h = [h0, h1, h2, h3, h4];
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn compute_block_aarch64_neon(state: &mut State, block: &[u8; 16], partial: bool) {
  use core::arch::aarch64::{uint64x2_t, vaddq_u64, vcreate_u32, vmull_u32, vst1q_u64};

  #[inline(always)]
  fn sum4_mul(lhs: [u32; 4], rhs: [u32; 4]) -> u64 {
    // SAFETY: the enclosing kernel enables NEON and the destination arrays hold exactly one
    // `uint64x2_t` each for unaligned stores.
    unsafe {
      let lo = vmull_u32(
        vcreate_u32((u64::from(lhs[1]) << 32) | u64::from(lhs[0])),
        vcreate_u32((u64::from(rhs[1]) << 32) | u64::from(rhs[0])),
      );
      let hi = vmull_u32(
        vcreate_u32((u64::from(lhs[3]) << 32) | u64::from(lhs[2])),
        vcreate_u32((u64::from(rhs[3]) << 32) | u64::from(rhs[2])),
      );
      let sum: uint64x2_t = vaddq_u64(lo, hi);
      let mut lanes = [0u64; 2];
      vst1q_u64(lanes.as_mut_ptr(), sum);
      lanes[0] + lanes[1]
    }
  }

  let hibit = if partial { 0 } else { FULL_BLOCK_HIBIT };

  let r0 = state.r[0];
  let r1 = state.r[1];
  let r2 = state.r[2];
  let r3 = state.r[3];
  let r4 = state.r[4];

  let s1 = r1 * 5;
  let s2 = r2 * 5;
  let s3 = r3 * 5;
  let s4 = r4 * 5;

  let mut h0 = state.h[0];
  let mut h1 = state.h[1];
  let mut h2 = state.h[2];
  let mut h3 = state.h[3];
  let mut h4 = state.h[4];

  h0 = h0.wrapping_add(load_u32_le(&block[0..4]) & LIMB_MASK);
  h1 = h1.wrapping_add((load_u32_le(&block[3..7]) >> 2) & LIMB_MASK);
  h2 = h2.wrapping_add((load_u32_le(&block[6..10]) >> 4) & LIMB_MASK);
  h3 = h3.wrapping_add((load_u32_le(&block[9..13]) >> 6) & LIMB_MASK);
  h4 = h4.wrapping_add((load_u32_le(&block[12..16]) >> 8) | hibit);

  let d0 = sum4_mul([h0, h1, h2, h3], [r0, s4, s3, s2]) + (u64::from(h4) * u64::from(s1));
  let mut d1 = sum4_mul([h0, h1, h2, h3], [r1, r0, s4, s3]) + (u64::from(h4) * u64::from(s2));
  let mut d2 = sum4_mul([h0, h1, h2, h3], [r2, r1, r0, s4]) + (u64::from(h4) * u64::from(s3));
  let mut d3 = sum4_mul([h0, h1, h2, h3], [r3, r2, r1, r0]) + (u64::from(h4) * u64::from(s4));
  let mut d4 = sum4_mul([h0, h1, h2, h3], [r4, r3, r2, r1]) + (u64::from(h4) * u64::from(r0));

  let mut c = (d0 >> 26) as u32;
  h0 = (d0 as u32) & LIMB_MASK;
  d1 += u64::from(c);

  c = (d1 >> 26) as u32;
  h1 = (d1 as u32) & LIMB_MASK;
  d2 += u64::from(c);

  c = (d2 >> 26) as u32;
  h2 = (d2 as u32) & LIMB_MASK;
  d3 += u64::from(c);

  c = (d3 >> 26) as u32;
  h3 = (d3 as u32) & LIMB_MASK;
  d4 += u64::from(c);

  c = (d4 >> 26) as u32;
  h4 = (d4 as u32) & LIMB_MASK;
  h0 = h0.wrapping_add(c * 5);

  c = h0 >> 26;
  h0 &= LIMB_MASK;
  h1 = h1.wrapping_add(c);

  state.h = [h0, h1, h2, h3, h4];
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn compute_block_wasm_simd128(state: &mut State, block: &[u8; 16], partial: bool) {
  use core::arch::wasm32::{i64x2_add, u32x4, u64x2_extmul_high_u32x4, u64x2_extmul_low_u32x4, u64x2_extract_lane};

  #[inline(always)]
  fn sum4_mul(lhs: [u32; 4], rhs: [u32; 4]) -> u64 {
    let a = u32x4(lhs[0], lhs[1], lhs[2], lhs[3]);
    let b = u32x4(rhs[0], rhs[1], rhs[2], rhs[3]);
    let lo = u64x2_extmul_low_u32x4(a, b);
    let hi = u64x2_extmul_high_u32x4(a, b);
    let sum = i64x2_add(lo, hi);
    u64x2_extract_lane::<0>(sum) + u64x2_extract_lane::<1>(sum)
  }

  let hibit = if partial { 0 } else { FULL_BLOCK_HIBIT };

  let r0 = state.r[0];
  let r1 = state.r[1];
  let r2 = state.r[2];
  let r3 = state.r[3];
  let r4 = state.r[4];

  let s1 = r1 * 5;
  let s2 = r2 * 5;
  let s3 = r3 * 5;
  let s4 = r4 * 5;

  let mut h0 = state.h[0];
  let mut h1 = state.h[1];
  let mut h2 = state.h[2];
  let mut h3 = state.h[3];
  let mut h4 = state.h[4];

  h0 = h0.wrapping_add(load_u32_le(&block[0..4]) & LIMB_MASK);
  h1 = h1.wrapping_add((load_u32_le(&block[3..7]) >> 2) & LIMB_MASK);
  h2 = h2.wrapping_add((load_u32_le(&block[6..10]) >> 4) & LIMB_MASK);
  h3 = h3.wrapping_add((load_u32_le(&block[9..13]) >> 6) & LIMB_MASK);
  h4 = h4.wrapping_add((load_u32_le(&block[12..16]) >> 8) | hibit);

  let d0 = sum4_mul([h0, h1, h2, h3], [r0, s4, s3, s2]) + (u64::from(h4) * u64::from(s1));
  let mut d1 = sum4_mul([h0, h1, h2, h3], [r1, r0, s4, s3]) + (u64::from(h4) * u64::from(s2));
  let mut d2 = sum4_mul([h0, h1, h2, h3], [r2, r1, r0, s4]) + (u64::from(h4) * u64::from(s3));
  let mut d3 = sum4_mul([h0, h1, h2, h3], [r3, r2, r1, r0]) + (u64::from(h4) * u64::from(s4));
  let mut d4 = sum4_mul([h0, h1, h2, h3], [r4, r3, r2, r1]) + (u64::from(h4) * u64::from(r0));

  let mut c = (d0 >> 26) as u32;
  h0 = (d0 as u32) & LIMB_MASK;
  d1 += u64::from(c);

  c = (d1 >> 26) as u32;
  h1 = (d1 as u32) & LIMB_MASK;
  d2 += u64::from(c);

  c = (d2 >> 26) as u32;
  h2 = (d2 as u32) & LIMB_MASK;
  d3 += u64::from(c);

  c = (d3 >> 26) as u32;
  h3 = (d3 as u32) & LIMB_MASK;
  d4 += u64::from(c);

  c = (d4 >> 26) as u32;
  h4 = (d4 as u32) & LIMB_MASK;
  h0 = h0.wrapping_add(c * 5);

  c = h0 >> 26;
  h0 &= LIMB_MASK;
  h1 = h1.wrapping_add(c);

  state.h = [h0, h1, h2, h3, h4];
}

#[derive(Clone, Default)]
struct State {
  r: [u32; 5],
  h: [u32; 5],
  pad: [u32; 4],
}

impl State {
  #[inline]
  fn new(key: &[u8; 32]) -> Self {
    Self {
      r: [
        load_u32_le(&key[0..4]) & LIMB_MASK,
        (load_u32_le(&key[3..7]) >> 2) & 0x03ff_ff03,
        (load_u32_le(&key[6..10]) >> 4) & 0x03ff_c0ff,
        (load_u32_le(&key[9..13]) >> 6) & 0x03f0_3fff,
        (load_u32_le(&key[12..16]) >> 8) & 0x000f_ffff,
      ],
      h: [0u32; 5],
      pad: [
        load_u32_le(&key[16..20]),
        load_u32_le(&key[20..24]),
        load_u32_le(&key[24..28]),
        load_u32_le(&key[28..32]),
      ],
    }
  }

  fn compute_block_portable(&mut self, block: &[u8; 16], partial: bool) {
    let hibit = if partial { 0 } else { FULL_BLOCK_HIBIT };

    let r0 = self.r[0];
    let r1 = self.r[1];
    let r2 = self.r[2];
    let r3 = self.r[3];
    let r4 = self.r[4];

    let s1 = r1 * 5;
    let s2 = r2 * 5;
    let s3 = r3 * 5;
    let s4 = r4 * 5;

    let mut h0 = self.h[0];
    let mut h1 = self.h[1];
    let mut h2 = self.h[2];
    let mut h3 = self.h[3];
    let mut h4 = self.h[4];

    h0 = h0.wrapping_add(load_u32_le(&block[0..4]) & LIMB_MASK);
    h1 = h1.wrapping_add((load_u32_le(&block[3..7]) >> 2) & LIMB_MASK);
    h2 = h2.wrapping_add((load_u32_le(&block[6..10]) >> 4) & LIMB_MASK);
    h3 = h3.wrapping_add((load_u32_le(&block[9..13]) >> 6) & LIMB_MASK);
    h4 = h4.wrapping_add((load_u32_le(&block[12..16]) >> 8) | hibit);

    let d0 = (u64::from(h0) * u64::from(r0))
      + (u64::from(h1) * u64::from(s4))
      + (u64::from(h2) * u64::from(s3))
      + (u64::from(h3) * u64::from(s2))
      + (u64::from(h4) * u64::from(s1));
    let mut d1 = (u64::from(h0) * u64::from(r1))
      + (u64::from(h1) * u64::from(r0))
      + (u64::from(h2) * u64::from(s4))
      + (u64::from(h3) * u64::from(s3))
      + (u64::from(h4) * u64::from(s2));
    let mut d2 = (u64::from(h0) * u64::from(r2))
      + (u64::from(h1) * u64::from(r1))
      + (u64::from(h2) * u64::from(r0))
      + (u64::from(h3) * u64::from(s4))
      + (u64::from(h4) * u64::from(s3));
    let mut d3 = (u64::from(h0) * u64::from(r3))
      + (u64::from(h1) * u64::from(r2))
      + (u64::from(h2) * u64::from(r1))
      + (u64::from(h3) * u64::from(r0))
      + (u64::from(h4) * u64::from(s4));
    let mut d4 = (u64::from(h0) * u64::from(r4))
      + (u64::from(h1) * u64::from(r3))
      + (u64::from(h2) * u64::from(r2))
      + (u64::from(h3) * u64::from(r1))
      + (u64::from(h4) * u64::from(r0));

    let mut c = (d0 >> 26) as u32;
    h0 = (d0 as u32) & LIMB_MASK;
    d1 += u64::from(c);

    c = (d1 >> 26) as u32;
    h1 = (d1 as u32) & LIMB_MASK;
    d2 += u64::from(c);

    c = (d2 >> 26) as u32;
    h2 = (d2 as u32) & LIMB_MASK;
    d3 += u64::from(c);

    c = (d3 >> 26) as u32;
    h3 = (d3 as u32) & LIMB_MASK;
    d4 += u64::from(c);

    c = (d4 >> 26) as u32;
    h4 = (d4 as u32) & LIMB_MASK;
    h0 = h0.wrapping_add(c * 5);

    c = h0 >> 26;
    h0 &= LIMB_MASK;
    h1 = h1.wrapping_add(c);

    self.h = [h0, h1, h2, h3, h4];
  }

  #[cfg(test)]
  fn update_message(&mut self, message: &[u8], compute_block: ComputeBlockFn) {
    let mut blocks = message.chunks_exact(16);
    for chunk in &mut blocks {
      let mut block = [0u8; 16];
      block.copy_from_slice(chunk);
      compute_block(self, &block, false);
    }

    let remainder = blocks.remainder();
    if remainder.is_empty() {
      return;
    }

    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    block[remainder.len()] = 1;
    compute_block(self, &block, true);
  }

  fn update_padded_segment(&mut self, segment: &[u8], compute_block: ComputeBlockFn) {
    let mut blocks = segment.chunks_exact(16);
    for chunk in &mut blocks {
      let mut block = [0u8; 16];
      block.copy_from_slice(chunk);
      compute_block(self, &block, false);
    }

    let remainder = blocks.remainder();
    if remainder.is_empty() {
      return;
    }

    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    compute_block(self, &block, false);
  }

  fn finalize(self) -> [u8; 16] {
    let mut h0 = self.h[0];
    let mut h1 = self.h[1];
    let mut h2 = self.h[2];
    let mut h3 = self.h[3];
    let mut h4 = self.h[4];

    let mut c = h1 >> 26;
    h1 &= LIMB_MASK;
    h2 = h2.wrapping_add(c);

    c = h2 >> 26;
    h2 &= LIMB_MASK;
    h3 = h3.wrapping_add(c);

    c = h3 >> 26;
    h3 &= LIMB_MASK;
    h4 = h4.wrapping_add(c);

    c = h4 >> 26;
    h4 &= LIMB_MASK;
    h0 = h0.wrapping_add(c * 5);

    c = h0 >> 26;
    h0 &= LIMB_MASK;
    h1 = h1.wrapping_add(c);

    let mut g0 = h0.wrapping_add(5);
    c = g0 >> 26;
    g0 &= LIMB_MASK;

    let mut g1 = h1.wrapping_add(c);
    c = g1 >> 26;
    g1 &= LIMB_MASK;

    let mut g2 = h2.wrapping_add(c);
    c = g2 >> 26;
    g2 &= LIMB_MASK;

    let mut g3 = h3.wrapping_add(c);
    c = g3 >> 26;
    g3 &= LIMB_MASK;

    let mut g4 = h4.wrapping_add(c).wrapping_sub(1 << 26);

    let mut mask = (g4 >> 31).wrapping_sub(1);
    g0 &= mask;
    g1 &= mask;
    g2 &= mask;
    g3 &= mask;
    g4 &= mask;
    mask = !mask;

    h0 = (h0 & mask) | g0;
    h1 = (h1 & mask) | g1;
    h2 = (h2 & mask) | g2;
    h3 = (h3 & mask) | g3;
    h4 = (h4 & mask) | g4;

    h0 |= h1 << 26;
    h1 = (h1 >> 6) | (h2 << 20);
    h2 = (h2 >> 12) | (h3 << 14);
    h3 = (h3 >> 18) | (h4 << 8);

    let mut f = u64::from(h0) + u64::from(self.pad[0]);
    h0 = f as u32;
    f = u64::from(h1) + u64::from(self.pad[1]) + (f >> 32);
    h1 = f as u32;
    f = u64::from(h2) + u64::from(self.pad[2]) + (f >> 32);
    h2 = f as u32;
    f = u64::from(h3) + u64::from(self.pad[3]) + (f >> 32);
    h3 = f as u32;

    let mut tag = [0u8; 16];
    tag[0..4].copy_from_slice(&h0.to_le_bytes());
    tag[4..8].copy_from_slice(&h1.to_le_bytes());
    tag[8..12].copy_from_slice(&h2.to_le_bytes());
    tag[12..16].copy_from_slice(&h3.to_le_bytes());
    tag
  }
}

#[cfg(test)]
#[must_use]
pub(crate) fn authenticate(message: &[u8], key: &[u8; 32]) -> [u8; 16] {
  let mut state = State::new(key);
  state.update_message(message, State::compute_block_portable);
  state.finalize()
}

pub(crate) fn authenticate_aead(
  primitive: AeadPrimitive,
  aad: &[u8],
  ciphertext: &[u8],
  key: &[u8; 32],
) -> Result<[u8; 16], LengthOverflow> {
  let lengths = super::AeadByteLengths::try_new(aad.len(), ciphertext.len())?;

  #[cfg(target_arch = "x86_64")]
  {
    use crate::platform::caps::x86;
    if lengths.total_at_least(64) && current_caps().has(x86::AVX2) {
      return Ok(avx2_par4::authenticate_aead_par4(aad, ciphertext, key, lengths));
    }
  }
  authenticate_aead_with(aad, ciphertext, key, compute_block_resolved(primitive), lengths)
}

fn authenticate_aead_with(
  aad: &[u8],
  ciphertext: &[u8],
  key: &[u8; 32],
  compute_block: ComputeBlockFn,
  lengths: super::AeadByteLengths,
) -> Result<[u8; 16], LengthOverflow> {
  let mut state = State::new(key);
  state.update_padded_segment(aad, compute_block);
  state.update_padded_segment(ciphertext, compute_block);

  let mut length_block = lengths.to_le_bytes_block();
  compute_block(&mut state, &length_block, false);

  let tag = state.finalize();
  ct::zeroize(&mut length_block);
  Ok(tag)
}

#[cfg(target_arch = "aarch64")]
#[path = "poly1305/aarch64_neon.rs"]
mod aarch64_neon;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[path = "poly1305/x86_64_avx2_par4.rs"]
mod avx2_par4;
#[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
#[path = "poly1305/powerpc64_vsx.rs"]
mod power_vsx;
#[cfg(target_arch = "riscv64")]
#[path = "poly1305/riscv64_vector.rs"]
mod riscv64_vector;
#[cfg(target_arch = "s390x")]
#[path = "poly1305/s390x_vector.rs"]
mod s390x_vector;
#[cfg(target_arch = "wasm32")]
#[path = "poly1305/wasm32_simd128.rs"]
mod wasm_simd128;
#[cfg(target_arch = "x86_64")]
#[path = "poly1305/x86_64_avx2.rs"]
mod x86_avx2;
#[cfg(target_arch = "x86_64")]
#[path = "poly1305/x86_64_avx512.rs"]
mod x86_avx512;
#[cfg(test)]
mod tests {
  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "riscv64"))]
  use alloc::vec::Vec;

  use super::authenticate;
  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "riscv64"))]
  use super::{ComputeBlockFn, State, authenticate_aead_with};
  use crate::aead::{AeadByteLengths, targets::AeadPrimitive};
  #[cfg(target_arch = "aarch64")]
  use crate::platform::caps::aarch64;
  #[cfg(target_arch = "riscv64")]
  use crate::platform::caps::riscv;
  #[cfg(target_arch = "x86_64")]
  use crate::platform::caps::x86;

  #[test]
  fn poly1305_matches_rfc_8439_section_2_5_2() {
    let key = [
      0x85, 0xd6, 0xbe, 0x78, 0x57, 0x55, 0x6d, 0x33, 0x7f, 0x44, 0x52, 0xfe, 0x42, 0xd5, 0x06, 0xa8, 0x01, 0x03, 0x80,
      0x8a, 0xfb, 0x0d, 0xb2, 0xfd, 0x4a, 0xbf, 0xf6, 0xaf, 0x41, 0x49, 0xf5, 0x1b,
    ];
    let message = b"Cryptographic Forum Research Group";
    let expected = [
      0xa8, 0x06, 0x1d, 0xc1, 0x30, 0x51, 0x36, 0xc6, 0xc2, 0x2b, 0x8b, 0xaf, 0x0c, 0x01, 0x27, 0xa9,
    ];

    assert_eq!(authenticate(message, &key), expected);
  }

  #[test]
  fn aead_poly1305_matches_rfc_8439_section_2_8_2() {
    let aad = [0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7];
    let ciphertext = [
      0xd3, 0x1a, 0x8d, 0x34, 0x64, 0x8e, 0x60, 0xdb, 0x7b, 0x86, 0xaf, 0xbc, 0x53, 0xef, 0x7e, 0xc2, 0xa4, 0xad, 0xed,
      0x51, 0x29, 0x6e, 0x08, 0xfe, 0xa9, 0xe2, 0xb5, 0xa7, 0x36, 0xee, 0x62, 0xd6, 0x3d, 0xbe, 0xa4, 0x5e, 0x8c, 0xa9,
      0x67, 0x12, 0x82, 0xfa, 0xfb, 0x69, 0xda, 0x92, 0x72, 0x8b, 0x1a, 0x71, 0xde, 0x0a, 0x9e, 0x06, 0x0b, 0x29, 0x05,
      0xd6, 0xa5, 0xb6, 0x7e, 0xcd, 0x3b, 0x36, 0x92, 0xdd, 0xbd, 0x7f, 0x2d, 0x77, 0x8b, 0x8c, 0x98, 0x03, 0xae, 0xe3,
      0x28, 0x09, 0x1b, 0x58, 0xfa, 0xb3, 0x24, 0xe4, 0xfa, 0xd6, 0x75, 0x94, 0x55, 0x85, 0x80, 0x8b, 0x48, 0x31, 0xd7,
      0xbc, 0x3f, 0xf4, 0xde, 0xf0, 0x8e, 0x4b, 0x7a, 0x9d, 0xe5, 0x76, 0xd2, 0x65, 0x86, 0xce, 0xc6, 0x4b, 0x61, 0x16,
    ];
    let poly_key = [
      0x7b, 0xac, 0x2b, 0x25, 0x2d, 0xb4, 0x47, 0xaf, 0x09, 0xb6, 0x7a, 0x55, 0xa4, 0xe9, 0x55, 0x84, 0x0a, 0xe1, 0xd6,
      0x73, 0x10, 0x75, 0xd9, 0xeb, 0x2a, 0x93, 0x75, 0x78, 0x3e, 0xd5, 0x53, 0xff,
    ];
    let expected = [
      0x1a, 0xe1, 0x0b, 0x59, 0x4f, 0x09, 0xe2, 0x6a, 0x7e, 0x90, 0x2e, 0xcb, 0xd0, 0x60, 0x06, 0x91,
    ];

    assert_eq!(
      super::authenticate_aead(AeadPrimitive::ChaCha20Poly1305, &aad, &ciphertext, &poly_key).unwrap(),
      expected
    );
  }

  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "riscv64"))]
  fn authenticate_aead_portable(aad: &[u8], ciphertext: &[u8], key: &[u8; 32]) -> [u8; 16] {
    let lengths = AeadByteLengths::try_new(aad.len(), ciphertext.len()).unwrap();
    authenticate_aead_with(aad, ciphertext, key, State::compute_block_portable, lengths).unwrap()
  }

  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "riscv64"))]
  fn exercise_backend(backend: ComputeBlockFn) {
    let key = [0x5au8; 32];
    for aad_len in [0usize, 1, 15, 16, 17, 31, 32, 33, 80] {
      for ciphertext_len in [0usize, 1, 15, 16, 17, 31, 32, 33, 191, 256] {
        let aad = (0..aad_len)
          .map(|index| index.strict_mul(11).strict_add(7) as u8)
          .collect::<Vec<_>>();
        let ciphertext = (0..ciphertext_len)
          .map(|index| index.strict_mul(17).strict_add(3) as u8)
          .collect::<Vec<_>>();
        let portable = authenticate_aead_portable(&aad, &ciphertext, &key);
        let lengths = AeadByteLengths::try_new(aad.len(), ciphertext.len()).unwrap();
        let accelerated = authenticate_aead_with(&aad, &ciphertext, &key, backend, lengths).unwrap();
        assert_eq!(accelerated, portable);
      }
    }
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn avx512_backend_matches_portable_when_available() {
    if !crate::platform::caps().has(x86::AVX512_READY) {
      return;
    }

    exercise_backend(super::x86_avx512::compute_block);
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn avx2_backend_matches_portable_when_available() {
    if !crate::platform::caps().has(x86::AVX2) {
      return;
    }

    exercise_backend(super::x86_avx2::compute_block);
  }

  #[test]
  #[cfg(target_arch = "aarch64")]
  fn neon_backend_matches_portable_when_available() {
    if !crate::platform::caps().has(aarch64::NEON) {
      return;
    }

    exercise_backend(super::aarch64_neon::compute_block);
  }

  #[test]
  #[cfg(target_arch = "riscv64")]
  fn rvv_backend_matches_portable_when_available() {
    if !crate::platform::caps().has(riscv::V) {
      return;
    }

    exercise_backend(super::riscv64_vector::compute_block);
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn avx2_par4_matches_portable() {
    if !crate::platform::caps().has(x86::AVX2) {
      return;
    }

    let key = [0x5au8; 32];
    for aad_len in [0usize, 1, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 80, 128] {
      for ct_len in [0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 191, 256, 1024, 4096] {
        let aad: Vec<u8> = (0..aad_len).map(|i| i.strict_mul(11).strict_add(7) as u8).collect();
        let ct: Vec<u8> = (0..ct_len).map(|i| i.strict_mul(17).strict_add(3) as u8).collect();
        let portable = authenticate_aead_portable(&aad, &ct, &key);
        let parallel = super::avx2_par4::authenticate_aead_par4(&aad, &ct, &key);
        assert_eq!(parallel, portable, "mismatch at aad={aad_len} ct={ct_len}");
      }
    }
  }

  /// Verify the RFC 8439 AEAD test vector goes through the parallel path.
  #[test]
  #[cfg(target_arch = "x86_64")]
  fn avx2_par4_rfc_8439_aead_vector() {
    if !crate::platform::caps().has(x86::AVX2) {
      return;
    }

    let aad = [0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7];
    let ciphertext = [
      0xd3, 0x1a, 0x8d, 0x34, 0x64, 0x8e, 0x60, 0xdb, 0x7b, 0x86, 0xaf, 0xbc, 0x53, 0xef, 0x7e, 0xc2, 0xa4, 0xad, 0xed,
      0x51, 0x29, 0x6e, 0x08, 0xfe, 0xa9, 0xe2, 0xb5, 0xa7, 0x36, 0xee, 0x62, 0xd6, 0x3d, 0xbe, 0xa4, 0x5e, 0x8c, 0xa9,
      0x67, 0x12, 0x82, 0xfa, 0xfb, 0x69, 0xda, 0x92, 0x72, 0x8b, 0x1a, 0x71, 0xde, 0x0a, 0x9e, 0x06, 0x0b, 0x29, 0x05,
      0xd6, 0xa5, 0xb6, 0x7e, 0xcd, 0x3b, 0x36, 0x92, 0xdd, 0xbd, 0x7f, 0x2d, 0x77, 0x8b, 0x8c, 0x98, 0x03, 0xae, 0xe3,
      0x28, 0x09, 0x1b, 0x58, 0xfa, 0xb3, 0x24, 0xe4, 0xfa, 0xd6, 0x75, 0x94, 0x55, 0x85, 0x80, 0x8b, 0x48, 0x31, 0xd7,
      0xbc, 0x3f, 0xf4, 0xde, 0xf0, 0x8e, 0x4b, 0x7a, 0x9d, 0xe5, 0x76, 0xd2, 0x65, 0x86, 0xce, 0xc6, 0x4b, 0x61, 0x16,
    ];
    let poly_key = [
      0x7b, 0xac, 0x2b, 0x25, 0x2d, 0xb4, 0x47, 0xaf, 0x09, 0xb6, 0x7a, 0x55, 0xa4, 0xe9, 0x55, 0x84, 0x0a, 0xe1, 0xd6,
      0x73, 0x10, 0x75, 0xd9, 0xeb, 0x2a, 0x93, 0x75, 0x78, 0x3e, 0xd5, 0x53, 0xff,
    ];
    let expected = [
      0x1a, 0xe1, 0x0b, 0x59, 0x4f, 0x09, 0xe2, 0x6a, 0x7e, 0x90, 0x2e, 0xcb, 0xd0, 0x60, 0x06, 0x91,
    ];

    let result = super::avx2_par4::authenticate_aead_par4(&aad, &ciphertext, &poly_key);
    assert_eq!(result, expected);
  }
}
