#![allow(clippy::indexing_slicing)]

//! Portable Poly1305 core.

#[cfg(feature = "std")]
use crate::backend::cache::OnceCache;
use crate::{
  aead::targets::{AeadPrimitive, select_backend},
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

#[must_use]
pub(crate) fn authenticate_aead(primitive: AeadPrimitive, aad: &[u8], ciphertext: &[u8], key: &[u8; 32]) -> [u8; 16] {
  #[cfg(target_arch = "x86_64")]
  {
    use crate::platform::caps::x86;
    let total = aad.len().strict_add(ciphertext.len());
    if total >= 64 && current_caps().has(x86::AVX2) {
      return avx2_par4::authenticate_aead_par4(aad, ciphertext, key);
    }
  }
  authenticate_aead_with(aad, ciphertext, key, compute_block_resolved(primitive))
}

#[must_use]
fn authenticate_aead_with(aad: &[u8], ciphertext: &[u8], key: &[u8; 32], compute_block: ComputeBlockFn) -> [u8; 16] {
  let mut state = State::new(key);
  state.update_padded_segment(aad, compute_block);
  state.update_padded_segment(ciphertext, compute_block);

  let aad_len = match u64::try_from(aad.len()) {
    Ok(len) => len,
    Err(_) => panic!("AAD length exceeds u64"),
  };
  let ciphertext_len = match u64::try_from(ciphertext.len()) {
    Ok(len) => len,
    Err(_) => panic!("ciphertext length exceeds u64"),
  };

  let mut lengths = [0u8; 16];
  lengths[0..8].copy_from_slice(&aad_len.to_le_bytes());
  lengths[8..16].copy_from_slice(&ciphertext_len.to_le_bytes());
  compute_block(&mut state, &lengths, false);

  let tag = state.finalize();
  ct::zeroize(&mut lengths);
  tag
}

#[cfg(target_arch = "x86_64")]
mod x86_avx512 {
  use super::{State, compute_block_x86_avx512};

  #[inline]
  pub(super) fn compute_block(state: &mut State, block: &[u8; 16], partial: bool) {
    // SAFETY: Backend selection guarantees the AVX-512 feature set required by this kernel.
    unsafe { compute_block_impl(state, block, partial) }
  }

  #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
  unsafe fn compute_block_impl(state: &mut State, block: &[u8; 16], partial: bool) {
    // SAFETY: this wrapper enables the AVX-512 feature set required by the shared AVX-512 body.
    unsafe { compute_block_x86_avx512(state, block, partial) }
  }
}

#[cfg(target_arch = "x86_64")]
mod x86_avx2 {
  use super::{State, compute_block_x86_avx2};

  #[inline]
  pub(super) fn compute_block(state: &mut State, block: &[u8; 16], partial: bool) {
    // SAFETY: Backend selection guarantees AVX2 is available before this wrapper is chosen.
    unsafe { compute_block_impl(state, block, partial) }
  }

  #[target_feature(enable = "avx2")]
  unsafe fn compute_block_impl(state: &mut State, block: &[u8; 16], partial: bool) {
    // SAFETY: this wrapper enables AVX2 before calling the shared AVX2 body.
    unsafe { compute_block_x86_avx2(state, block, partial) }
  }
}

#[cfg(target_arch = "aarch64")]
mod aarch64_neon {
  use super::{State, compute_block_aarch64_neon};

  #[inline]
  pub(super) fn compute_block(state: &mut State, block: &[u8; 16], partial: bool) {
    // SAFETY: Backend selection guarantees NEON is available before this wrapper is chosen.
    unsafe { compute_block_impl(state, block, partial) }
  }

  #[target_feature(enable = "neon")]
  unsafe fn compute_block_impl(state: &mut State, block: &[u8; 16], partial: bool) {
    // SAFETY: this wrapper enables NEON before calling the shared NEON body.
    unsafe { compute_block_aarch64_neon(state, block, partial) }
  }
}

#[cfg(target_arch = "wasm32")]
mod wasm_simd128 {
  use super::{State, compute_block_wasm_simd128};

  #[inline]
  pub(super) fn compute_block(state: &mut State, block: &[u8; 16], partial: bool) {
    // SAFETY: backend selection guarantees `simd128` is available before this wrapper is chosen.
    unsafe { compute_block_wasm_simd128(state, block, partial) }
  }
}

#[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
mod power_vsx {
  use core::simd::i64x2;

  use super::{FULL_BLOCK_HIBIT, LIMB_MASK, State, load_u32_le};

  #[inline]
  pub(super) fn compute_block(state: &mut State, block: &[u8; 16], partial: bool) {
    // SAFETY: Backend selection guarantees POWER vector support before this wrapper is chosen.
    unsafe { compute_block_impl(state, block, partial) }
  }

  #[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
  unsafe fn compute_block_impl(state: &mut State, block: &[u8; 16], partial: bool) {
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

    // SAFETY: target_feature ensures VSX availability for all sum4_mul calls below.
    let d0 = unsafe { sum4_mul([h0, h1, h2, h3], [r0, s4, s3, s2]) } + (u64::from(h4) * u64::from(s1));
    // SAFETY: target_feature ensures VSX availability.
    let mut d1 = unsafe { sum4_mul([h0, h1, h2, h3], [r1, r0, s4, s3]) } + (u64::from(h4) * u64::from(s2));
    // SAFETY: target_feature ensures VSX availability.
    let mut d2 = unsafe { sum4_mul([h0, h1, h2, h3], [r2, r1, r0, s4]) } + (u64::from(h4) * u64::from(s3));
    // SAFETY: target_feature ensures VSX availability.
    let mut d3 = unsafe { sum4_mul([h0, h1, h2, h3], [r3, r2, r1, r0]) } + (u64::from(h4) * u64::from(s4));
    // SAFETY: target_feature ensures VSX availability.
    let mut d4 = unsafe { sum4_mul([h0, h1, h2, h3], [r4, r3, r2, r1]) } + (u64::from(h4) * u64::from(r0));

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

  /// Vectorized 4-element dot product: `vmulouw` + `vaddudm`.
  ///
  /// Computes `lhs[0]*rhs[0] + lhs[1]*rhs[1] + lhs[2]*rhs[2] + lhs[3]*rhs[3]`
  /// using two 128-bit multiply-odd and one 128-bit add.
  #[inline(always)]
  unsafe fn sum4_mul(lhs: [u32; 4], rhs: [u32; 4]) -> u64 {
    // SAFETY: POWER8+ VSX available via enclosing target_feature.
    unsafe {
      let a_lo = i64x2::from_array([i64::from(lhs[0]), i64::from(lhs[1])]);
      let b_lo = i64x2::from_array([i64::from(rhs[0]), i64::from(rhs[1])]);
      let prod_lo = vmulouw(a_lo, b_lo);

      let a_hi = i64x2::from_array([i64::from(lhs[2]), i64::from(lhs[3])]);
      let b_hi = i64x2::from_array([i64::from(rhs[2]), i64::from(rhs[3])]);
      let prod_hi = vmulouw(a_hi, b_hi);

      let sum = vaddudm(prod_lo, prod_hi);
      let lanes = sum.to_array();
      (lanes[0] as u64).wrapping_add(lanes[1] as u64)
    }
  }

  /// Multiply low 32 bits of each u64 lane → u64: `vmulouw`.
  #[inline(always)]
  unsafe fn vmulouw(a: i64x2, b: i64x2) -> i64x2 {
    let out: i64x2;
    // SAFETY: POWER8+ VSX available via enclosing target_feature.
    unsafe {
      core::arch::asm!(
        "vmulouw {out}, {a}, {b}",
        out = lateout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(nomem, nostack, pure)
      );
    }
    out
  }

  /// Add u64 lanes: `vaddudm`.
  #[inline(always)]
  unsafe fn vaddudm(a: i64x2, b: i64x2) -> i64x2 {
    let out: i64x2;
    // SAFETY: POWER8+ VSX available via enclosing target_feature.
    unsafe {
      core::arch::asm!(
        "vaddudm {out}, {a}, {b}",
        out = lateout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(nomem, nostack, pure)
      );
    }
    out
  }
}

#[cfg(target_arch = "s390x")]
mod s390x_vector {
  use core::simd::i64x2;

  use super::{FULL_BLOCK_HIBIT, LIMB_MASK, State, load_u32_le};

  #[inline]
  pub(super) fn compute_block(state: &mut State, block: &[u8; 16], partial: bool) {
    // SAFETY: Backend selection guarantees the z/Vector facility before this wrapper is chosen.
    unsafe { compute_block_impl(state, block, partial) }
  }

  #[target_feature(enable = "vector")]
  unsafe fn compute_block_impl(state: &mut State, block: &[u8; 16], partial: bool) {
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

    // SAFETY: target_feature ensures z/Vector availability for all sum4_mul calls below.
    let d0 = unsafe { sum4_mul([h0, h1, h2, h3], [r0, s4, s3, s2]) } + (u64::from(h4) * u64::from(s1));
    // SAFETY: target_feature ensures z/Vector availability.
    let mut d1 = unsafe { sum4_mul([h0, h1, h2, h3], [r1, r0, s4, s3]) } + (u64::from(h4) * u64::from(s2));
    // SAFETY: target_feature ensures z/Vector availability.
    let mut d2 = unsafe { sum4_mul([h0, h1, h2, h3], [r2, r1, r0, s4]) } + (u64::from(h4) * u64::from(s3));
    // SAFETY: target_feature ensures z/Vector availability.
    let mut d3 = unsafe { sum4_mul([h0, h1, h2, h3], [r3, r2, r1, r0]) } + (u64::from(h4) * u64::from(s4));
    // SAFETY: target_feature ensures z/Vector availability.
    let mut d4 = unsafe { sum4_mul([h0, h1, h2, h3], [r4, r3, r2, r1]) } + (u64::from(h4) * u64::from(r0));

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

  /// Vectorized 4-element dot product: `vmlof` + `vag`.
  ///
  /// Computes `lhs[0]*rhs[0] + lhs[1]*rhs[1] + lhs[2]*rhs[2] + lhs[3]*rhs[3]`
  /// using two 128-bit multiply-odd and one 128-bit add.
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn sum4_mul(lhs: [u32; 4], rhs: [u32; 4]) -> u64 {
    // SAFETY: z/Vector facility available via target_feature.
    unsafe {
      let a_lo = i64x2::from_array([i64::from(lhs[0]), i64::from(lhs[1])]);
      let b_lo = i64x2::from_array([i64::from(rhs[0]), i64::from(rhs[1])]);
      let prod_lo = vmlof(a_lo, b_lo);

      let a_hi = i64x2::from_array([i64::from(lhs[2]), i64::from(lhs[3])]);
      let b_hi = i64x2::from_array([i64::from(rhs[2]), i64::from(rhs[3])]);
      let prod_hi = vmlof(a_hi, b_hi);

      let sum = vag(prod_lo, prod_hi);
      let lanes = sum.to_array();
      (lanes[0] as u64).wrapping_add(lanes[1] as u64)
    }
  }

  /// Multiply odd-indexed u32 lanes → u64: `vmlof`.
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn vmlof(a: i64x2, b: i64x2) -> i64x2 {
    let out: i64x2;
    // SAFETY: z/Vector facility available via target_feature.
    unsafe {
      core::arch::asm!(
        "vmlof {out}, {a}, {b}",
        out = lateout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(nomem, nostack, pure)
      );
    }
    out
  }

  /// Add u64 lanes: `vag`.
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn vag(a: i64x2, b: i64x2) -> i64x2 {
    let out: i64x2;
    // SAFETY: z/Vector facility available via target_feature.
    unsafe {
      core::arch::asm!(
        "vag {out}, {a}, {b}",
        out = lateout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(nomem, nostack, pure)
      );
    }
    out
  }
}

#[cfg(target_arch = "riscv64")]
mod riscv64_vector {
  use super::State;

  #[inline]
  pub(super) fn compute_block(state: &mut State, block: &[u8; 16], partial: bool) {
    state.compute_block_portable(block, partial);
  }
}

/// Goll-Gueron 4-way parallel Poly1305 for AEAD.
///
/// Processes 4 blocks (64 bytes) simultaneously using AVX2. Rewrites the serial
/// Horner evaluation `h = ((h + m₀)·R + m₁)·R + …` as
/// `h·R⁴ + m₀·R⁴ + m₁·R³ + m₂·R² + m₃·R¹`, giving 4 independent multiplies.
///
/// Ported from RustCrypto `poly1305` 0.8.0 (Goll-Gueron algorithm).
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod avx2_par4 {
  use core::arch::x86_64::*;

  use super::State;
  use crate::traits::ct;

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
    #[inline(always)]
    unsafe fn from_limbs(limbs: [u32; 5]) -> Self {
      Aligned130(_mm256_setr_epi32(
        limbs[0] as i32,
        limbs[1] as i32,
        limbs[2] as i32,
        limbs[3] as i32,
        limbs[4] as i32,
        0,
        0,
        0,
      ))
    }

    /// Extract five scalar 26-bit limbs.
    #[inline(always)]
    unsafe fn into_limbs(self) -> [u32; 5] {
      let mut buf = [0u32; 8];
      _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, self.0);
      [buf[0], buf[1], buf[2], buf[3], buf[4]]
    }

    /// Load a full 16-byte block, split to 26-bit limbs, set hibit.
    ///
    /// AEAD-only: unconditionally sets the 2¹²⁸ high bit. Not suitable for raw
    /// Poly1305 where partial blocks omit the hibit.
    #[inline(always)]
    unsafe fn from_block(block: &[u8; 16]) -> Self {
      Self::split_to_26bit(_mm256_or_si256(
        _mm256_and_si256(
          _mm256_castsi128_si256(_mm_loadu_si128(block.as_ptr() as *const _)),
          _mm256_set_epi64x(0, 0, -1, -1),
        ),
        _mm256_set_epi64x(0, 1, 0, 0),
      ))
    }

    /// Split a 130-bit integer (low 5 words) into 26-bit limbs.
    #[inline(always)]
    unsafe fn split_to_26bit(x: __m256i) -> Self {
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

    #[inline(always)]
    unsafe fn add(self, other: Aligned130) -> Aligned130 {
      Aligned130(_mm256_add_epi32(self.0, other.0))
    }
  }

  // ── PrecomputedMultiplier ────────────────────────────────────────────────

  impl PrecomputedMultiplier {
    #[inline(always)]
    unsafe fn from_aligned(r: Aligned130) -> Self {
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

  // ── Single multiply: Aligned130 × PrecomputedMultiplier → Unreduced130 ──

  #[inline(always)]
  unsafe fn mul_single(x: Aligned130, r: PrecomputedMultiplier) -> Unreduced130 {
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

  // ── Unreduced130 carry chain and reduction ───────────────────────────────

  /// Carry: propagate bits >26 from v0 into v1.
  #[inline(always)]
  unsafe fn adc_single(v1: __m256i, v0: __m256i) -> (__m256i, __m256i) {
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

  /// Reduce modulo 2¹³⁰ − 5: fold top limb back into bottom.
  #[inline(always)]
  unsafe fn red_single(v1: __m256i, v0: __m256i) -> (__m256i, __m256i) {
    let t = _mm256_srlv_epi64(v1, _mm256_set_epi64x(64, 64, 64, 26));
    let red_0 = _mm256_add_epi64(_mm256_add_epi64(v0, t), _mm256_slli_epi64(t, 2));
    let red_1 = _mm256_and_si256(v1, _mm256_set_epi64x(0, 0, 0, 0x3ff_ffff));
    (red_1, red_0)
  }

  impl Unreduced130 {
    #[inline(always)]
    unsafe fn reduce(self) -> Aligned130 {
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

  // ── Aligned4x130 ────────────────────────────────────────────────────────

  impl Aligned4x130 {
    #[inline(always)]
    unsafe fn from_blocks(src: &[[u8; 16]; 4]) -> Self {
      // SAFETY: `[[u8; 16]; 4]` is 64 contiguous bytes; two 32-byte loads are valid.
      let ptr = src.as_ptr() as *const __m256i;
      let blocks_01 = _mm256_loadu_si256(ptr);
      let blocks_23 = _mm256_loadu_si256(ptr.add(1));
      Self::from_loaded_blocks(blocks_01, blocks_23)
    }

    /// Interleave 4 blocks into 20 packed 26-bit limbs across 3 vectors.
    #[inline(always)]
    unsafe fn from_loaded_blocks(blocks_01: __m256i, blocks_23: __m256i) -> Self {
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

    #[inline(always)]
    unsafe fn add(self, other: Aligned4x130) -> Aligned4x130 {
      Aligned4x130 {
        v0: _mm256_add_epi32(self.v0, other.v0),
        v1: _mm256_add_epi32(self.v1, other.v1),
        v2: _mm256_add_epi32(self.v2, other.v2),
      }
    }
  }

  // ── 4-way parallel multiply ──────────────────────────────────────────────

  /// Multiply 4 values by the same R: `(x0·R, x1·R, x2·R, x3·R)`.
  #[inline(always)]
  unsafe fn mul_4x130(x: &Aligned4x130, r: PrecomputedMultiplier) -> Unreduced4x130 {
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

  // ── Spaced multiply ─────────────────────────────────────────────────────

  /// Multiply lane i by R^(4−i): `(x0·R⁴, x1·R³, x2·R², x3·R¹)`.
  #[inline(always)]
  unsafe fn mul_spaced(x: Aligned4x130, m: SpacedMultiplier4x130) -> Unreduced4x130 {
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

  // ── Unreduced4x130 ──────────────────────────────────────────────────────

  impl Unreduced4x130 {
    /// Carry-reduce 4 values in parallel back to 26-bit limbs.
    #[inline(always)]
    unsafe fn reduce(self) -> Aligned4x130 {
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

    /// Horizontal sum of 4 lanes into a single `Unreduced130`.
    #[inline(always)]
    unsafe fn sum(self) -> Unreduced130 {
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

  // ── SpacedMultiplier4x130 ───────────────────────────────────────────────

  impl SpacedMultiplier4x130 {
    /// Compute `(multiplier, R⁴)` from `(R¹, R²)`.
    #[inline(always)]
    unsafe fn new(r1: PrecomputedMultiplier, r2: PrecomputedMultiplier) -> (Self, PrecomputedMultiplier) {
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

  // ── Top-level AEAD kernel ───────────────────────────────────────────────

  /// Accumulated 4-way polynomial state.
  #[derive(Clone, Copy)]
  struct Par4State {
    poly: Aligned4x130,
    spaced: SpacedMultiplier4x130,
    r4: PrecomputedMultiplier,
  }

  /// Authenticate `(aad, ciphertext)` using 4-way parallel Poly1305.
  ///
  /// Uses its own AVX2 kernel — ignores the per-block `ComputeBlockFn` dispatch.
  pub(super) fn authenticate_aead_par4(aad: &[u8], ciphertext: &[u8], key: &[u8; 32]) -> [u8; 16] {
    // SAFETY: caller verified AVX2 capability via `current_caps().has(x86::AVX2)`.
    unsafe { authenticate_aead_par4_avx2(aad, ciphertext, key) }
  }

  #[target_feature(enable = "avx2")]
  unsafe fn authenticate_aead_par4_avx2(aad: &[u8], ciphertext: &[u8], key: &[u8; 32]) -> [u8; 16] {
    let state = State::new(key);

    // Precompute R¹, R² as AVX2 multipliers.
    let r = Aligned130::from_limbs(state.r);
    let r1 = PrecomputedMultiplier::from_aligned(r);
    let r2 = PrecomputedMultiplier::from_aligned(mul_single(Aligned130(r1.a), r1).reduce());

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
        num_cached = push_block(block, &mut cached, num_cached, &mut acc, r1, r2);
      }
      let rem = chunks.remainder();
      if !rem.is_empty() {
        let mut block = [0u8; 16];
        block[..rem.len()].copy_from_slice(rem);
        num_cached = push_block(block, &mut cached, num_cached, &mut acc, r1, r2);
      }
    }

    // Process lengths block.
    let aad_len = match u64::try_from(aad.len()) {
      Ok(len) => len,
      Err(_) => panic!("AAD length exceeds u64"),
    };
    let ct_len = match u64::try_from(ciphertext.len()) {
      Ok(len) => len,
      Err(_) => panic!("ciphertext length exceeds u64"),
    };
    let mut lengths = [0u8; 16];
    lengths[0..8].copy_from_slice(&aad_len.to_le_bytes());
    lengths[8..16].copy_from_slice(&ct_len.to_le_bytes());
    num_cached = push_block(lengths, &mut cached, num_cached, &mut acc, r1, r2);

    // Finalize: merge 4 lanes, process remaining blocks.
    let mut p: Option<Aligned130> = acc.map(|s| mul_spaced(s.poly, s.spaced).sum().reduce());

    // 2-block tail.
    if num_cached >= 2 {
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

    // 1-block tail.
    if num_cached == 1 {
      let mut c = Aligned130::from_block(&cached[0]);
      if let Some(pv) = p {
        c = c.add(pv);
      }
      p = Some(mul_single(c, r1).reduce());
    }

    // Convert AVX2 result back to scalar and finalize.
    let mut final_state = state;
    if let Some(pv) = p {
      final_state.h = pv.into_limbs();
    }
    let tag = final_state.finalize();
    ct::zeroize(&mut lengths);
    ct::zeroize(cached.as_flattened_mut());
    tag
  }

  /// Cache one block; flush a 4-block group when full. Returns updated `num_cached`.
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
      accumulate_4_blocks(cached, acc, r1, r2);
      0
    } else {
      n
    }
  }

  /// Process a full 4-block group into the parallel accumulator.
  #[inline(always)]
  unsafe fn accumulate_4_blocks(
    cached: &[[u8; 16]; 4],
    acc: &mut Option<Par4State>,
    r1: PrecomputedMultiplier,
    r2: PrecomputedMultiplier,
  ) {
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

#[cfg(test)]
mod tests {
  use alloc::vec::Vec;

  use super::authenticate;
  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
  use super::{ComputeBlockFn, State, authenticate_aead_with};
  use crate::aead::targets::AeadPrimitive;
  #[cfg(target_arch = "aarch64")]
  use crate::platform::caps::aarch64;
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
      super::authenticate_aead(AeadPrimitive::ChaCha20Poly1305, &aad, &ciphertext, &poly_key),
      expected
    );
  }

  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
  fn authenticate_aead_portable(aad: &[u8], ciphertext: &[u8], key: &[u8; 32]) -> [u8; 16] {
    authenticate_aead_with(aad, ciphertext, key, State::compute_block_portable)
  }

  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
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
        let accelerated = authenticate_aead_with(&aad, &ciphertext, &key, backend);
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
