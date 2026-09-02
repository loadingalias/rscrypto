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
#[cfg(target_arch = "riscv64")]
const RISCV64_PAR4_MIN: u64 = 4096;

type ComputeBlockFn = fn(&mut State, &[u8; 16], bool);

#[cfg(feature = "std")]
#[cfg(feature = "xchacha20poly1305")]
static XCHACHA20POLY1305_COMPUTE_BLOCK_DISPATCH: OnceCache<ComputeBlockFn> = OnceCache::new();
#[cfg(feature = "std")]
#[cfg(feature = "chacha20poly1305")]
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
  target_arch = "wasm32",
))]
#[inline(always)]
fn low_u32(value: u64) -> u32 {
  let [b0, b1, b2, b3, _, _, _, _] = value.to_le_bytes();
  u32::from_le_bytes([b0, b1, b2, b3])
}

#[cfg(any(
  all(target_arch = "powerpc64", target_endian = "little"),
  target_arch = "riscv64",
  target_arch = "s390x",
  target_arch = "wasm32",
))]
#[inline(always)]
fn add_limb_product(accumulator: u64, left: u32, right: u32) -> u64 {
  accumulator.wrapping_add(u64::from(left).wrapping_mul(u64::from(right)))
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

  let s1 = r1.wrapping_mul(5);
  let s2 = r2.wrapping_mul(5);
  let s3 = r3.wrapping_mul(5);
  let s4 = r4.wrapping_mul(5);

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

  let d0 = add_limb_product(sum4_mul([h0, h1, h2, h3], [r0, s4, s3, s2]), h4, s1);
  let mut d1 = add_limb_product(sum4_mul([h0, h1, h2, h3], [r1, r0, s4, s3]), h4, s2);
  let mut d2 = add_limb_product(sum4_mul([h0, h1, h2, h3], [r2, r1, r0, s4]), h4, s3);
  let mut d3 = add_limb_product(sum4_mul([h0, h1, h2, h3], [r3, r2, r1, r0]), h4, s4);
  let mut d4 = add_limb_product(sum4_mul([h0, h1, h2, h3], [r4, r3, r2, r1]), h4, r0);

  let mut c = low_u32(d0 >> 26);
  h0 = low_u32(d0) & LIMB_MASK;
  d1 = d1.wrapping_add(u64::from(c));

  c = low_u32(d1 >> 26);
  h1 = low_u32(d1) & LIMB_MASK;
  d2 = d2.wrapping_add(u64::from(c));

  c = low_u32(d2 >> 26);
  h2 = low_u32(d2) & LIMB_MASK;
  d3 = d3.wrapping_add(u64::from(c));

  c = low_u32(d3 >> 26);
  h3 = low_u32(d3) & LIMB_MASK;
  d4 = d4.wrapping_add(u64::from(c));

  c = low_u32(d4 >> 26);
  h4 = low_u32(d4) & LIMB_MASK;
  h0 = h0.wrapping_add(c.wrapping_mul(5));

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
      #[cfg(feature = "xchacha20poly1305")]
      AeadPrimitive::XChaCha20Poly1305 => {
        XCHACHA20POLY1305_COMPUTE_BLOCK_DISPATCH.get_or_init(|| resolve_compute_block(primitive))
      }
      #[cfg(feature = "chacha20poly1305")]
      AeadPrimitive::ChaCha20Poly1305 => {
        CHACHA20POLY1305_COMPUTE_BLOCK_DISPATCH.get_or_init(|| resolve_compute_block(primitive))
      }
      #[cfg(any(test, feature = "aegis256", feature = "aes-gcm", feature = "aes-gcm-siv"))]
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
/// Absorbs one Poly1305 block with the x86-64 AVX2 multiplier.
///
/// # Safety
///
/// The current CPU must support AVX2. `state` must satisfy the internal clamped-key and reduced-accumulator limb
/// bounds established by `State::new` and the Poly1305 block kernels.
unsafe fn compute_block_x86_avx2(state: &mut State, block: &[u8; 16], partial: bool) {
  use core::arch::x86_64::{__m256i, _mm256_mul_epu32, _mm256_setr_epi32, _mm256_storeu_si256};
  use core::ptr::NonNull;

  #[inline(always)]
  fn sum4_mul(lhs: [u32; 4], rhs: [u32; 4]) -> u64 {
    let mut lanes = [0u64; 4];
    let destination = NonNull::from(&mut lanes).cast::<__m256i>().as_ptr();

    // SAFETY: this helper is called only from the enclosing AVX2 kernel. `destination` retains the provenance of
    // the writable 32-byte `lanes` array, and `_mm256_storeu_si256` permits its 8-byte alignment.
    unsafe {
      let a = _mm256_setr_epi32(
        lhs[0].cast_signed(),
        0,
        lhs[1].cast_signed(),
        0,
        lhs[2].cast_signed(),
        0,
        lhs[3].cast_signed(),
        0,
      );
      let b = _mm256_setr_epi32(
        rhs[0].cast_signed(),
        0,
        rhs[1].cast_signed(),
        0,
        rhs[2].cast_signed(),
        0,
        rhs[3].cast_signed(),
        0,
      );
      let products = _mm256_mul_epu32(a, b);
      _mm256_storeu_si256(destination, products);
    }

    let sum = u128::from(lanes[0])
      .strict_add(u128::from(lanes[1]))
      .strict_add(u128::from(lanes[2]))
      .strict_add(u128::from(lanes[3]));
    debug_assert!(sum <= u128::from(u64::MAX));
    let [b0, b1, b2, b3, b4, b5, b6, b7, _, _, _, _, _, _, _, _] = sum.to_le_bytes();
    u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7])
  }

  #[inline(always)]
  fn fivefold_limb(limb: u32) -> u32 {
    const MAX_UNSCALED: u32 = 858_993_459;
    debug_assert!(limb <= MAX_UNSCALED);

    let product = u64::from(limb).strict_mul(5);
    let [b0, b1, b2, b3, _, _, _, _] = product.to_le_bytes();
    u32::from_le_bytes([b0, b1, b2, b3])
  }

  #[inline(always)]
  fn sum5_mul(lhs: [u32; 5], rhs: [u32; 5]) -> u64 {
    let [l0, l1, l2, l3, l4] = lhs;
    let [r0, r1, r2, r3, r4] = rhs;
    let sum =
      u128::from(sum4_mul([l0, l1, l2, l3], [r0, r1, r2, r3])).strict_add(u128::from(l4).strict_mul(u128::from(r4)));
    debug_assert!(sum <= u128::from(u64::MAX));

    let [b0, b1, b2, b3, b4, b5, b6, b7, _, _, _, _, _, _, _, _] = sum.to_le_bytes();
    u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7])
  }

  #[inline(always)]
  fn narrow_limb(value: u64) -> u32 {
    debug_assert_eq!(value >> u32::BITS, 0);
    let [b0, b1, b2, b3, _, _, _, _] = value.to_le_bytes();
    u32::from_le_bytes([b0, b1, b2, b3])
  }

  let hibit = if partial { 0 } else { FULL_BLOCK_HIBIT };

  let r0 = state.r[0];
  let r1 = state.r[1];
  let r2 = state.r[2];
  let r3 = state.r[3];
  let r4 = state.r[4];

  let s1 = fivefold_limb(r1);
  let s2 = fivefold_limb(r2);
  let s3 = fivefold_limb(r3);
  let s4 = fivefold_limb(r4);

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

  let d0 = sum5_mul([h0, h1, h2, h3, h4], [r0, s4, s3, s2, s1]);
  let mut d1 = sum5_mul([h0, h1, h2, h3, h4], [r1, r0, s4, s3, s2]);
  let mut d2 = sum5_mul([h0, h1, h2, h3, h4], [r2, r1, r0, s4, s3]);
  let mut d3 = sum5_mul([h0, h1, h2, h3, h4], [r3, r2, r1, r0, s4]);
  let mut d4 = sum5_mul([h0, h1, h2, h3, h4], [r4, r3, r2, r1, r0]);

  let mut c = d0 >> 26;
  h0 = narrow_limb(d0 & u64::from(LIMB_MASK));
  d1 = d1.strict_add(c);

  c = d1 >> 26;
  h1 = narrow_limb(d1 & u64::from(LIMB_MASK));
  d2 = d2.strict_add(c);

  c = d2 >> 26;
  h2 = narrow_limb(d2 & u64::from(LIMB_MASK));
  d3 = d3.strict_add(c);

  c = d3 >> 26;
  h3 = narrow_limb(d3 & u64::from(LIMB_MASK));
  d4 = d4.strict_add(c);

  c = d4 >> 26;
  h4 = narrow_limb(d4 & u64::from(LIMB_MASK));
  h0 = h0.wrapping_add(fivefold_limb(narrow_limb(c)));

  let c = h0 >> 26;
  h0 &= LIMB_MASK;
  h1 = h1.wrapping_add(c);

  state.h = [h0, h1, h2, h3, h4];
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
/// Absorbs one Poly1305 block with the x86-64 AVX-512 multiplier.
///
/// # Safety
///
/// The current CPU must support AVX-512F, AVX-512VL, AVX-512BW, and AVX-512DQ. `state` must satisfy the internal
/// clamped-key and reduced-accumulator limb bounds established by `State::new` and the Poly1305 block kernels.
unsafe fn compute_block_x86_avx512(state: &mut State, block: &[u8; 16], partial: bool) {
  use core::arch::x86_64::{__m512i, _mm512_mul_epu32, _mm512_setr_epi32, _mm512_storeu_si512};
  use core::ptr::NonNull;

  #[inline(always)]
  fn narrow_sum(value: u128) -> u64 {
    debug_assert!(value <= u128::from(u64::MAX));
    let [b0, b1, b2, b3, b4, b5, b6, b7, _, _, _, _, _, _, _, _] = value.to_le_bytes();
    u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7])
  }

  #[inline(always)]
  fn fivefold_limb(limb: u32) -> u32 {
    const MAX_UNSCALED: u32 = 858_993_459;
    debug_assert!(limb <= MAX_UNSCALED);

    let product = u64::from(limb).strict_mul(5);
    let [b0, b1, b2, b3, _, _, _, _] = product.to_le_bytes();
    u32::from_le_bytes([b0, b1, b2, b3])
  }

  #[inline(always)]
  fn pair_sum4_mul(lhs: [u32; 4], rhs_lo: [u32; 4], rhs_hi: [u32; 4]) -> (u64, u64) {
    let mut lanes = [0u64; 8];
    let destination = NonNull::from(&mut lanes).cast::<__m512i>().as_ptr();

    // SAFETY: this helper is called only from the enclosing AVX-512 kernel. `destination` retains the provenance of
    // the writable 64-byte `lanes` array, and `_mm512_storeu_si512` permits its 8-byte alignment.
    unsafe {
      let a = _mm512_setr_epi32(
        lhs[0].cast_signed(),
        0,
        lhs[1].cast_signed(),
        0,
        lhs[2].cast_signed(),
        0,
        lhs[3].cast_signed(),
        0,
        lhs[0].cast_signed(),
        0,
        lhs[1].cast_signed(),
        0,
        lhs[2].cast_signed(),
        0,
        lhs[3].cast_signed(),
        0,
      );
      let b = _mm512_setr_epi32(
        rhs_lo[0].cast_signed(),
        0,
        rhs_lo[1].cast_signed(),
        0,
        rhs_lo[2].cast_signed(),
        0,
        rhs_lo[3].cast_signed(),
        0,
        rhs_hi[0].cast_signed(),
        0,
        rhs_hi[1].cast_signed(),
        0,
        rhs_hi[2].cast_signed(),
        0,
        rhs_hi[3].cast_signed(),
        0,
      );
      let products = _mm512_mul_epu32(a, b);
      _mm512_storeu_si512(destination, products);
    }

    let low = u128::from(lanes[0])
      .strict_add(u128::from(lanes[1]))
      .strict_add(u128::from(lanes[2]))
      .strict_add(u128::from(lanes[3]));
    let high = u128::from(lanes[4])
      .strict_add(u128::from(lanes[5]))
      .strict_add(u128::from(lanes[6]))
      .strict_add(u128::from(lanes[7]));
    (narrow_sum(low), narrow_sum(high))
  }

  #[inline(always)]
  fn single_sum4_mul(lhs: [u32; 4], rhs: [u32; 4]) -> u64 {
    pair_sum4_mul(lhs, rhs, [0; 4]).0
  }

  #[inline(always)]
  fn add_product(base: u64, lhs: u32, rhs: u32) -> u64 {
    let sum = u128::from(base).strict_add(u128::from(lhs).strict_mul(u128::from(rhs)));
    narrow_sum(sum)
  }

  #[inline(always)]
  fn narrow_limb(value: u64) -> u32 {
    debug_assert_eq!(value >> u32::BITS, 0);
    let [b0, b1, b2, b3, _, _, _, _] = value.to_le_bytes();
    u32::from_le_bytes([b0, b1, b2, b3])
  }

  let hibit = if partial { 0 } else { FULL_BLOCK_HIBIT };

  let r0 = state.r[0];
  let r1 = state.r[1];
  let r2 = state.r[2];
  let r3 = state.r[3];
  let r4 = state.r[4];

  let s1 = fivefold_limb(r1);
  let s2 = fivefold_limb(r2);
  let s3 = fivefold_limb(r3);
  let s4 = fivefold_limb(r4);

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

  let d0 = add_product(d0_base, h4, s1);
  let mut d1 = add_product(d1_base, h4, s2);
  let mut d2 = add_product(d2_base, h4, s3);
  let mut d3 = add_product(d3_base, h4, s4);
  let mut d4 = add_product(d4_base, h4, r0);

  let mut c = d0 >> 26;
  h0 = narrow_limb(d0 & u64::from(LIMB_MASK));
  d1 = d1.strict_add(c);

  c = d1 >> 26;
  h1 = narrow_limb(d1 & u64::from(LIMB_MASK));
  d2 = d2.strict_add(c);

  c = d2 >> 26;
  h2 = narrow_limb(d2 & u64::from(LIMB_MASK));
  d3 = d3.strict_add(c);

  c = d3 >> 26;
  h3 = narrow_limb(d3 & u64::from(LIMB_MASK));
  d4 = d4.strict_add(c);

  c = d4 >> 26;
  h4 = narrow_limb(d4 & u64::from(LIMB_MASK));
  h0 = h0.wrapping_add(fivefold_limb(narrow_limb(c)));

  let c = h0 >> 26;
  h0 &= LIMB_MASK;
  h1 = h1.wrapping_add(c);

  state.h = [h0, h1, h2, h3, h4];
}

/// Absorbs one full Poly1305 block with the AArch64 NEON multiplier.
///
/// # Safety
///
/// The current CPU must support AArch64 NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn compute_block_aarch64_neon(state: &mut State, block: &[u8; 16], partial: bool) {
  use core::arch::aarch64::{vaddq_u64, vaddvq_u64, vcreate_u32, vmull_u32};

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
      vaddvq_u64(vaddq_u64(lo, hi))
    }
  }

  #[inline(always)]
  fn fivefold_limb(limb: u32) -> u32 {
    const MAX_UNSCALED: u32 = 858_993_459;
    debug_assert!(limb <= MAX_UNSCALED);

    let product = u64::from(limb).strict_mul(5);
    let [b0, b1, b2, b3, _, _, _, _] = product.to_le_bytes();
    u32::from_le_bytes([b0, b1, b2, b3])
  }

  #[inline(always)]
  fn sum5_mul(lhs: [u32; 5], rhs: [u32; 5]) -> u64 {
    let [l0, l1, l2, l3, l4] = lhs;
    let [r0, r1, r2, r3, r4] = rhs;
    let sum =
      u128::from(sum4_mul([l0, l1, l2, l3], [r0, r1, r2, r3])).strict_add(u128::from(l4).strict_mul(u128::from(r4)));
    debug_assert!(sum <= u128::from(u64::MAX));

    let [b0, b1, b2, b3, b4, b5, b6, b7, _, _, _, _, _, _, _, _] = sum.to_le_bytes();
    u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7])
  }

  #[inline(always)]
  fn narrow_limb(value: u64) -> u32 {
    debug_assert_eq!(value >> u32::BITS, 0);
    let [b0, b1, b2, b3, _, _, _, _] = value.to_le_bytes();
    u32::from_le_bytes([b0, b1, b2, b3])
  }

  let hibit = if partial { 0 } else { FULL_BLOCK_HIBIT };

  let r0 = state.r[0];
  let r1 = state.r[1];
  let r2 = state.r[2];
  let r3 = state.r[3];
  let r4 = state.r[4];

  let s1 = fivefold_limb(r1);
  let s2 = fivefold_limb(r2);
  let s3 = fivefold_limb(r3);
  let s4 = fivefold_limb(r4);

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

  let d0 = sum5_mul([h0, h1, h2, h3, h4], [r0, s4, s3, s2, s1]);
  let mut d1 = sum5_mul([h0, h1, h2, h3, h4], [r1, r0, s4, s3, s2]);
  let mut d2 = sum5_mul([h0, h1, h2, h3, h4], [r2, r1, r0, s4, s3]);
  let mut d3 = sum5_mul([h0, h1, h2, h3, h4], [r3, r2, r1, r0, s4]);
  let mut d4 = sum5_mul([h0, h1, h2, h3, h4], [r4, r3, r2, r1, r0]);

  let mut c = d0 >> 26;
  h0 = narrow_limb(d0 & u64::from(LIMB_MASK));
  d1 = d1.strict_add(c);

  c = d1 >> 26;
  h1 = narrow_limb(d1 & u64::from(LIMB_MASK));
  d2 = d2.strict_add(c);

  c = d2 >> 26;
  h2 = narrow_limb(d2 & u64::from(LIMB_MASK));
  d3 = d3.strict_add(c);

  c = d3 >> 26;
  h3 = narrow_limb(d3 & u64::from(LIMB_MASK));
  d4 = d4.strict_add(c);

  c = d4 >> 26;
  h4 = narrow_limb(d4 & u64::from(LIMB_MASK));
  h0 = h0.wrapping_add(fivefold_limb(narrow_limb(c)));

  let c = h0 >> 26;
  h0 &= LIMB_MASK;
  h1 = h1.wrapping_add(c);

  state.h = [h0, h1, h2, h3, h4];
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
/// Absorbs one Poly1305 block with the WASM SIMD128 multiplier.
///
/// # Safety
///
/// The current WebAssembly instance must support SIMD128. `state` must satisfy the internal clamped-key and
/// reduced-accumulator limb bounds established by `State::new` and the Poly1305 block kernels.
unsafe fn compute_block_wasm_simd128(state: &mut State, block: &[u8; 16], partial: bool) {
  use core::arch::wasm32::{i64x2_add, u32x4, u64x2_extmul_high_u32x4, u64x2_extmul_low_u32x4, u64x2_extract_lane};

  #[inline(always)]
  fn sum4_mul(lhs: [u32; 4], rhs: [u32; 4]) -> u64 {
    let a = u32x4(lhs[0], lhs[1], lhs[2], lhs[3]);
    let b = u32x4(rhs[0], rhs[1], rhs[2], rhs[3]);
    let lo = u64x2_extmul_low_u32x4(a, b);
    let hi = u64x2_extmul_high_u32x4(a, b);
    let sum = i64x2_add(lo, hi);
    u64x2_extract_lane::<0>(sum).wrapping_add(u64x2_extract_lane::<1>(sum))
  }

  let hibit = if partial { 0 } else { FULL_BLOCK_HIBIT };

  let r0 = state.r[0];
  let r1 = state.r[1];
  let r2 = state.r[2];
  let r3 = state.r[3];
  let r4 = state.r[4];

  let s1 = r1.wrapping_mul(5);
  let s2 = r2.wrapping_mul(5);
  let s3 = r3.wrapping_mul(5);
  let s4 = r4.wrapping_mul(5);

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

  let d0 = add_limb_product(sum4_mul([h0, h1, h2, h3], [r0, s4, s3, s2]), h4, s1);
  let mut d1 = add_limb_product(sum4_mul([h0, h1, h2, h3], [r1, r0, s4, s3]), h4, s2);
  let mut d2 = add_limb_product(sum4_mul([h0, h1, h2, h3], [r2, r1, r0, s4]), h4, s3);
  let mut d3 = add_limb_product(sum4_mul([h0, h1, h2, h3], [r3, r2, r1, r0]), h4, s4);
  let mut d4 = add_limb_product(sum4_mul([h0, h1, h2, h3], [r4, r3, r2, r1]), h4, r0);

  let mut c = low_u32(d0 >> 26);
  h0 = low_u32(d0) & LIMB_MASK;
  d1 = d1.wrapping_add(u64::from(c));

  c = low_u32(d1 >> 26);
  h1 = low_u32(d1) & LIMB_MASK;
  d2 = d2.wrapping_add(u64::from(c));

  c = low_u32(d2 >> 26);
  h2 = low_u32(d2) & LIMB_MASK;
  d3 = d3.wrapping_add(u64::from(c));

  c = low_u32(d3 >> 26);
  h3 = low_u32(d3) & LIMB_MASK;
  d4 = d4.wrapping_add(u64::from(c));

  c = low_u32(d4 >> 26);
  h4 = low_u32(d4) & LIMB_MASK;
  h0 = h0.wrapping_add(c.wrapping_mul(5));

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

impl Drop for State {
  fn drop(&mut self) {
    ct::zeroize_words_no_fence(&mut self.r);
    ct::zeroize_words_no_fence(&mut self.h);
    ct::zeroize_words_no_fence(&mut self.pad);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
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

  #[inline(always)]
  fn compute_block_portable(&mut self, block: &[u8; 16], partial: bool) {
    #[inline(always)]
    fn fivefold_limb(limb: u32) -> u32 {
      const MAX_UNSCALED: u32 = 858_993_459;
      debug_assert!(limb <= MAX_UNSCALED);

      let product = u64::from(limb).strict_mul(5);
      let [b0, b1, b2, b3, _, _, _, _] = product.to_le_bytes();
      u32::from_le_bytes([b0, b1, b2, b3])
    }

    #[inline(always)]
    fn scalar_dot5(lhs: [u32; 5], rhs: [u32; 5]) -> u64 {
      let [l0, l1, l2, l3, l4] = lhs;
      let [r0, r1, r2, r3, r4] = rhs;
      let sum = u128::from(l0)
        .strict_mul(u128::from(r0))
        .strict_add(u128::from(l1).strict_mul(u128::from(r1)))
        .strict_add(u128::from(l2).strict_mul(u128::from(r2)))
        .strict_add(u128::from(l3).strict_mul(u128::from(r3)))
        .strict_add(u128::from(l4).strict_mul(u128::from(r4)));
      debug_assert!(sum <= u128::from(u64::MAX));

      let [b0, b1, b2, b3, b4, b5, b6, b7, _, _, _, _, _, _, _, _] = sum.to_le_bytes();
      u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7])
    }

    #[inline(always)]
    fn narrow_limb(value: u64) -> u32 {
      debug_assert_eq!(value >> u32::BITS, 0);
      let [b0, b1, b2, b3, _, _, _, _] = value.to_le_bytes();
      u32::from_le_bytes([b0, b1, b2, b3])
    }

    let hibit = if partial { 0 } else { FULL_BLOCK_HIBIT };

    let r0 = self.r[0];
    let r1 = self.r[1];
    let r2 = self.r[2];
    let r3 = self.r[3];
    let r4 = self.r[4];

    let s1 = fivefold_limb(r1);
    let s2 = fivefold_limb(r2);
    let s3 = fivefold_limb(r3);
    let s4 = fivefold_limb(r4);

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

    let d0 = scalar_dot5([h0, h1, h2, h3, h4], [r0, s4, s3, s2, s1]);
    let mut d1 = scalar_dot5([h0, h1, h2, h3, h4], [r1, r0, s4, s3, s2]);
    let mut d2 = scalar_dot5([h0, h1, h2, h3, h4], [r2, r1, r0, s4, s3]);
    let mut d3 = scalar_dot5([h0, h1, h2, h3, h4], [r3, r2, r1, r0, s4]);
    let mut d4 = scalar_dot5([h0, h1, h2, h3, h4], [r4, r3, r2, r1, r0]);

    let mut c = d0 >> 26;
    h0 = narrow_limb(d0 & u64::from(LIMB_MASK));
    d1 = d1.strict_add(c);

    c = d1 >> 26;
    h1 = narrow_limb(d1 & u64::from(LIMB_MASK));
    d2 = d2.strict_add(c);

    c = d2 >> 26;
    h2 = narrow_limb(d2 & u64::from(LIMB_MASK));
    d3 = d3.strict_add(c);

    c = d3 >> 26;
    h3 = narrow_limb(d3 & u64::from(LIMB_MASK));
    d4 = d4.strict_add(c);

    c = d4 >> 26;
    h4 = narrow_limb(d4 & u64::from(LIMB_MASK));
    h0 = h0.wrapping_add(fivefold_limb(narrow_limb(c)));

    let c = h0 >> 26;
    h0 &= LIMB_MASK;
    h1 = h1.wrapping_add(c);

    self.h = [h0, h1, h2, h3, h4];
  }

  #[cfg(test)]
  fn update_message(&mut self, message: &[u8], compute_block: ComputeBlockFn) {
    let (blocks, remainder) = message.as_chunks::<16>();
    for block in blocks {
      compute_block(self, block, false);
    }

    if remainder.is_empty() {
      return;
    }

    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    block[remainder.len()] = 1;
    compute_block(self, &block, true);
  }

  fn update_padded_segment(&mut self, segment: &[u8], compute_block: ComputeBlockFn) {
    let (blocks, remainder) = segment.as_chunks::<16>();
    for block in blocks {
      compute_block(self, block, false);
    }

    if remainder.is_empty() {
      return;
    }

    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    compute_block(self, &block, false);
  }

  #[inline(always)]
  fn finalize(mut self) -> [u8; 16] {
    self.finalize_in_place()
  }

  #[inline(always)]
  fn finalize_in_place(&mut self) -> [u8; 16] {
    #[inline(always)]
    fn fivefold_carry(carry: u32) -> u32 {
      const MAX_UNSCALED: u32 = 858_993_459;
      debug_assert!(carry <= MAX_UNSCALED);

      let product = u64::from(carry).strict_mul(5);
      let [b0, b1, b2, b3, _, _, _, _] = product.to_le_bytes();
      u32::from_le_bytes([b0, b1, b2, b3])
    }

    #[inline(always)]
    fn low_word(value: u64) -> u32 {
      let [b0, b1, b2, b3, _, _, _, _] = value.to_le_bytes();
      u32::from_le_bytes([b0, b1, b2, b3])
    }

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
    h0 = h0.wrapping_add(fivefold_carry(c));

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

    let mut f = u64::from(h0).strict_add(u64::from(self.pad[0]));
    h0 = low_word(f);
    f = u64::from(h1).strict_add(u64::from(self.pad[1])).strict_add(f >> 32);
    h1 = low_word(f);
    f = u64::from(h2).strict_add(u64::from(self.pad[2])).strict_add(f >> 32);
    h2 = low_word(f);
    f = u64::from(h3).strict_add(u64::from(self.pad[3])).strict_add(f >> 32);
    h3 = low_word(f);

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
      // SAFETY: this branch verifies AVX2 before selecting the x86-64 parallel kernel.
      return Ok(unsafe { avx2_par4::authenticate_aead_par4(aad, ciphertext, key, lengths) });
    }
  }
  #[cfg(target_arch = "aarch64")]
  {
    use crate::platform::caps::aarch64;
    if lengths.total_at_least(64) && current_caps().has(aarch64::NEON) {
      return Ok(aarch64_neon::authenticate_aead_par4(aad, ciphertext, key, lengths));
    }
  }
  #[cfg(target_arch = "riscv64")]
  {
    use crate::platform::caps::riscv;
    if lengths.total_at_least(RISCV64_PAR4_MIN) && current_caps().has(riscv::V) {
      return Ok(riscv64_vector::authenticate_aead_par4(aad, ciphertext, key, lengths));
    }
  }
  authenticate_aead_with(aad, ciphertext, key, compute_block_resolved(primitive), lengths)
}

#[cfg(any(
  test,
  all(
    feature = "chacha20poly1305",
    any(target_arch = "x86_64", all(target_arch = "powerpc64", target_endian = "little"))
  )
))]
fn authenticate_aead_portable_blocks(
  aad: &[u8],
  ciphertext: &[u8],
  key: &[u8; 32],
  lengths: super::AeadByteLengths,
) -> [u8; 16] {
  let mut state = State::new(key);
  state.update_padded_segment(aad, State::compute_block_portable);
  state.update_padded_segment(ciphertext, State::compute_block_portable);

  let mut length_block = lengths.to_le_bytes_block();
  state.compute_block_portable(&length_block, false);

  let tag = state.finalize();
  ct::zeroize(&mut length_block);
  tag
}

#[cfg(any(
  test,
  all(
    feature = "chacha20poly1305",
    any(target_arch = "x86_64", all(target_arch = "powerpc64", target_endian = "little"))
  )
))]
pub(crate) fn authenticate_aead_empty_text_portable(aad: &[u8], key: &[u8; 32]) -> [u8; 16] {
  authenticate_aead_portable_blocks(aad, &[], key, super::AeadByteLengths::from_usize(aad.len(), 0))
}

#[cfg(all(feature = "chacha20poly1305", target_arch = "powerpc64", target_endian = "little"))]
pub(crate) fn authenticate_aead_short_text_portable(aad: &[u8], ciphertext: &[u8], key: &[u8; 32]) -> [u8; 16] {
  authenticate_aead_portable_blocks(
    aad,
    ciphertext,
    key,
    super::AeadByteLengths::from_usize(aad.len(), ciphertext.len()),
  )
}

#[cfg(feature = "diag")]
/// Computes a ChaCha20-Poly1305 authenticator through the selected Poly1305 backend.
///
/// Returns `None` when the associated-data or ciphertext length cannot be encoded by the AEAD construction.
pub fn diag_chacha20poly1305_authenticate_aead(aad: &[u8], ciphertext: &[u8], key: &[u8; 32]) -> Option<[u8; 16]> {
  #[cfg(feature = "chacha20poly1305")]
  let primitive = AeadPrimitive::ChaCha20Poly1305;
  #[cfg(all(not(feature = "chacha20poly1305"), feature = "xchacha20poly1305"))]
  let primitive = AeadPrimitive::XChaCha20Poly1305;
  authenticate_aead(primitive, aad, ciphertext, key).ok()
}

#[cfg(feature = "diag")]
#[unsafe(no_mangle)]
#[inline(never)]
/// Computes a diagnostic Poly1305 tag after one block using the portable backend.
///
/// `partial` suppresses the full-block high bit for a caller-prepared partial-block encoding.
pub fn diag_poly1305_block_portable_digest(key: &[u8; 32], block: &[u8; 16], partial: bool) -> [u8; 16] {
  let mut state = State::new(key);
  state.compute_block_portable(block, partial);
  state.finalize()
}

#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(target_os = "linux", target_os = "macos")
))]
/// Computes a ChaCha20-Poly1305 authenticator with the four-lane AArch64 NEON backend.
///
/// Returns `None` when the associated-data or ciphertext length cannot be encoded by the AEAD construction.
pub fn diag_chacha20poly1305_authenticate_aead_aarch64_neon_par4(
  aad: &[u8],
  ciphertext: &[u8],
  key: &[u8; 32],
) -> Option<[u8; 16]> {
  let lengths = super::AeadByteLengths::try_new(aad.len(), ciphertext.len()).ok()?;
  Some(aarch64_neon::authenticate_aead_par4(aad, ciphertext, key, lengths))
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
pub(crate) mod aarch64_neon;
#[cfg(target_arch = "x86_64")]
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
  use alloc::vec::Vec;

  use super::authenticate;
  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "riscv64"))]
  use super::{ComputeBlockFn, authenticate_aead_with};
  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "riscv64"))]
  use crate::aead::AeadByteLengths;
  use crate::aead::targets::AeadPrimitive;

  fn primitive() -> AeadPrimitive {
    #[cfg(feature = "chacha20poly1305")]
    {
      AeadPrimitive::ChaCha20Poly1305
    }
    #[cfg(all(not(feature = "chacha20poly1305"), feature = "xchacha20poly1305"))]
    {
      AeadPrimitive::XChaCha20Poly1305
    }
  }
  #[cfg(target_arch = "aarch64")]
  use crate::platform::caps::aarch64;
  #[cfg(target_arch = "riscv64")]
  use crate::platform::caps::riscv;
  #[cfg(target_arch = "x86_64")]
  use crate::platform::caps::x86;

  fn patterned_bytes(length: usize, factor: usize, offset: usize) -> Vec<u8> {
    (0..length)
      .map(|index| {
        let [byte, ..] = index.strict_mul(factor).strict_add(offset).to_le_bytes();
        byte
      })
      .collect()
  }

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

    let actual = super::authenticate_aead(primitive(), &aad, &ciphertext, &poly_key);
    assert_eq!(actual, Ok(expected));
  }

  #[test]
  fn empty_text_fast_path_matches_generic_aead_authentication() {
    let key = [0x5au8; 32];

    for aad_len in [0usize, 1, 14, 15, 16, 17, 31, 32, 33, 63] {
      let aad = patterned_bytes(aad_len, 11, 7);
      let expected = super::authenticate_aead(primitive(), &aad, &[], &key);
      let actual = super::authenticate_aead_empty_text_portable(&aad, &key);
      assert_eq!(
        expected,
        Ok(actual),
        "empty-text authentication mismatch at aad_len={aad_len}"
      );
    }
  }

  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "riscv64"))]
  fn authenticate_aead_portable(aad: &[u8], ciphertext: &[u8], key: &[u8; 32]) -> [u8; 16] {
    let lengths = AeadByteLengths::from_usize(aad.len(), ciphertext.len());
    super::authenticate_aead_portable_blocks(aad, ciphertext, key, lengths)
  }

  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "riscv64"))]
  fn exercise_backend(backend: ComputeBlockFn) {
    let key = [0x5au8; 32];
    for aad_len in [0usize, 1, 15, 16, 17, 31, 32, 33, 80] {
      for ciphertext_len in [0usize, 1, 15, 16, 17, 31, 32, 33, 191, 256] {
        let aad = patterned_bytes(aad_len, 11, 7);
        let ciphertext = patterned_bytes(ciphertext_len, 17, 3);
        let portable = authenticate_aead_portable(&aad, &ciphertext, &key);
        let lengths = AeadByteLengths::from_usize(aad.len(), ciphertext.len());
        let accelerated = authenticate_aead_with(&aad, &ciphertext, &key, backend, lengths);
        assert_eq!(accelerated, Ok(portable));
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
        let aad = patterned_bytes(aad_len, 11, 7);
        let ct = patterned_bytes(ct_len, 17, 3);
        let lengths = AeadByteLengths::from_usize(aad.len(), ct.len());
        let portable = authenticate_aead_portable(&aad, &ct, &key);
        // SAFETY: the test returned above unless AVX2 is available.
        let parallel = unsafe { super::avx2_par4::authenticate_aead_par4(&aad, &ct, &key, lengths) };
        assert_eq!(parallel, portable, "mismatch at aad={aad_len} ct={ct_len}");
      }
    }
  }

  #[test]
  #[cfg(target_arch = "aarch64")]
  fn neon_par4_matches_portable() {
    if !crate::platform::caps().has(aarch64::NEON) {
      return;
    }

    let key = [0x5au8; 32];
    for aad_len in [0usize, 1, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 80, 128] {
      for ct_len in [0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 191, 256, 1024, 4096] {
        let aad = patterned_bytes(aad_len, 11, 7);
        let ct = patterned_bytes(ct_len, 17, 3);
        let lengths = AeadByteLengths::from_usize(aad.len(), ct.len());
        let portable = authenticate_aead_portable(&aad, &ct, &key);
        let parallel = super::aarch64_neon::authenticate_aead_par4(&aad, &ct, &key, lengths);
        assert_eq!(parallel, portable, "mismatch at aad={aad_len} ct={ct_len}");
      }
    }
  }

  #[test]
  #[cfg(target_arch = "riscv64")]
  fn rvv_par4_matches_portable() {
    if !crate::platform::caps().has(riscv::V) {
      return;
    }

    let key = [0x5au8; 32];
    for aad_len in [0usize, 1, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 80, 128] {
      for ct_len in [0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 191, 256, 1024, 4096] {
        let aad = patterned_bytes(aad_len, 11, 7);
        let ct = patterned_bytes(ct_len, 17, 3);
        let lengths = AeadByteLengths::from_usize(aad.len(), ct.len());
        let portable = authenticate_aead_portable(&aad, &ct, &key);
        let parallel = super::riscv64_vector::authenticate_aead_par4(&aad, &ct, &key, lengths);
        assert_eq!(parallel, portable, "mismatch at aad={aad_len} ct={ct_len}");
      }
    }
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn avx2_par4_handles_high_carry_reduction() {
    if !crate::platform::caps().has(x86::AVX2) {
      return;
    }

    let key = [0xffu8; 32];
    let aad = [0xffu8; 257];
    let ct = [0xffu8; 4112];
    let lengths = AeadByteLengths::from_usize(aad.len(), ct.len());

    let portable = authenticate_aead_portable(&aad, &ct, &key);
    // SAFETY: the test returned above unless AVX2 is available.
    let parallel = unsafe { super::avx2_par4::authenticate_aead_par4(&aad, &ct, &key, lengths) };
    assert_eq!(parallel, portable);
  }

  #[test]
  #[cfg(target_arch = "aarch64")]
  fn neon_par4_handles_high_carry_reduction() {
    if !crate::platform::caps().has(aarch64::NEON) {
      return;
    }

    let key = [0xffu8; 32];
    let aad = [0xffu8; 257];
    let ct = [0xffu8; 4096];
    let lengths = AeadByteLengths::from_usize(aad.len(), ct.len());

    let portable = authenticate_aead_portable(&aad, &ct, &key);
    let parallel = super::aarch64_neon::authenticate_aead_par4(&aad, &ct, &key, lengths);
    assert_eq!(parallel, portable);
  }

  #[test]
  #[cfg(target_arch = "riscv64")]
  fn rvv_par4_handles_high_carry_reduction() {
    if !crate::platform::caps().has(riscv::V) {
      return;
    }

    let key = [0xffu8; 32];
    let aad = [0xffu8; 257];
    let ct = [0xffu8; 4096];
    let lengths = AeadByteLengths::from_usize(aad.len(), ct.len());

    let portable = authenticate_aead_portable(&aad, &ct, &key);
    let parallel = super::riscv64_vector::authenticate_aead_par4(&aad, &ct, &key, lengths);
    assert_eq!(parallel, portable);
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

    let lengths = AeadByteLengths::from_usize(aad.len(), ciphertext.len());
    // SAFETY: the test returned above unless AVX2 is available.
    let result = unsafe { super::avx2_par4::authenticate_aead_par4(&aad, &ciphertext, &poly_key, lengths) };
    assert_eq!(result, expected);
  }

  /// Verify the RFC 8439 AEAD test vector goes through the aarch64 parallel path.
  #[test]
  #[cfg(target_arch = "aarch64")]
  fn neon_par4_rfc_8439_aead_vector() {
    if !crate::platform::caps().has(aarch64::NEON) {
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

    let lengths = AeadByteLengths::from_usize(aad.len(), ciphertext.len());
    let result = super::aarch64_neon::authenticate_aead_par4(&aad, &ciphertext, &poly_key, lengths);
    assert_eq!(result, expected);
  }
}
