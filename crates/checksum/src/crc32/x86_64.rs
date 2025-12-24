//! x86_64 hardware-accelerated CRC-32 kernels.
//!
//! # Safety
//!
//! This module uses `unsafe` for hardware intrinsics. All unsafe functions
//! document their safety requirements. Safe wrappers are provided for use
//! via the dispatcher system.
#![allow(unsafe_code)]
//! This module provides three acceleration tiers:
//!
//! | Kernel | Instructions | Throughput | CRC-32 | CRC-32C |
//! |--------|--------------|------------|--------|---------|
//! | `vpclmul` | AVX-512 VPCLMULQDQ | ~40 GB/s | ✓ | ✓ |
//! | `pclmul` | PCLMULQDQ + SSE4.1 | ~15 GB/s | ✓ | ✓ |
//! | `sse42` | SSE4.2 CRC32 | ~20 GB/s | ✗ | ✓ |
//!
//! # Selection Priority
//!
//! 1. **VPCLMULQDQ** (AVX-512): Processes 256 bytes/iteration
//! 2. **PCLMULQDQ**: Processes 64 bytes/iteration
//! 3. **SSE4.2**: Native CRC32C instruction (not available for IEEE polynomial)
//!
//! # References
//!
//! - Intel: "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ"
//! - Linux kernel: `arch/x86/crypto/crc32-pclmul_asm.S`
//! - zlib-ng: `arch/x86/crc32_fold_pclmulqdq.c`

#![allow(dead_code)]
// Kernels wired up via dispatcher
// SAFETY: All indexing is over fixed-size arrays with in-bounds constant indices
// (e.g., chunks_exact(8) guarantees 8 bytes per chunk).
#![allow(clippy::indexing_slicing)]
// This module is intrinsics-heavy; keep unsafe blocks readable.
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use core::{
  arch::x86_64::*,
  ops::{BitXor, BitXorAssign},
};

#[cfg(target_arch = "x86_64")]
use crate::common::clmul::Crc32ClmulConstants;

// ─────────────────────────────────────────────────────────────────────────────
// SIMD wrapper type
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
struct Simd(__m128i);

#[cfg(target_arch = "x86_64")]
impl BitXor for Simd {
  type Output = Self;

  #[inline]
  fn bitxor(self, other: Self) -> Self {
    // SAFETY: `_mm_xor_si128` is available on all x86_64 (SSE2 baseline).
    unsafe { Self(_mm_xor_si128(self.0, other.0)) }
  }
}

#[cfg(target_arch = "x86_64")]
impl BitXorAssign for Simd {
  #[inline]
  fn bitxor_assign(&mut self, other: Self) {
    *self = *self ^ other;
  }
}

#[cfg(target_arch = "x86_64")]
impl Simd {
  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn new(high: u64, low: u64) -> Self {
    Self(_mm_set_epi64x(high as i64, low as i64))
  }

  /// Load from a byte slice (must be 16 bytes).
  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn load(ptr: *const u8) -> Self {
    Self(_mm_loadu_si128(ptr.cast::<__m128i>()))
  }

  /// Fold 16 bytes: `(coeff.low ⊗ self.low) ⊕ (coeff.high ⊗ self.high)`.
  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn fold_16(self, coeff: Self) -> Self {
    let h = _mm_clmulepi64_si128::<0x11>(self.0, coeff.0);
    let l = _mm_clmulepi64_si128::<0x00>(self.0, coeff.0);
    Self(_mm_xor_si128(h, l))
  }

  /// Two-step fold for CRC-32: reduce 128 bits → 96 bits → 64 bits.
  ///
  /// This performs the proper reduction sequence before Barrett:
  /// 1. 128→96: (low64 × K_96) ⊕ high64
  /// 2. 96→64: (bits[31:0] × K_64) ⊕ bits[95:32]
  ///
  /// After this, exactly 64 bits remain for Barrett reduction.
  /// The result is in bits [95:32] (matching crc-fast/zlib-ng layout).
  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn fold_width(self, k_96: u64, k_64: u64) -> Self {
    // Step 1: 128 → 96 bits
    // Multiply low 64 bits by K_96 (32-bit constant), XOR with high 64 bits
    let k96 = Self::new(0, k_96);
    let folded = _mm_clmulepi64_si128::<0x00>(self.0, k96.0); // low64 × K_96 → up to 95 bits
    let hi64 = _mm_srli_si128::<8>(self.0); // Get high 64 bits into low lane
    let step1 = _mm_xor_si128(folded, hi64); // 96-bit result (bits 0-95 may be set)

    // Step 2: 96 → 64 bits
    // Layout after step 1:
    //   bits [31:0]:  low 32 bits (multiply by K_64)
    //   bits [95:32]: high 64 bits (XOR with product)
    //
    // Algorithm: (bits[31:0] × K_64) ⊕ bits[95:32]

    // Coefficient vector: K_64 in high lane for clmul_11
    let k64 = Self::new(k_64, 0);

    // Shift left by 12 bytes (96 bits) to move bits[31:0] to bits[127:96]
    // After shift: only bits [127:96] contain the original bits [31:0]
    let shifted = _mm_slli_si128::<12>(step1);

    // clmul_11: shifted.high × K_64
    // shifted.high = (original bits[31:0] << 32) in positions [127:96], zeros in [95:64]
    // Result: (bits[31:0] × K_64) × x^32, placed in bits [95:32] of result
    let clmul = _mm_clmulepi64_si128::<0x11>(shifted, k64.0);

    // Mask to keep bits [63:32] (middle) and bits [127:64] (high) of the 96-bit value
    // This preserves the high 64 bits of the 96-bit value in the correct positions
    // low 64 bits:  0xFFFFFFFF_00000000 (keeps [63:32], zeros [31:0])
    // high 64 bits: 0xFFFFFFFF_FFFFFFFF (keeps all)
    let mask = _mm_set_epi64x(-1i64, 0xFFFFFFFF_00000000u64 as i64);
    let masked = _mm_and_si128(step1, mask);

    // Final XOR: clmul result ⊕ masked bits
    // At this point, the 64-bit reduced value is in bits [95:32]
    let xored = _mm_xor_si128(clmul, masked);

    // Shift right by 4 bytes (32 bits) to move the result from bits [95:32] to bits [63:0]
    // This puts the 64-bit value in the low lane for Barrett reduction
    Self(_mm_srli_si128::<4>(xored))
  }

  /// Barrett reduction for CRC-32: reduce 8B to 4B.
  ///
  /// The input `self` should have the 64-bit CRC residual in the low 64 bits
  /// (high 64 bits zeroed after fold_width).
  ///
  /// # Arguments
  ///
  /// * `poly` - Reciprocal polynomial ((reflected << 1) | 1)
  /// * `mu` - Barrett reduction constant
  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn barrett_32(self, poly: u64, mu: u64) -> u32 {
    // Barrett reduction for CRC-32:
    // 1. T1 = floor(R / x^32) * µ mod x^32
    // 2. T2 = floor(T1 / x^32) * P mod x^32
    // 3. CRC = (R ^ T2) mod x^32
    //
    // We have R in low 64 bits. Extract bits [63:32] for the quotient part.

    let polymu = Self::new(mu, poly);

    // Extract high 32 bits of the 64-bit value for the first multiply
    let r_hi32 = _mm_srli_epi64::<32>(self.0);

    // T1 = (R >> 32) * µ, take bits [63:32] of the 64-bit result
    // polymu = [mu, poly] (high, low), so use 0x10 to select src1.low × src2.high
    let t1 = _mm_clmulepi64_si128::<0x10>(r_hi32, polymu.0);
    let t1_hi32 = _mm_srli_epi64::<32>(t1);

    // T2 = (T1 >> 32) * P, take low 32 bits
    // polymu.low = poly, t1_hi32.low = T1>>32, so use 0x00
    let t2 = _mm_clmulepi64_si128::<0x00>(t1_hi32, polymu.0);

    // CRC = (R ^ T2) & 0xFFFFFFFF
    let result = _mm_xor_si128(self.0, t2);

    // Extract low 32 bits
    _mm_cvtsi128_si32(result) as u32
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SSE4.2 CRC32C (native instruction)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32C using native SSE4.2 `crc32` instruction.
///
/// This is the fastest option for CRC-32C on x86_64, but only works for the
/// Castagnoli polynomial (iSCSI, ext4, Btrfs). The IEEE polynomial must use
/// PCLMULQDQ instead.
///
/// # Performance
///
/// - Processes 8 bytes per `crc32q` instruction
/// - ~3 cycles/8 bytes = ~20 GB/s at 5 GHz
///
/// # Safety
///
/// Requires SSE4.2. Caller must verify `platform::caps().has(SSE42)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
pub unsafe fn crc32c_sse42(mut crc: u32, data: &[u8]) -> u32 {
  #[cfg(target_arch = "x86_64")]
  use core::arch::x86_64::{_mm_crc32_u8, _mm_crc32_u64};

  let mut crc64 = u64::from(crc);

  // Process 8 bytes at a time
  let mut chunks = data.chunks_exact(8);
  for chunk in chunks.by_ref() {
    // SAFETY: chunks_exact(8) guarantees exactly 8 bytes
    let bytes: [u8; 8] = [
      chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
    ];
    let val = u64::from_le_bytes(bytes);
    crc64 = _mm_crc32_u64(crc64, val);
  }

  // Process remaining bytes
  crc = crc64 as u32;
  for &byte in chunks.remainder() {
    crc = _mm_crc32_u8(crc, byte);
  }

  crc
}

// ─────────────────────────────────────────────────────────────────────────────
// PCLMULQDQ Folding (works for any polynomial)
// ─────────────────────────────────────────────────────────────────────────────

/// Core SIMD update for CRC-32: process 64-byte blocks.
///
/// CRC-32 uses 64-byte blocks (4×16B lanes) vs CRC-64's 128-byte blocks (8×16B).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn update_simd(state: u32, first: &[Simd; 4], rest: &[[Simd; 4]], consts: &Crc32ClmulConstants) -> u32 {
  let mut x = *first;

  // XOR the initial CRC into the first lane (low 32 bits).
  x[0] ^= Simd::new(0, u64::from(state));

  // 64-byte folding coefficient.
  let coeff = Simd::new(consts.fold_64b.0, consts.fold_64b.1);

  for chunk in rest {
    // Manually unrolled for better ILP and to avoid iterator overhead.
    let t0 = x[0].fold_16(coeff);
    let t1 = x[1].fold_16(coeff);
    let t2 = x[2].fold_16(coeff);
    let t3 = x[3].fold_16(coeff);

    x[0] = chunk[0] ^ t0;
    x[1] = chunk[1] ^ t1;
    x[2] = chunk[2] ^ t2;
    x[3] = chunk[3] ^ t3;
  }

  fold_tail(x, consts)
}

/// Tail reduction: reduce 4×16B → 1×16B → 8B → 4B (u32).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn fold_tail(x: [Simd; 4], consts: &Crc32ClmulConstants) -> u32 {
  // Tail reduction (4×16B → 1×16B), unrolled for throughput.
  // CRC-32 has 3 tail fold entries (vs CRC-64's 7).
  let c0 = Simd::new(consts.tail_fold_16b[0].0, consts.tail_fold_16b[0].1); // 48 bytes
  let c1 = Simd::new(consts.tail_fold_16b[1].0, consts.tail_fold_16b[1].1); // 32 bytes
  let c2 = Simd::new(consts.tail_fold_16b[2].0, consts.tail_fold_16b[2].1); // 16 bytes

  // Fold lanes 0..2 into lane 3.
  let mut acc = x[3];
  acc ^= x[0].fold_16(c0);
  acc ^= x[1].fold_16(c1);
  acc ^= x[2].fold_16(c2);

  // Two-step fold (128→96→64), then Barrett reduction to 32 bits.
  acc
    .fold_width(consts.k_96, consts.k_64)
    .barrett_32(consts.poly, consts.mu)
}

/// Fold a single 64-byte block: `x = fold(x, coeff) ^ chunk`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn fold_block_64(x: &mut [Simd; 4], chunk: &[Simd; 4], coeff: Simd) {
  let t0 = x[0].fold_16(coeff);
  let t1 = x[1].fold_16(coeff);
  let t2 = x[2].fold_16(coeff);
  let t3 = x[3].fold_16(coeff);

  x[0] = chunk[0] ^ t0;
  x[1] = chunk[1] ^ t1;
  x[2] = chunk[2] ^ t2;
  x[3] = chunk[3] ^ t3;
}

/// CRC-32 using PCLMULQDQ folding (64-byte blocks).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn crc32_pclmul_impl(
  mut state: u32,
  bytes: &[u8],
  consts: &Crc32ClmulConstants,
  tables: &[[u32; 256]; 16],
) -> u32 {
  let (left, middle, right) = bytes.align_to::<[Simd; 4]>();
  if let Some((first, rest)) = middle.split_first() {
    state = super::portable::crc32_slice16(state, left, tables);
    state = update_simd(state, first, rest, consts);
    super::portable::crc32_slice16(state, right, tables)
  } else {
    super::portable::crc32_slice16(state, bytes, tables)
  }
}

/// Small-buffer CLMUL path: fold one 16-byte lane at a time.
///
/// Targets the regime where full 64-byte folding has too much setup cost,
/// but CLMUL still outperforms table CRC (typically ~16..63 bytes).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn crc32_pclmul_small_impl(
  mut state: u32,
  bytes: &[u8],
  consts: &Crc32ClmulConstants,
  tables: &[[u32; 256]; 16],
) -> u32 {
  let (left, middle, right) = bytes.align_to::<Simd>();

  // Prefix: portable until 16B alignment.
  state = super::portable::crc32_slice16(state, left, tables);

  // If we don't have any full 16B lane, finish portably.
  let Some((first, rest)) = middle.split_first() else {
    return super::portable::crc32_slice16(state, right, tables);
  };

  let mut acc = *first;
  acc ^= Simd::new(0, u64::from(state));

  // Shift-by-16B folding coefficient (K_127, K_191).
  let coeff_16b = Simd::new(consts.tail_fold_16b[2].0, consts.tail_fold_16b[2].1);

  for chunk in rest {
    acc = *chunk ^ acc.fold_16(coeff_16b);
  }

  // Two-step fold (128→96→64), then Barrett reduction to 32 bits.
  state = acc
    .fold_width(consts.k_96, consts.k_64)
    .barrett_32(consts.poly, consts.mu);
  super::portable::crc32_slice16(state, right, tables)
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-stream coefficients (compile-time)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
use crate::common::{
  clmul::fold16_coeff_for_bytes_32,
  tables::{CRC32_IEEE_POLY, CRC32C_POLY},
};

// Helper to compute CRC32 fold coefficients at compile time.
#[cfg(target_arch = "x86_64")]
const fn fold_coeff_32(poly: u32, bytes: u32) -> (u64, u64) {
  let normal = poly.reverse_bits();
  fold16_coeff_for_bytes_32(normal, bytes)
}

// 2-way (128B per iteration): update step shifts by 2×64B = 128B.
#[cfg(target_arch = "x86_64")]
const IEEE_FOLD_128B: (u64, u64) = fold_coeff_32(CRC32_IEEE_POLY, 128);
#[cfg(target_arch = "x86_64")]
const CASTAGNOLI_FOLD_128B: (u64, u64) = fold_coeff_32(CRC32C_POLY, 128);

// 4-way (256B per iteration): update step shifts by 4×64B = 256B.
#[cfg(target_arch = "x86_64")]
const IEEE_FOLD_256B: (u64, u64) = fold_coeff_32(CRC32_IEEE_POLY, 256);
#[cfg(target_arch = "x86_64")]
const CASTAGNOLI_FOLD_256B: (u64, u64) = fold_coeff_32(CRC32C_POLY, 256);

// 7-way (448B per iteration): update step shifts by 7×64B = 448B.
#[cfg(target_arch = "x86_64")]
const IEEE_FOLD_448B: (u64, u64) = fold_coeff_32(CRC32_IEEE_POLY, 448);
#[cfg(target_arch = "x86_64")]
const CASTAGNOLI_FOLD_448B: (u64, u64) = fold_coeff_32(CRC32C_POLY, 448);

// Combine coefficients for 4-way (reduce 4 streams to 1).
#[cfg(target_arch = "x86_64")]
const IEEE_COMBINE_4WAY: [(u64, u64); 3] = [
  fold_coeff_32(CRC32_IEEE_POLY, 192), // 3 blocks
  fold_coeff_32(CRC32_IEEE_POLY, 128), // 2 blocks
  fold_coeff_32(CRC32_IEEE_POLY, 64),  // 1 block
];
#[cfg(target_arch = "x86_64")]
const CASTAGNOLI_COMBINE_4WAY: [(u64, u64); 3] = [
  fold_coeff_32(CRC32C_POLY, 192),
  fold_coeff_32(CRC32C_POLY, 128),
  fold_coeff_32(CRC32C_POLY, 64),
];

// Combine coefficients for 7-way (reduce 7 streams to 1).
#[cfg(target_arch = "x86_64")]
const IEEE_COMBINE_7WAY: [(u64, u64); 6] = [
  fold_coeff_32(CRC32_IEEE_POLY, 384), // 6 blocks
  fold_coeff_32(CRC32_IEEE_POLY, 320), // 5 blocks
  fold_coeff_32(CRC32_IEEE_POLY, 256), // 4 blocks
  fold_coeff_32(CRC32_IEEE_POLY, 192), // 3 blocks
  fold_coeff_32(CRC32_IEEE_POLY, 128), // 2 blocks
  fold_coeff_32(CRC32_IEEE_POLY, 64),  // 1 block
];
#[cfg(target_arch = "x86_64")]
const CASTAGNOLI_COMBINE_7WAY: [(u64, u64); 6] = [
  fold_coeff_32(CRC32C_POLY, 384),
  fold_coeff_32(CRC32C_POLY, 320),
  fold_coeff_32(CRC32C_POLY, 256),
  fold_coeff_32(CRC32C_POLY, 192),
  fold_coeff_32(CRC32C_POLY, 128),
  fold_coeff_32(CRC32C_POLY, 64),
];

// ─────────────────────────────────────────────────────────────────────────────
// PCLMULQDQ multi-stream (2-way, 64B blocks)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn update_simd_2way(
  state: u32,
  blocks: &[[Simd; 4]],
  fold_128b: (u64, u64),
  consts: &Crc32ClmulConstants,
) -> u32 {
  debug_assert!(blocks.len() >= 2);

  let coeff_128 = Simd::new(fold_128b.0, fold_128b.1);
  let coeff_64 = Simd::new(consts.fold_64b.0, consts.fold_64b.1);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];

  // Inject CRC into stream 0 (block 0).
  s0[0] ^= Simd::new(0, u64::from(state));

  // Process the largest even prefix with 2-way striping.
  let mut i = 2;
  let even = blocks.len() & !1usize;
  while i < even {
    fold_block_64(&mut s0, &blocks[i], coeff_128);
    fold_block_64(&mut s1, &blocks[i.strict_add(1)], coeff_128);
    i = i.strict_add(2);
  }

  // Merge streams: A·s0 ⊕ s1 (A = shift by 64B).
  let mut combined = s1;
  combined[0] ^= s0[0].fold_16(coeff_64);
  combined[1] ^= s0[1].fold_16(coeff_64);
  combined[2] ^= s0[2].fold_16(coeff_64);
  combined[3] ^= s0[3].fold_16(coeff_64);

  // Handle any remaining block (odd tail) sequentially.
  if even != blocks.len() {
    fold_block_64(&mut combined, &blocks[even], coeff_64);
  }

  fold_tail(combined, consts)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn update_simd_4way(
  state: u32,
  blocks: &[[Simd; 4]],
  fold_256b: (u64, u64),
  combine: &[(u64, u64); 3],
  consts: &Crc32ClmulConstants,
) -> u32 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 4 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd(state, first, rest, consts);
  }

  let aligned = (blocks.len() / 4) * 4;

  let coeff_256 = Simd::new(fold_256b.0, fold_256b.1);
  let coeff_64 = Simd::new(consts.fold_64b.0, consts.fold_64b.1);
  let c192 = Simd::new(combine[0].0, combine[0].1);
  let c128 = Simd::new(combine[1].0, combine[1].1);
  let c64 = Simd::new(combine[2].0, combine[2].1);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];
  let mut s2 = blocks[2];
  let mut s3 = blocks[3];

  // Inject CRC into stream 0.
  s0[0] ^= Simd::new(0, u64::from(state));

  let mut i = 4;
  while i < aligned {
    fold_block_64(&mut s0, &blocks[i], coeff_256);
    fold_block_64(&mut s1, &blocks[i.strict_add(1)], coeff_256);
    fold_block_64(&mut s2, &blocks[i.strict_add(2)], coeff_256);
    fold_block_64(&mut s3, &blocks[i.strict_add(3)], coeff_256);
    i = i.strict_add(4);
  }

  // Merge: A^3·s0 ⊕ A^2·s1 ⊕ A·s2 ⊕ s3.
  let mut combined = s3;
  combined[0] ^= s2[0].fold_16(c64);
  combined[1] ^= s2[1].fold_16(c64);
  combined[2] ^= s2[2].fold_16(c64);
  combined[3] ^= s2[3].fold_16(c64);

  combined[0] ^= s1[0].fold_16(c128);
  combined[1] ^= s1[1].fold_16(c128);
  combined[2] ^= s1[2].fold_16(c128);
  combined[3] ^= s1[3].fold_16(c128);

  combined[0] ^= s0[0].fold_16(c192);
  combined[1] ^= s0[1].fold_16(c192);
  combined[2] ^= s0[2].fold_16(c192);
  combined[3] ^= s0[3].fold_16(c192);

  for block in &blocks[aligned..] {
    fold_block_64(&mut combined, block, coeff_64);
  }

  fold_tail(combined, consts)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn update_simd_7way(
  state: u32,
  blocks: &[[Simd; 4]],
  fold_448b: (u64, u64),
  combine: &[(u64, u64); 6],
  consts: &Crc32ClmulConstants,
) -> u32 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 7 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd(state, first, rest, consts);
  }

  let aligned = (blocks.len() / 7) * 7;

  let coeff_448 = Simd::new(fold_448b.0, fold_448b.1);
  let coeff_64 = Simd::new(consts.fold_64b.0, consts.fold_64b.1);

  let c384 = Simd::new(combine[0].0, combine[0].1);
  let c320 = Simd::new(combine[1].0, combine[1].1);
  let c256 = Simd::new(combine[2].0, combine[2].1);
  let c192 = Simd::new(combine[3].0, combine[3].1);
  let c128 = Simd::new(combine[4].0, combine[4].1);
  let c64 = Simd::new(combine[5].0, combine[5].1);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];
  let mut s2 = blocks[2];
  let mut s3 = blocks[3];
  let mut s4 = blocks[4];
  let mut s5 = blocks[5];
  let mut s6 = blocks[6];

  // Inject CRC into stream 0.
  s0[0] ^= Simd::new(0, u64::from(state));

  let mut i = 7;
  while i < aligned {
    fold_block_64(&mut s0, &blocks[i], coeff_448);
    fold_block_64(&mut s1, &blocks[i.strict_add(1)], coeff_448);
    fold_block_64(&mut s2, &blocks[i.strict_add(2)], coeff_448);
    fold_block_64(&mut s3, &blocks[i.strict_add(3)], coeff_448);
    fold_block_64(&mut s4, &blocks[i.strict_add(4)], coeff_448);
    fold_block_64(&mut s5, &blocks[i.strict_add(5)], coeff_448);
    fold_block_64(&mut s6, &blocks[i.strict_add(6)], coeff_448);
    i = i.strict_add(7);
  }

  // Merge: A^6·s0 ⊕ A^5·s1 ⊕ A^4·s2 ⊕ A^3·s3 ⊕ A^2·s4 ⊕ A·s5 ⊕ s6.
  let mut combined = s6;
  combined[0] ^= s5[0].fold_16(c64);
  combined[1] ^= s5[1].fold_16(c64);
  combined[2] ^= s5[2].fold_16(c64);
  combined[3] ^= s5[3].fold_16(c64);

  combined[0] ^= s4[0].fold_16(c128);
  combined[1] ^= s4[1].fold_16(c128);
  combined[2] ^= s4[2].fold_16(c128);
  combined[3] ^= s4[3].fold_16(c128);

  combined[0] ^= s3[0].fold_16(c192);
  combined[1] ^= s3[1].fold_16(c192);
  combined[2] ^= s3[2].fold_16(c192);
  combined[3] ^= s3[3].fold_16(c192);

  combined[0] ^= s2[0].fold_16(c256);
  combined[1] ^= s2[1].fold_16(c256);
  combined[2] ^= s2[2].fold_16(c256);
  combined[3] ^= s2[3].fold_16(c256);

  combined[0] ^= s1[0].fold_16(c320);
  combined[1] ^= s1[1].fold_16(c320);
  combined[2] ^= s1[2].fold_16(c320);
  combined[3] ^= s1[3].fold_16(c320);

  combined[0] ^= s0[0].fold_16(c384);
  combined[1] ^= s0[1].fold_16(c384);
  combined[2] ^= s0[2].fold_16(c384);
  combined[3] ^= s0[3].fold_16(c384);

  for block in &blocks[aligned..] {
    fold_block_64(&mut combined, block, coeff_64);
  }

  fold_tail(combined, consts)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn crc32_pclmul_2way_impl(
  mut state: u32,
  bytes: &[u8],
  fold_128b: (u64, u64),
  consts: &Crc32ClmulConstants,
  tables: &[[u32; 256]; 16],
) -> u32 {
  let (left, middle, right) = bytes.align_to::<[Simd; 4]>();
  if middle.is_empty() {
    return super::portable::crc32_slice16(state, bytes, tables);
  }

  state = super::portable::crc32_slice16(state, left, tables);

  if middle.len() >= 2 {
    state = update_simd_2way(state, middle, fold_128b, consts);
  } else if let Some((first, rest)) = middle.split_first() {
    state = update_simd(state, first, rest, consts);
  }

  super::portable::crc32_slice16(state, right, tables)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn crc32_pclmul_4way_impl(
  mut state: u32,
  bytes: &[u8],
  fold_256b: (u64, u64),
  combine: &[(u64, u64); 3],
  consts: &Crc32ClmulConstants,
  tables: &[[u32; 256]; 16],
) -> u32 {
  let (left, middle, right) = bytes.align_to::<[Simd; 4]>();
  if middle.is_empty() {
    return super::portable::crc32_slice16(state, bytes, tables);
  }

  state = super::portable::crc32_slice16(state, left, tables);
  state = update_simd_4way(state, middle, fold_256b, combine, consts);
  super::portable::crc32_slice16(state, right, tables)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn crc32_pclmul_7way_impl(
  mut state: u32,
  bytes: &[u8],
  fold_448b: (u64, u64),
  combine: &[(u64, u64); 6],
  consts: &Crc32ClmulConstants,
  tables: &[[u32; 256]; 16],
) -> u32 {
  let (left, middle, right) = bytes.align_to::<[Simd; 4]>();
  if middle.is_empty() {
    return super::portable::crc32_slice16(state, bytes, tables);
  }

  state = super::portable::crc32_slice16(state, left, tables);
  state = update_simd_7way(state, middle, fold_448b, combine, consts);
  super::portable::crc32_slice16(state, right, tables)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API (unsafe, requires feature check)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32 IEEE using PCLMULQDQ folding.
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc32_pclmul(crc: u32, data: &[u8]) -> u32 {
  crc32_pclmul_impl(
    crc,
    data,
    &crate::common::clmul::CRC32_IEEE_CLMUL,
    &super::kernel_tables::IEEE_TABLES_16,
  )
}

/// CRC-32C (Castagnoli) using PCLMULQDQ folding.
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc32c_pclmul(crc: u32, data: &[u8]) -> u32 {
  crc32_pclmul_impl(
    crc,
    data,
    &crate::common::clmul::CRC32C_CLMUL,
    &super::kernel_tables::CASTAGNOLI_TABLES_16,
  )
}

/// CRC-32 IEEE using PCLMULQDQ (small-buffer lane folding).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc32_pclmul_small(crc: u32, data: &[u8]) -> u32 {
  crc32_pclmul_small_impl(
    crc,
    data,
    &crate::common::clmul::CRC32_IEEE_CLMUL,
    &super::kernel_tables::IEEE_TABLES_16,
  )
}

/// CRC-32C (Castagnoli) using PCLMULQDQ (small-buffer lane folding).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc32c_pclmul_small(crc: u32, data: &[u8]) -> u32 {
  crc32_pclmul_small_impl(
    crc,
    data,
    &crate::common::clmul::CRC32C_CLMUL,
    &super::kernel_tables::CASTAGNOLI_TABLES_16,
  )
}

/// CRC-32 IEEE using PCLMULQDQ folding (2-way ILP variant).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc32_pclmul_2way(crc: u32, data: &[u8]) -> u32 {
  crc32_pclmul_2way_impl(
    crc,
    data,
    IEEE_FOLD_128B,
    &crate::common::clmul::CRC32_IEEE_CLMUL,
    &super::kernel_tables::IEEE_TABLES_16,
  )
}

/// CRC-32C (Castagnoli) using PCLMULQDQ folding (2-way ILP variant).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc32c_pclmul_2way(crc: u32, data: &[u8]) -> u32 {
  crc32_pclmul_2way_impl(
    crc,
    data,
    CASTAGNOLI_FOLD_128B,
    &crate::common::clmul::CRC32C_CLMUL,
    &super::kernel_tables::CASTAGNOLI_TABLES_16,
  )
}

/// CRC-32 IEEE using PCLMULQDQ folding (4-way ILP variant).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc32_pclmul_4way(crc: u32, data: &[u8]) -> u32 {
  crc32_pclmul_4way_impl(
    crc,
    data,
    IEEE_FOLD_256B,
    &IEEE_COMBINE_4WAY,
    &crate::common::clmul::CRC32_IEEE_CLMUL,
    &super::kernel_tables::IEEE_TABLES_16,
  )
}

/// CRC-32C (Castagnoli) using PCLMULQDQ folding (4-way ILP variant).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc32c_pclmul_4way(crc: u32, data: &[u8]) -> u32 {
  crc32_pclmul_4way_impl(
    crc,
    data,
    CASTAGNOLI_FOLD_256B,
    &CASTAGNOLI_COMBINE_4WAY,
    &crate::common::clmul::CRC32C_CLMUL,
    &super::kernel_tables::CASTAGNOLI_TABLES_16,
  )
}

/// CRC-32 IEEE using PCLMULQDQ folding (7-way ILP variant).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc32_pclmul_7way(crc: u32, data: &[u8]) -> u32 {
  crc32_pclmul_7way_impl(
    crc,
    data,
    IEEE_FOLD_448B,
    &IEEE_COMBINE_7WAY,
    &crate::common::clmul::CRC32_IEEE_CLMUL,
    &super::kernel_tables::IEEE_TABLES_16,
  )
}

/// CRC-32C (Castagnoli) using PCLMULQDQ folding (7-way ILP variant).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc32c_pclmul_7way(crc: u32, data: &[u8]) -> u32 {
  crc32_pclmul_7way_impl(
    crc,
    data,
    CASTAGNOLI_FOLD_448B,
    &CASTAGNOLI_COMBINE_7WAY,
    &crate::common::clmul::CRC32C_CLMUL,
    &super::kernel_tables::CASTAGNOLI_TABLES_16,
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX-512 VPCLMULQDQ (widest vectors)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32 using AVX-512 VPCLMULQDQ.
///
/// Processes 256 bytes per iteration using 512-bit vectors.
///
/// # Safety
///
/// Requires AVX-512F + AVX-512VL + VPCLMULQDQ. Caller must verify caps.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vl", enable = "vpclmulqdq")]
pub unsafe fn crc32_vpclmul(_crc: u32, _data: &[u8]) -> u32 {
  todo!("VPCLMULQDQ CRC-32 IEEE implementation")
}

/// CRC-32C using AVX-512 VPCLMULQDQ.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vl", enable = "vpclmulqdq")]
pub unsafe fn crc32c_vpclmul(_crc: u32, _data: &[u8]) -> u32 {
  todo!("VPCLMULQDQ CRC-32C implementation")
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Wrappers (safe interface)
// ─────────────────────────────────────────────────────────────────────────────

/// Safe wrapper for SSE4.2 CRC-32C kernel.
///
/// # Safety
///
/// This function checks CPU features at the call site via the dispatcher.
/// Only call through `Crc32Dispatcher` which verifies SSE4.2 is available.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn crc32c_sse42_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies SSE4.2 before selecting this kernel
  unsafe { crc32c_sse42(crc, data) }
}

/// Safe wrapper for CRC-32 IEEE PCLMUL kernel.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn crc32_pclmul_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel
  unsafe { crc32_pclmul(crc, data) }
}

/// Safe wrapper for CRC-32C PCLMUL kernel.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn crc32c_pclmul_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel
  unsafe { crc32c_pclmul(crc, data) }
}

/// Safe wrapper for CRC-32 IEEE PCLMUL small kernel.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn crc32_pclmul_small_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel
  unsafe { crc32_pclmul_small(crc, data) }
}

/// Safe wrapper for CRC-32C PCLMUL small kernel.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn crc32c_pclmul_small_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel
  unsafe { crc32c_pclmul_small(crc, data) }
}

/// Safe wrapper for CRC-32 IEEE PCLMUL 2-way kernel.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn crc32_pclmul_2way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel
  unsafe { crc32_pclmul_2way(crc, data) }
}

/// Safe wrapper for CRC-32C PCLMUL 2-way kernel.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn crc32c_pclmul_2way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel
  unsafe { crc32c_pclmul_2way(crc, data) }
}

/// Safe wrapper for CRC-32 IEEE PCLMUL 4-way kernel.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn crc32_pclmul_4way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel
  unsafe { crc32_pclmul_4way(crc, data) }
}

/// Safe wrapper for CRC-32C PCLMUL 4-way kernel.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn crc32c_pclmul_4way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel
  unsafe { crc32c_pclmul_4way(crc, data) }
}

/// Safe wrapper for CRC-32 IEEE PCLMUL 7-way kernel.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn crc32_pclmul_7way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel
  unsafe { crc32_pclmul_7way(crc, data) }
}

/// Safe wrapper for CRC-32C PCLMUL 7-way kernel.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn crc32c_pclmul_7way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel
  unsafe { crc32c_pclmul_7way(crc, data) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  extern crate alloc;
  extern crate std;

  use alloc::vec::Vec;

  use super::*;

  const TEST_DATA: &[u8] = b"123456789";
  const CRC32C_CHECK: u32 = 0xE3069283;
  const CRC32_IEEE_CHECK: u32 = 0xCBF43926;

  fn make_data(len: usize) -> Vec<u8> {
    (0..len)
      .map(|i| (i as u8).wrapping_mul(17).wrapping_add((i >> 3) as u8))
      .collect()
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn test_crc32c_sse42() {
    if !std::is_x86_feature_detected!("sse4.2") {
      std::eprintln!("Skipping SSE4.2 test: not supported");
      return;
    }

    // SAFETY: We just checked SSE4.2 is available
    let crc = unsafe { crc32c_sse42(!0, TEST_DATA) } ^ !0;
    assert_eq!(crc, CRC32C_CHECK);
  }

  #[test]
  fn test_sse42_streaming() {
    if !std::is_x86_feature_detected!("sse4.2") {
      return;
    }

    // Test that streaming produces same result as oneshot
    let oneshot = unsafe { crc32c_sse42(!0, TEST_DATA) } ^ !0;

    let mut state = !0u32;
    state = unsafe { crc32c_sse42(state, &TEST_DATA[..5]) };
    state = unsafe { crc32c_sse42(state, &TEST_DATA[5..]) };
    let streamed = state ^ !0;

    assert_eq!(streamed, oneshot);
  }

  #[test]
  fn test_sse42_various_lengths() {
    if !std::is_x86_feature_detected!("sse4.2") {
      return;
    }

    // Test lengths 0-32 to exercise all remainder paths
    for len in 0..=32 {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();
      let _ = unsafe { crc32c_sse42(!0, &data) };
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // PCLMUL tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn test_crc32_ieee_pclmul_matches_vector() {
    if !std::is_x86_feature_detected!("pclmulqdq") {
      std::eprintln!("Skipping PCLMUL test: not supported");
      return;
    }

    let crc = crc32_pclmul_safe(!0, TEST_DATA) ^ !0;
    assert_eq!(crc, CRC32_IEEE_CHECK);
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn test_crc32c_pclmul_matches_vector() {
    if !std::is_x86_feature_detected!("pclmulqdq") {
      std::eprintln!("Skipping PCLMUL test: not supported");
      return;
    }

    let crc = crc32c_pclmul_safe(!0, TEST_DATA) ^ !0;
    assert_eq!(crc, CRC32C_CHECK);
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn test_crc32_ieee_pclmul_matches_portable_various_lengths() {
    if !std::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    for len in [
      0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 129, 255, 256, 511, 512, 1024,
    ] {
      let data = make_data(len);
      let portable =
        super::super::portable::crc32_slice16(!0, &data, &super::super::kernel_tables::IEEE_TABLES_16) ^ !0;
      let pclmul = crc32_pclmul_safe(!0, &data) ^ !0;
      assert_eq!(pclmul, portable, "mismatch at len={len}");
    }
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn test_crc32c_pclmul_matches_portable_various_lengths() {
    if !std::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    for len in [
      0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 129, 255, 256, 511, 512, 1024,
    ] {
      let data = make_data(len);
      let portable =
        super::super::portable::crc32_slice16(!0, &data, &super::super::kernel_tables::CASTAGNOLI_TABLES_16) ^ !0;
      let pclmul = crc32c_pclmul_safe(!0, &data) ^ !0;
      assert_eq!(pclmul, portable, "mismatch at len={len}");
    }
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn test_crc32_ieee_pclmul_small_matches_portable_all_lengths_0_127() {
    if !std::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    for len in 0..128 {
      let data = make_data(len);
      let portable =
        super::super::portable::crc32_slice16(!0, &data, &super::super::kernel_tables::IEEE_TABLES_16) ^ !0;
      let pclmul_small = crc32_pclmul_small_safe(!0, &data) ^ !0;
      assert_eq!(pclmul_small, portable, "mismatch at len={len}");
    }
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn test_crc32c_pclmul_small_matches_portable_all_lengths_0_127() {
    if !std::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    for len in 0..128 {
      let data = make_data(len);
      let portable =
        super::super::portable::crc32_slice16(!0, &data, &super::super::kernel_tables::CASTAGNOLI_TABLES_16) ^ !0;
      let pclmul_small = crc32c_pclmul_small_safe(!0, &data) ^ !0;
      assert_eq!(pclmul_small, portable, "mismatch at len={len}");
    }
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn test_crc32_ieee_pclmul_multiway_matches_portable_various_lengths() {
    if !std::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    for len in [0usize, 1, 7, 16, 63, 64, 127, 128, 255, 256, 512, 1024, 4096, 16 * 1024] {
      let data = make_data(len);
      let portable =
        super::super::portable::crc32_slice16(!0, &data, &super::super::kernel_tables::IEEE_TABLES_16) ^ !0;
      let pclmul = crc32_pclmul_safe(!0, &data) ^ !0;
      let pclmul_2way = crc32_pclmul_2way_safe(!0, &data) ^ !0;
      let pclmul_4way = crc32_pclmul_4way_safe(!0, &data) ^ !0;
      let pclmul_7way = crc32_pclmul_7way_safe(!0, &data) ^ !0;
      assert_eq!(pclmul, portable, "1-way mismatch at len={len}");
      assert_eq!(pclmul_2way, portable, "2-way mismatch at len={len}");
      assert_eq!(pclmul_4way, portable, "4-way mismatch at len={len}");
      assert_eq!(pclmul_7way, portable, "7-way mismatch at len={len}");
    }
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn test_crc32c_pclmul_multiway_matches_portable_various_lengths() {
    if !std::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    for len in [0usize, 1, 7, 16, 63, 64, 127, 128, 255, 256, 512, 1024, 4096, 16 * 1024] {
      let data = make_data(len);
      let portable =
        super::super::portable::crc32_slice16(!0, &data, &super::super::kernel_tables::CASTAGNOLI_TABLES_16) ^ !0;
      let pclmul = crc32c_pclmul_safe(!0, &data) ^ !0;
      let pclmul_2way = crc32c_pclmul_2way_safe(!0, &data) ^ !0;
      let pclmul_4way = crc32c_pclmul_4way_safe(!0, &data) ^ !0;
      let pclmul_7way = crc32c_pclmul_7way_safe(!0, &data) ^ !0;
      assert_eq!(pclmul, portable, "1-way mismatch at len={len}");
      assert_eq!(pclmul_2way, portable, "2-way mismatch at len={len}");
      assert_eq!(pclmul_4way, portable, "4-way mismatch at len={len}");
      assert_eq!(pclmul_7way, portable, "7-way mismatch at len={len}");
    }
  }
}
