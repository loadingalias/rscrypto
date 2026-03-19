//! Keccak-f[1600] x86_64 AVX2 kernel.
//!
//! Uses scalar operations for θ, ρ+π, and ι (near-optimal with BMI1 ANDN and
//! ROL/RORX on x86-64) and AVX2 for the χ step.
//!
//! χ: `a ^ (~b & c)` via `VPANDN` + `VPXOR` (2 ops per 4 elements), with
//! shifted versions built via `VPERMQ` + `VPBLENDD`. The 5th element of each
//! row uses scalar ANDN+XOR.
//!
//! This is the fallback SIMD tier when AVX-512 is unavailable. The gain over
//! portable is modest — x86-64 scalar with BMI1 ANDN is already strong for
//! single-state Keccak. The primary value is completing the dispatch cascade
//! (AVX-512 > AVX2 > Portable) and establishing the kernel slot for future
//! multi-state parallel optimizations.
//!
//! Available on Intel Haswell+ (2013+) and AMD Excavator+/Zen+ (2017+).
//!
//! # Safety
//!
//! All functions require the `avx2` target feature.

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::inline_always)]
#![allow(clippy::undocumented_unsafe_blocks)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Keccak-f[1600] permutation with AVX2 chi step.
///
/// θ, ρ+π, ι remain scalar (25 GPR variables). χ uses `VPANDN` + `VPXOR` to
/// compute `a ^ (~b & c)` for 4 elements per row.
///
/// # Safety
///
/// Caller must ensure `avx2` CPU feature is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn keccakf_avx2_impl(state: &mut [u64; 25]) {
  let mut a0 = state[0];
  let mut a1 = state[1];
  let mut a2 = state[2];
  let mut a3 = state[3];
  let mut a4 = state[4];
  let mut a5 = state[5];
  let mut a6 = state[6];
  let mut a7 = state[7];
  let mut a8 = state[8];
  let mut a9 = state[9];
  let mut a10 = state[10];
  let mut a11 = state[11];
  let mut a12 = state[12];
  let mut a13 = state[13];
  let mut a14 = state[14];
  let mut a15 = state[15];
  let mut a16 = state[16];
  let mut a17 = state[17];
  let mut a18 = state[18];
  let mut a19 = state[19];
  let mut a20 = state[20];
  let mut a21 = state[21];
  let mut a22 = state[22];
  let mut a23 = state[23];
  let mut a24 = state[24];

  for &rc in &super::RC {
    // ---- θ: column parity ----
    let c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20;
    let c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21;
    let c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22;
    let c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23;
    let c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24;

    // ---- θ: diffusion ----
    let d0 = c4 ^ c1.rotate_left(1);
    let d1 = c0 ^ c2.rotate_left(1);
    let d2 = c1 ^ c3.rotate_left(1);
    let d3 = c2 ^ c4.rotate_left(1);
    let d4 = c3 ^ c0.rotate_left(1);

    // ---- θ: XOR-back ----
    a0 ^= d0;
    a5 ^= d0;
    a10 ^= d0;
    a15 ^= d0;
    a20 ^= d0;

    a1 ^= d1;
    a6 ^= d1;
    a11 ^= d1;
    a16 ^= d1;
    a21 ^= d1;

    a2 ^= d2;
    a7 ^= d2;
    a12 ^= d2;
    a17 ^= d2;
    a22 ^= d2;

    a3 ^= d3;
    a8 ^= d3;
    a13 ^= d3;
    a18 ^= d3;
    a23 ^= d3;

    a4 ^= d4;
    a9 ^= d4;
    a14 ^= d4;
    a19 ^= d4;
    a24 ^= d4;

    // ---- ρ + π ----
    let b0 = a0;
    let b10 = a1.rotate_left(1);
    let b20 = a2.rotate_left(62);
    let b5 = a3.rotate_left(28);
    let b15 = a4.rotate_left(27);

    let b16 = a5.rotate_left(36);
    let b1 = a6.rotate_left(44);
    let b11 = a7.rotate_left(6);
    let b21 = a8.rotate_left(55);
    let b6 = a9.rotate_left(20);

    let b7 = a10.rotate_left(3);
    let b17 = a11.rotate_left(10);
    let b2 = a12.rotate_left(43);
    let b12 = a13.rotate_left(25);
    let b22 = a14.rotate_left(39);

    let b23 = a15.rotate_left(41);
    let b8 = a16.rotate_left(45);
    let b18 = a17.rotate_left(15);
    let b3 = a18.rotate_left(21);
    let b13 = a19.rotate_left(8);

    let b14 = a20.rotate_left(18);
    let b24 = a21.rotate_left(2);
    let b9 = a22.rotate_left(61);
    let b19 = a23.rotate_left(56);
    let b4 = a24.rotate_left(14);

    // ---- χ: VPANDN + VPXOR = a ^ (~b & c) ----
    //
    // For each row of 5 elements: pack b[0..4] into __m256i, create shifted
    // versions via VPERMQ + VPBLENDD, apply VPANDN + VPXOR, extract 4 results.
    // The 5th element uses scalar ANDN+XOR.
    macro_rules! chi_row {
      ($a0:ident, $a1:ident, $a2:ident, $a3:ident, $a4:ident,
       $b0:ident, $b1:ident, $b2:ident, $b3:ident, $b4:ident) => {{
        let row = _mm256_set_epi64x($b3 as i64, $b2 as i64, $b1 as i64, $b0 as i64);
        let b4_bc = _mm256_set1_epi64x($b4 as i64);
        // shifted1[i] = b[i+1]: [b1, b2, b3, b4]
        // VPERMQ rotates lanes left by 1: [b1, b2, b3, b0]
        // VPBLENDD replaces lane 3 (32-bit lanes 6,7) with b4
        let perm1 = _mm256_permute4x64_epi64(row, 0x39);
        let shifted1 = _mm256_blend_epi32(perm1, b4_bc, 0xC0_u32 as i32);
        // shifted2[i] = b[i+2]: [b2, b3, b4, b0]
        // VPERMQ rotates lanes left by 2: [b2, b3, b0, b0]
        // VPBLENDD replaces lane 2 (32-bit lanes 4,5) with b4
        let perm2 = _mm256_permute4x64_epi64(row, 0x0E);
        let shifted2 = _mm256_blend_epi32(perm2, b4_bc, 0x30);
        // VPANDN(a, b) = ~a & b, then XOR with row
        let not_b_and_c = _mm256_andnot_si256(shifted1, shifted2);
        let result = _mm256_xor_si256(row, not_b_and_c);
        // Extract 4 lanes
        let lo = _mm256_castsi256_si128(result);
        let hi = _mm256_extracti128_si256(result, 1);
        $a0 = _mm_extract_epi64(lo, 0) as u64;
        $a1 = _mm_extract_epi64(lo, 1) as u64;
        $a2 = _mm_extract_epi64(hi, 0) as u64;
        $a3 = _mm_extract_epi64(hi, 1) as u64;
        // 5th element: scalar ANDN+XOR
        $a4 = $b4 ^ (!$b0 & $b1);
      }};
    }

    // Row 0
    chi_row!(a0, a1, a2, a3, a4, b0, b1, b2, b3, b4);
    // Row 1
    chi_row!(a5, a6, a7, a8, a9, b5, b6, b7, b8, b9);
    // Row 2
    chi_row!(a10, a11, a12, a13, a14, b10, b11, b12, b13, b14);
    // Row 3
    chi_row!(a15, a16, a17, a18, a19, b15, b16, b17, b18, b19);
    // Row 4
    chi_row!(a20, a21, a22, a23, a24, b20, b21, b22, b23, b24);

    // ---- ι ----
    a0 ^= rc;
  }

  // Store state back.
  state[0] = a0;
  state[1] = a1;
  state[2] = a2;
  state[3] = a3;
  state[4] = a4;
  state[5] = a5;
  state[6] = a6;
  state[7] = a7;
  state[8] = a8;
  state[9] = a9;
  state[10] = a10;
  state[11] = a11;
  state[12] = a12;
  state[13] = a13;
  state[14] = a14;
  state[15] = a15;
  state[16] = a16;
  state[17] = a17;
  state[18] = a18;
  state[19] = a19;
  state[20] = a20;
  state[21] = a21;
  state[22] = a22;
  state[23] = a23;
  state[24] = a24;
}

/// Keccak-f[1600] permutation using x86_64 AVX2.
///
/// Requires `x86::AVX2` capability (verified by dispatch).
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn keccakf_x86_avx2(state: &mut [u64; 25]) {
  // SAFETY: Dispatch verifies AVX2 capability before selecting this kernel.
  unsafe { keccakf_avx2_impl(state) }
}
