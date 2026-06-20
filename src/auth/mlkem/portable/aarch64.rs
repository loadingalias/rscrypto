#![allow(clippy::indexing_slicing)] // Fixed-size ML-KEM native tables and assembly ABI.
#![allow(unsafe_code)]

use core::arch::global_asm;

#[cfg(test)]
use super::SAMPLE_NTT_ACC_CHUNK_COEFFS;
use super::{GAMMAS_MONT, Poly, PolyVec};

#[cfg(all(any(test, feature = "diag"), target_os = "macos"))]
global_asm!(include_str!("../asm/rscrypto_mlkem_ntt_aarch64_apple_darwin.s"));
#[cfg(target_os = "macos")]
global_asm!(include_str!("../asm/rscrypto_mlkem_basemul_aarch64_apple_darwin.s"));
#[cfg(all(any(test, feature = "diag"), target_os = "linux"))]
global_asm!(include_str!("../asm/rscrypto_mlkem_ntt_aarch64_linux.s"));
#[cfg(target_os = "linux")]
global_asm!(include_str!("../asm/rscrypto_mlkem_basemul_aarch64_linux.s"));

#[cfg(any(test, feature = "diag"))]
#[repr(align(16))]
struct AlignedI16<const N: usize>([i16; N]);

#[cfg(any(test, feature = "diag"))]
static NTT_ZETAS_LAYER_12345: AlignedI16<80> = AlignedI16([
  -1600, -15749, -749, -7373, -40, -394, -687, -6762, 630, 6201, -1432, -14095, 848, 8347, 0, 0, 1062, 10453, 296,
  2914, -882, -8682, 0, 0, -1410, -13879, 1339, 13180, 1476, 14529, 0, 0, 193, 1900, -283, -2786, 56, 551, 0, 0, 797,
  7845, -1089, -10719, 1333, 13121, 0, 0, -543, -5345, 1426, 14036, -1235, -12156, 0, 0, -69, -679, 535, 5266, -447,
  -4400, 0, 0, 569, 5601, -936, -9213, -450, -4429, 0, 0, -1583, -15582, -1355, -13338, 821, 8081, 0, 0,
]);

#[cfg(any(test, feature = "diag"))]
static NTT_ZETAS_LAYER_67: AlignedI16<384> = AlignedI16([
  289, 289, 331, 331, -76, -76, -1573, -1573, 2845, 2845, 3258, 3258, -748, -748, -15483, -15483, 17, 17, 583, 583,
  1637, 1637, -1041, -1041, 167, 167, 5739, 5739, 16113, 16113, -10247, -10247, -568, -568, -680, -680, 723, 723, 1100,
  1100, -5591, -5591, -6693, -6693, 7117, 7117, 10828, 10828, 1197, 1197, -1025, -1025, -1052, -1052, -1274, -1274,
  11782, 11782, -10089, -10089, -10355, -10355, -12540, -12540, 1409, 1409, -48, -48, 756, 756, -314, -314, 13869,
  13869, -472, -472, 7441, 7441, -3091, -3091, -667, -667, 233, 233, -1173, -1173, -279, -279, -6565, -6565, 2293,
  2293, -11546, -11546, -2746, -2746, 650, 650, -1352, -1352, -816, -816, 632, 632, 6398, 6398, -13308, -13308, -8032,
  -8032, 6221, 6221, -1626, -1626, -540, -540, -1482, -1482, 1461, 1461, -16005, -16005, -5315, -5315, -14588, -14588,
  14381, 14381, 1651, 1651, -1540, -1540, 952, 952, -642, -642, 16251, 16251, -15159, -15159, 9371, 9371, -6319, -6319,
  -464, -464, 33, 33, 1320, 1320, -1414, -1414, -4567, -4567, 325, 325, 12993, 12993, -13918, -13918, 939, 939, -892,
  -892, 733, 733, 268, 268, 9243, 9243, -8780, -8780, 7215, 7215, 2638, 2638, -1021, -1021, -941, -941, -992, -992,
  641, 641, -10050, -10050, -9262, -9262, -9764, -9764, 6309, 6309, -1010, -1010, 1435, 1435, 807, 807, 452, 452,
  -9942, -9942, 14125, 14125, 7943, 7943, 4449, 4449, 1584, 1584, -1292, -1292, 375, 375, -1239, -1239, 15592, 15592,
  -12717, -12717, 3691, 3691, -12196, -12196, -1031, -1031, -109, -109, -780, -780, 1645, 1645, -10148, -10148, -1073,
  -1073, -7678, -7678, 16192, 16192, 1438, 1438, -461, -461, 1534, 1534, -927, -927, 14155, 14155, -4538, -4538, 15099,
  15099, -9125, -9125, 1063, 1063, -556, -556, -1230, -1230, -863, -863, 10463, 10463, -5473, -5473, -12107, -12107,
  -8495, -8495, 319, 319, 757, 757, 561, 561, -735, -735, 3140, 3140, 7451, 7451, 5522, 5522, -7235, -7235, -682, -682,
  -712, -712, 1481, 1481, 648, 648, -6713, -6713, -7008, -7008, 14578, 14578, 6378, 6378, -525, -525, 403, 403, 1143,
  1143, -554, -554, -5168, -5168, 3967, 3967, 11251, 11251, -5453, -5453, 1092, 1092, 1026, 1026, -1179, -1179, 886,
  886, 10749, 10749, 10099, 10099, -11605, -11605, 8721, 8721, -855, -855, -219, -219, 1227, 1227, 910, 910, -8416,
  -8416, -2156, -2156, 12078, 12078, 8957, 8957, -1607, -1607, -1455, -1455, -1219, -1219, 885, 885, -15818, -15818,
  -14322, -14322, -11999, -11999, 8711, 8711, 1212, 1212, 1029, 1029, -394, -394, -1175, -1175, 11930, 11930, 10129,
  10129, -3878, -3878, -11566, -11566,
]);

#[cfg(target_os = "macos")]
unsafe extern "C" {
  #[cfg(any(test, feature = "diag"))]
  fn rscrypto_mlkem_ntt_aarch64_apple_darwin(poly: *mut i16, zetas12345: *const i16, zetas67: *const i16);
  fn rscrypto_mlkem_basemul_accumulate_aarch64_apple_darwin(
    acc: *mut u16,
    a: *const u16,
    b: *const u16,
    gammas_mont: *const i16,
  );
  #[cfg(test)]
  fn rscrypto_mlkem_basemul_accumulate_chunk_aarch64_apple_darwin(
    acc: *mut u16,
    a: *const u16,
    b: *const u16,
    gammas_mont: *const i16,
  );
  fn rscrypto_mlkem_basemul_accumulate_k3_aarch64_apple_darwin(
    acc: *mut u16,
    a: *const u16,
    b: *const u16,
    gammas_mont: *const i16,
  );
  fn rscrypto_mlkem_basemul_accumulate_k4_aarch64_apple_darwin(
    acc: *mut u16,
    a: *const u16,
    b: *const u16,
    gammas_mont: *const i16,
  );
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
  #[cfg(any(test, feature = "diag"))]
  fn rscrypto_mlkem_ntt_aarch64_linux(poly: *mut i16, zetas12345: *const i16, zetas67: *const i16);
  fn rscrypto_mlkem_basemul_accumulate_aarch64_linux(
    acc: *mut u16,
    a: *const u16,
    b: *const u16,
    gammas_mont: *const i16,
  );
  #[cfg(test)]
  fn rscrypto_mlkem_basemul_accumulate_chunk_aarch64_linux(
    acc: *mut u16,
    a: *const u16,
    b: *const u16,
    gammas_mont: *const i16,
  );
  fn rscrypto_mlkem_basemul_accumulate_k3_aarch64_linux(
    acc: *mut u16,
    a: *const u16,
    b: *const u16,
    gammas_mont: *const i16,
  );
  fn rscrypto_mlkem_basemul_accumulate_k4_aarch64_linux(
    acc: *mut u16,
    a: *const u16,
    b: *const u16,
    gammas_mont: *const i16,
  );
}

#[cfg(any(test, feature = "diag"))]
#[inline]
unsafe fn ntt_asm_raw(poly: &mut Poly) {
  #[cfg(target_os = "macos")]
  {
    // SAFETY: ML-KEM aarch64 NTT assembly call because:
    // 1. `poly` is a fixed 256-lane u16 array with the same layout and size as the assembly's i16
    //    polynomial ABI.
    // 2. The twiddle tables are fixed 16-byte-aligned `i16` arrays with the exact lengths required by
    //    the assembly ABI.
    // 3. This module is compiled only for aarch64 macOS with baseline Advanced SIMD support.
    // 4. The assembly memory access schedule depends only on public ML-KEM dimensions.
    unsafe {
      rscrypto_mlkem_ntt_aarch64_apple_darwin(
        poly.as_mut_ptr().cast::<i16>(),
        NTT_ZETAS_LAYER_12345.0.as_ptr(),
        NTT_ZETAS_LAYER_67.0.as_ptr(),
      );
    }
  }

  #[cfg(target_os = "linux")]
  {
    // SAFETY: ML-KEM aarch64 NTT assembly call because:
    // 1. `poly` is a fixed 256-lane u16 array with the same layout and size as the assembly's i16
    //    polynomial ABI.
    // 2. The twiddle tables are fixed 16-byte-aligned `i16` arrays with the exact lengths required by
    //    the assembly ABI.
    // 3. This module is compiled only for aarch64 Linux with baseline Advanced SIMD support.
    // 4. The assembly memory access schedule depends only on public ML-KEM dimensions.
    unsafe {
      rscrypto_mlkem_ntt_aarch64_linux(
        poly.as_mut_ptr().cast::<i16>(),
        NTT_ZETAS_LAYER_12345.0.as_ptr(),
        NTT_ZETAS_LAYER_67.0.as_ptr(),
      );
    }
  }
}

#[cfg(any(test, feature = "diag"))]
#[inline]
unsafe fn ntt_asm(poly: &mut Poly) {
  // SAFETY: exact ML-KEM aarch64 NTT assembly call because:
  // 1. `poly` is a fixed 256-coefficient polynomial matching the assembly ABI.
  // 2. The raw wrapper supplies the fixed, aligned twiddle tables required by that ABI.
  // 3. The assembly canonicalizes its redundant signed output to the scalar/FIPS representation
  //    before returning.
  unsafe {
    ntt_asm_raw(poly);
  }
}

#[cfg(test)]
pub(super) unsafe fn test_ntt_asm_raw(poly: &mut Poly) {
  // SAFETY: test-only direct access to the raw assembly entry point because:
  // 1. The caller supplies a fixed 256-coefficient ML-KEM polynomial.
  // 2. This module is compiled only on aarch64 targets with the assembly backend available.
  // 3. The raw assembly symbol returns the scalar/FIPS canonical representation before tests compare
  //    it with the scalar oracle.
  unsafe {
    ntt_asm_raw(poly);
  }
}

#[cfg(test)]
pub(super) unsafe fn test_ntt_asm(poly: &mut Poly) {
  // SAFETY: test-only direct access to the exact assembly wrapper because:
  // 1. The caller supplies a fixed 256-coefficient ML-KEM polynomial.
  // 2. This module is compiled only on aarch64 targets with the assembly backend available.
  // 3. The assembly returns the same canonical representation as the scalar/FIPS path.
  unsafe {
    ntt_asm(poly);
  }
}

#[cfg(feature = "diag")]
pub(super) unsafe fn diag_ntt_asm_digest(seed: u16) -> u16 {
  let poly = super::diag_poly(seed);
  // SAFETY: forwarded from this function's caller contract.
  unsafe { diag_ntt_asm_input_digest(poly) }
}

#[cfg(feature = "diag")]
pub(super) unsafe fn diag_ntt_asm_input_digest(mut poly: Poly) -> u16 {
  // SAFETY: Direct NTT assembly diagnostic call because:
  // 1. The caller guarantees this runs only on an aarch64 CPU with Advanced SIMD.
  // 2. `poly` is a fixed 256-coefficient polynomial matching the assembly ABI.
  // 3. This diagnostic root intentionally bypasses production dispatch so benchmark and CT artifacts
  //    can inspect the aarch64 assembly candidate itself.
  unsafe {
    ntt_asm(&mut poly);
  }
  let digest = super::diag_fold_poly(&poly);
  super::zeroize_poly(&mut poly);
  digest
}

#[inline]
unsafe fn basemul_accumulate_asm(acc: &mut Poly, a: &Poly, b: &Poly) {
  #[cfg(target_os = "macos")]
  {
    // SAFETY: ML-KEM aarch64 base-multiply assembly call because:
    // 1. `acc`, `a`, and `b` are fixed 256-coefficient polynomials matching the assembly ABI.
    // 2. `GAMMAS_MONT` is a fixed 128-lane i16 table matching the 128 ML-KEM base-case products.
    // 3. This module is compiled only for aarch64 macOS with baseline Advanced SIMD support.
    // 4. The assembly performs 16 fixed chunks with memory addresses determined only by public ML-KEM
    //    dimensions.
    unsafe {
      rscrypto_mlkem_basemul_accumulate_aarch64_apple_darwin(
        acc.as_mut_ptr(),
        a.as_ptr(),
        b.as_ptr(),
        GAMMAS_MONT.as_ptr(),
      );
    }
  }

  #[cfg(target_os = "linux")]
  {
    // SAFETY: ML-KEM aarch64 base-multiply assembly call because:
    // 1. `acc`, `a`, and `b` are fixed 256-coefficient polynomials matching the assembly ABI.
    // 2. `GAMMAS_MONT` is a fixed 128-lane i16 table matching the 128 ML-KEM base-case products.
    // 3. This module is compiled only for aarch64 Linux with baseline Advanced SIMD support.
    // 4. The assembly performs 16 fixed chunks with memory addresses determined only by public ML-KEM
    //    dimensions.
    unsafe {
      rscrypto_mlkem_basemul_accumulate_aarch64_linux(acc.as_mut_ptr(), a.as_ptr(), b.as_ptr(), GAMMAS_MONT.as_ptr());
    }
  }
}

#[cfg(test)]
pub(super) unsafe fn test_basemul_accumulate_asm(acc: &mut Poly, a: &Poly, b: &Poly) {
  // SAFETY: test-only direct access to the full base-multiply assembly entry point because:
  // 1. `acc`, `a`, and `b` are fixed 256-coefficient ML-KEM polynomials.
  // 2. This module is compiled only on aarch64 targets with the assembly backend available.
  // 3. Tests compare the output against the scalar/FIPS accumulator before the symbol is considered
  //    for production dispatch.
  unsafe {
    basemul_accumulate_asm(acc, a, b);
  }
}

#[cfg(feature = "diag")]
pub(super) unsafe fn diag_basemul_accumulate_asm_digest(seed: u16) -> u16 {
  let a = super::diag_poly(seed);
  let b = super::diag_poly(seed.wrapping_add(1));
  let acc = super::diag_poly(seed.wrapping_add(2));
  // SAFETY: forwarded from this function's caller contract.
  unsafe { diag_basemul_accumulate_asm_input_digest(a, b, acc) }
}

#[inline]
unsafe fn basemul_accumulate_k3_asm(acc: &mut Poly, a: &PolyVec<3>, b: &PolyVec<3>) {
  #[cfg(target_os = "macos")]
  {
    // SAFETY: ML-KEM aarch64 K=3 dot-product assembly call because:
    // 1. `acc`, `a`, and `b` are fixed-size ML-KEM polynomial arrays matching the assembly ABI.
    // 2. `a` and `b` are contiguous `[[u16; 256]; 3]` values, so the assembly's fixed 512-byte
    //    polynomial strides stay in bounds.
    // 3. `GAMMAS_MONT` is a fixed 128-lane i16 table matching the 128 ML-KEM base-case products.
    // 4. This module is compiled only for aarch64 macOS with baseline Advanced SIMD support.
    // 5. The assembly performs fixed public loops with no coefficient-dependent memory addresses.
    unsafe {
      rscrypto_mlkem_basemul_accumulate_k3_aarch64_apple_darwin(
        acc.as_mut_ptr(),
        a.as_ptr().cast::<u16>(),
        b.as_ptr().cast::<u16>(),
        GAMMAS_MONT.as_ptr(),
      );
    }
  }

  #[cfg(target_os = "linux")]
  {
    // SAFETY: ML-KEM aarch64 K=3 dot-product assembly call because:
    // 1. `acc`, `a`, and `b` are fixed-size ML-KEM polynomial arrays matching the assembly ABI.
    // 2. `a` and `b` are contiguous `[[u16; 256]; 3]` values, so the assembly's fixed 512-byte
    //    polynomial strides stay in bounds.
    // 3. `GAMMAS_MONT` is a fixed 128-lane i16 table matching the 128 ML-KEM base-case products.
    // 4. This module is compiled only for aarch64 Linux with baseline Advanced SIMD support.
    // 5. The assembly performs fixed public loops with no coefficient-dependent memory addresses.
    unsafe {
      rscrypto_mlkem_basemul_accumulate_k3_aarch64_linux(
        acc.as_mut_ptr(),
        a.as_ptr().cast::<u16>(),
        b.as_ptr().cast::<u16>(),
        GAMMAS_MONT.as_ptr(),
      );
    }
  }
}

#[inline]
unsafe fn basemul_accumulate_k4_asm(acc: &mut Poly, a: &PolyVec<4>, b: &PolyVec<4>) {
  #[cfg(target_os = "macos")]
  {
    // SAFETY: ML-KEM aarch64 K=4 dot-product assembly call because:
    // 1. `acc`, `a`, and `b` are fixed-size ML-KEM polynomial arrays matching the assembly ABI.
    // 2. `a` and `b` are contiguous `[[u16; 256]; 4]` values, so the assembly's fixed 512-byte
    //    polynomial strides stay in bounds.
    // 3. `GAMMAS_MONT` is a fixed 128-lane i16 table matching the 128 ML-KEM base-case products.
    // 4. This module is compiled only for aarch64 macOS with baseline Advanced SIMD support.
    // 5. The assembly performs fixed public loops with no coefficient-dependent memory addresses.
    unsafe {
      rscrypto_mlkem_basemul_accumulate_k4_aarch64_apple_darwin(
        acc.as_mut_ptr(),
        a.as_ptr().cast::<u16>(),
        b.as_ptr().cast::<u16>(),
        GAMMAS_MONT.as_ptr(),
      );
    }
  }

  #[cfg(target_os = "linux")]
  {
    // SAFETY: ML-KEM aarch64 K=4 dot-product assembly call because:
    // 1. `acc`, `a`, and `b` are fixed-size ML-KEM polynomial arrays matching the assembly ABI.
    // 2. `a` and `b` are contiguous `[[u16; 256]; 4]` values, so the assembly's fixed 512-byte
    //    polynomial strides stay in bounds.
    // 3. `GAMMAS_MONT` is a fixed 128-lane i16 table matching the 128 ML-KEM base-case products.
    // 4. This module is compiled only for aarch64 Linux with baseline Advanced SIMD support.
    // 5. The assembly performs fixed public loops with no coefficient-dependent memory addresses.
    unsafe {
      rscrypto_mlkem_basemul_accumulate_k4_aarch64_linux(
        acc.as_mut_ptr(),
        a.as_ptr().cast::<u16>(),
        b.as_ptr().cast::<u16>(),
        GAMMAS_MONT.as_ptr(),
      );
    }
  }
}

/// Diagnostic digest for the rscrypto-owned aarch64 base-multiply accumulator.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 Linux/macOS targets with baseline
/// Advanced SIMD available. The module cfg enforces the target/OS half of that contract.
#[cfg(feature = "diag")]
pub(super) unsafe fn diag_basemul_accumulate_asm_input_digest(a: Poly, b: Poly, mut acc: Poly) -> u16 {
  // SAFETY: Direct owned-assembly diagnostic call because:
  // 1. The caller guarantees the function runs only on an aarch64 CPU with Advanced SIMD.
  // 2. `acc`, `a`, and `b` are fixed 256-coefficient polynomials matching the assembly ABI.
  // 3. The borrowed inputs are stack-owned in this function and cannot alias `acc`.
  // 4. This diagnostic root intentionally bypasses production dispatch so benchmark and CT artifacts
  //    can inspect the owned aarch64 assembly kernel itself.
  unsafe {
    basemul_accumulate_asm(&mut acc, &a, &b);
  }
  let digest = super::diag_fold_poly(&acc);
  super::zeroize_poly(&mut acc);
  digest
}

#[cfg(test)]
pub(super) unsafe fn test_basemul_accumulate_k3_asm(acc: &mut Poly, a: &PolyVec<3>, b: &PolyVec<3>) {
  // SAFETY: test-only direct access to the K=3 dot-product assembly entry point because:
  // 1. `acc`, `a`, and `b` are fixed-size ML-KEM polynomial arrays.
  // 2. This module is compiled only on aarch64 Linux/macOS targets with the assembly backend.
  // 3. Tests compare the output against the scalar/FIPS accumulator before production dispatch.
  unsafe {
    basemul_accumulate_k3_asm(acc, a, b);
  }
}

#[cfg(test)]
pub(super) unsafe fn test_basemul_accumulate_k4_asm(acc: &mut Poly, a: &PolyVec<4>, b: &PolyVec<4>) {
  // SAFETY: test-only direct access to the K=4 dot-product assembly entry point because:
  // 1. `acc`, `a`, and `b` are fixed-size ML-KEM polynomial arrays.
  // 2. This module is compiled only on aarch64 Linux/macOS targets with the assembly backend.
  // 3. Tests compare the output against the scalar/FIPS accumulator before production dispatch.
  unsafe {
    basemul_accumulate_k4_asm(acc, a, b);
  }
}

#[cfg(feature = "diag")]
pub(super) unsafe fn diag_basemul_accumulate_k3_asm_digest(seed: u16) -> u16 {
  let a = [
    super::diag_poly(seed),
    super::diag_poly(seed.wrapping_add(1)),
    super::diag_poly(seed.wrapping_add(2)),
  ];
  let b = [
    super::diag_poly(seed.wrapping_add(3)),
    super::diag_poly(seed.wrapping_add(4)),
    super::diag_poly(seed.wrapping_add(5)),
  ];
  let acc = super::diag_poly(seed.wrapping_add(6));
  // SAFETY: forwarded from this function's caller contract.
  unsafe { diag_basemul_accumulate_k3_asm_input_digest(a, b, acc) }
}

#[cfg(feature = "diag")]
pub(super) unsafe fn diag_basemul_accumulate_k4_asm_digest(seed: u16) -> u16 {
  let a = [
    super::diag_poly(seed),
    super::diag_poly(seed.wrapping_add(1)),
    super::diag_poly(seed.wrapping_add(2)),
    super::diag_poly(seed.wrapping_add(3)),
  ];
  let b = [
    super::diag_poly(seed.wrapping_add(4)),
    super::diag_poly(seed.wrapping_add(5)),
    super::diag_poly(seed.wrapping_add(6)),
    super::diag_poly(seed.wrapping_add(7)),
  ];
  let acc = super::diag_poly(seed.wrapping_add(8));
  // SAFETY: forwarded from this function's caller contract.
  unsafe { diag_basemul_accumulate_k4_asm_input_digest(a, b, acc) }
}

/// Diagnostic digest for the rscrypto-owned aarch64 K=3 base-multiply accumulator.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 Linux/macOS targets with baseline
/// Advanced SIMD available. The module cfg enforces the target/OS half of that contract.
#[cfg(feature = "diag")]
pub(super) unsafe fn diag_basemul_accumulate_k3_asm_input_digest(
  mut a: PolyVec<3>,
  mut b: PolyVec<3>,
  mut acc: Poly,
) -> u16 {
  // SAFETY: Direct owned-assembly diagnostic call because:
  // 1. The caller guarantees the function runs only on an aarch64 CPU with Advanced SIMD.
  // 2. `acc`, `a`, and `b` are fixed-size ML-KEM polynomial arrays matching the assembly ABI.
  // 3. The borrowed inputs are stack-owned in this function and cannot alias `acc`.
  // 4. This diagnostic root intentionally bypasses production dispatch so benchmark and CT artifacts
  //    can inspect the owned aarch64 assembly kernel itself.
  unsafe {
    basemul_accumulate_k3_asm(&mut acc, &a, &b);
  }
  let digest = super::diag_fold_poly(&acc);
  super::zeroize_polyvec(&mut a);
  super::zeroize_polyvec(&mut b);
  super::zeroize_poly(&mut acc);
  digest
}

/// Diagnostic digest for the rscrypto-owned aarch64 K=4 base-multiply accumulator.
///
/// # Safety
///
/// The caller must only execute this on supported aarch64 Linux/macOS targets with baseline
/// Advanced SIMD available. The module cfg enforces the target/OS half of that contract.
#[cfg(feature = "diag")]
pub(super) unsafe fn diag_basemul_accumulate_k4_asm_input_digest(
  mut a: PolyVec<4>,
  mut b: PolyVec<4>,
  mut acc: Poly,
) -> u16 {
  // SAFETY: Direct owned-assembly diagnostic call because:
  // 1. The caller guarantees the function runs only on an aarch64 CPU with Advanced SIMD.
  // 2. `acc`, `a`, and `b` are fixed-size ML-KEM polynomial arrays matching the assembly ABI.
  // 3. The borrowed inputs are stack-owned in this function and cannot alias `acc`.
  // 4. This diagnostic root intentionally bypasses production dispatch so benchmark and CT artifacts
  //    can inspect the owned aarch64 assembly kernel itself.
  unsafe {
    basemul_accumulate_k4_asm(&mut acc, &a, &b);
  }
  let digest = super::diag_fold_poly(&acc);
  super::zeroize_polyvec(&mut a);
  super::zeroize_polyvec(&mut b);
  super::zeroize_poly(&mut acc);
  digest
}

#[cfg(test)]
pub(super) unsafe fn test_basemul_accumulate_chunk_asm(
  acc: &mut Poly,
  a: &[u16; SAMPLE_NTT_ACC_CHUNK_COEFFS],
  b: &Poly,
  coeff_offset: usize,
) {
  debug_assert_eq!(coeff_offset % SAMPLE_NTT_ACC_CHUNK_COEFFS, 0);
  debug_assert!(coeff_offset.strict_add(SAMPLE_NTT_ACC_CHUNK_COEFFS) <= acc.len());
  let gamma_offset = coeff_offset / 2;

  #[cfg(target_os = "macos")]
  {
    // SAFETY: test-only direct access to the chunk base-multiply assembly entry point because:
    // 1. `a` contains exactly one 16-coefficient SampleNTT chunk.
    // 2. `coeff_offset` is checked to keep all `acc`, `b`, and `GAMMAS_MONT` accesses in bounds.
    // 3. Tests compare the output against the scalar/FIPS chunk accumulator before the symbol is
    //    considered for production dispatch.
    // 4. The assembly memory schedule is fixed for one public 16-coefficient chunk.
    unsafe {
      rscrypto_mlkem_basemul_accumulate_chunk_aarch64_apple_darwin(
        acc.as_mut_ptr().add(coeff_offset),
        a.as_ptr(),
        b.as_ptr().add(coeff_offset),
        GAMMAS_MONT.as_ptr().add(gamma_offset),
      );
    }
  }

  #[cfg(target_os = "linux")]
  {
    // SAFETY: test-only direct access to the chunk base-multiply assembly entry point because:
    // 1. `a` contains exactly one 16-coefficient SampleNTT chunk.
    // 2. `coeff_offset` is checked to keep all `acc`, `b`, and `GAMMAS_MONT` accesses in bounds.
    // 3. Tests compare the output against the scalar/FIPS chunk accumulator before the symbol is
    //    considered for production dispatch.
    // 4. The assembly memory schedule is fixed for one public 16-coefficient chunk.
    unsafe {
      rscrypto_mlkem_basemul_accumulate_chunk_aarch64_linux(
        acc.as_mut_ptr().add(coeff_offset),
        a.as_ptr(),
        b.as_ptr().add(coeff_offset),
        GAMMAS_MONT.as_ptr().add(gamma_offset),
      );
    }
  }
}
