use core::arch::global_asm;

use super::{GAMMAS_MONT, Poly};

#[cfg(target_os = "macos")]
global_asm!(include_str!("../asm/rscrypto_mlkem_basemul_aarch64_apple_darwin.s"));
#[cfg(target_os = "linux")]
global_asm!(include_str!("../asm/rscrypto_mlkem_rej_uniform_aarch64_linux.s"));
#[cfg(target_os = "linux")]
global_asm!(include_str!("../asm/rscrypto_mlkem_basemul_aarch64_linux.s"));

#[cfg(target_os = "macos")]
unsafe extern "C" {
  fn rscrypto_mlkem_basemul_accumulate_aarch64_apple_darwin(
    acc: *mut u16,
    a: *const u16,
    b: *const u16,
    gammas_mont: *const i16,
  );
  fn rscrypto_mlkem_basemul_accumulate_k2_aarch64_apple_darwin(
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
  fn rscrypto_mlkem_rej_uniform_block_aarch64_linux(out: *mut u16, input: *const u8) -> usize;
  fn rscrypto_mlkem_rej_uniform_block_bounded_aarch64_linux(out: *mut u16, input: *const u8, cap: usize) -> usize;
  fn rscrypto_mlkem_rej_uniform_triple_block_aarch64_linux(
    out0: *mut u16,
    input0: *const u8,
    out1: *mut u16,
    input1: *const u8,
    out2: *mut u16,
    input2: *const u8,
  ) -> u64;
  fn rscrypto_mlkem_rej_uniform_triple_block_bounded_aarch64_linux(
    out0: *mut u16,
    input0: *const u8,
    out1: *mut u16,
    input1: *const u8,
    out2: *mut u16,
    input2: *const u8,
    caps: *const usize,
  ) -> u64;
  fn rscrypto_mlkem_basemul_accumulate_k2_aarch64_linux(
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

#[cfg(target_os = "linux")]
#[inline(always)]
fn unpack_triple_counts(packed: u64) -> [usize; 3] {
  [
    (packed & 0xffff) as usize,
    ((packed >> 16) & 0xffff) as usize,
    ((packed >> 32) & 0xffff) as usize,
  ]
}

#[cfg(target_os = "linux")]
#[inline]
/// # Safety
/// `input` must name 168 initialized readable bytes of public matrix-A XOF material.
/// `out` must be aligned and writable for 112 `u16` values, without overlapping the input.
/// Advanced SIMD must be available. The parser may write slack lanes beyond the returned count.
pub(super) unsafe fn sample_ntt_rej_uniform_block_asm(out: *mut u16, input: *const u8) -> usize {
  // SAFETY: Linux aarch64 SampleNTT rejection parser call because:
  // 1. `input` points to one readable 168-byte SHAKE128 rate block.
  // 2. `out` points to writable capacity for all 112 possible accepted candidates.
  // 3. Advanced SIMD is baseline for supported aarch64 Linux targets.
  // 4. Rejection branches and write positions depend only on public matrix-A XOF bytes.
  unsafe { rscrypto_mlkem_rej_uniform_block_aarch64_linux(out, input) }
}

#[cfg(target_os = "linux")]
#[inline]
/// # Safety
/// `input` must name 168 initialized readable bytes of public matrix-A XOF material.
/// `out` must be aligned and writable for `cap` `u16` values, without overlapping the input.
/// Advanced SIMD must be available.
pub(super) unsafe fn sample_ntt_rej_uniform_block_bounded_asm(out: *mut u16, input: *const u8, cap: usize) -> usize {
  // SAFETY: Linux aarch64 bounded SampleNTT rejection parser call because:
  // 1. `input` points to one readable 168-byte SHAKE128 rate block.
  // 2. `out` points to writable capacity for `cap` accepted candidates.
  // 3. The assembly caps writes at `cap` and returns the accepted count actually written.
  // 4. Advanced SIMD is baseline for supported aarch64 Linux targets.
  // 5. Rejection branches and write positions depend only on public matrix-A XOF bytes.
  unsafe { rscrypto_mlkem_rej_uniform_block_bounded_aarch64_linux(out, input, cap) }
}

#[cfg(target_os = "linux")]
#[inline]
/// # Safety
/// Each input must name 168 initialized readable bytes of public matrix-A XOF material.
/// Each output must be aligned and writable for 112 `u16` values. Output regions must be
/// pairwise disjoint and must not overlap any input. Advanced SIMD must be available.
/// The parser may write slack lanes beyond each returned count.
pub(super) unsafe fn sample_ntt_rej_uniform_triple_block_asm(
  out0: *mut u16,
  input0: *const u8,
  out1: *mut u16,
  input1: *const u8,
  out2: *mut u16,
  input2: *const u8,
) -> [usize; 3] {
  // SAFETY: Linux aarch64 triple SampleNTT rejection parser call because:
  // 1. Each input pointer names one readable 168-byte SHAKE128 rate block.
  // 2. Each output pointer names writable capacity for all 112 possible accepted candidates from its
  //    corresponding input block.
  // 3. The output regions are distinct for the duration of the assembly call.
  // 4. Advanced SIMD is baseline for supported aarch64 Linux targets.
  // 5. Rejection branches and write positions depend only on public matrix-A XOF bytes.
  let packed =
    unsafe { rscrypto_mlkem_rej_uniform_triple_block_aarch64_linux(out0, input0, out1, input1, out2, input2) };
  unpack_triple_counts(packed)
}

#[cfg(target_os = "linux")]
#[inline]
/// # Safety
/// Each input must name 168 initialized readable bytes of public matrix-A XOF material.
/// Each output must be aligned and writable for the corresponding `caps` element's number of
/// `u16` values. Output regions must be pairwise disjoint and must not overlap any input.
/// Advanced SIMD must be available.
pub(super) unsafe fn sample_ntt_rej_uniform_triple_block_bounded_asm(
  out0: *mut u16,
  input0: *const u8,
  out1: *mut u16,
  input1: *const u8,
  out2: *mut u16,
  input2: *const u8,
  caps: [usize; 3],
) -> [usize; 3] {
  // SAFETY: Linux aarch64 bounded triple SampleNTT rejection parser call because:
  // 1. Each input pointer names one readable 168-byte SHAKE128 rate block.
  // 2. Each output pointer names writable capacity for the matching `caps` element.
  // 3. `caps.as_ptr()` names three initialized `usize` capacities for the duration of this call.
  // 4. The assembly caps each lane's writes at its corresponding capacity and returns each accepted
  //    count in a 16-bit packed field.
  // 5. The output regions are distinct for the duration of the assembly call.
  // 6. Advanced SIMD is baseline for supported aarch64 Linux targets.
  // 7. Rejection branches and write positions depend only on public matrix-A XOF bytes.
  let packed = unsafe {
    rscrypto_mlkem_rej_uniform_triple_block_bounded_aarch64_linux(
      out0,
      input0,
      out1,
      input1,
      out2,
      input2,
      caps.as_ptr(),
    )
  };
  unpack_triple_counts(packed)
}

#[inline]
#[cfg(target_os = "macos")]
/// # Safety
///
/// The active platform must provide the matching ML-KEM AArch64 assembly
/// backend. `acc`, `a`, and `b` must be valid, non-overlapping polynomials.
pub(super) unsafe fn basemul_accumulate_asm(acc: &mut Poly, a: &Poly, b: &Poly) {
  // SAFETY: ML-KEM aarch64 base-multiply assembly call because:
  // 1. `acc`, `a`, and `b` are fixed 256-coefficient polynomials matching the assembly ABI.
  // 2. `GAMMAS_MONT` is a fixed 128-lane i16 table matching the 128 ML-KEM base-case products.
  // 3. This function is compiled only for aarch64 macOS with baseline Advanced SIMD support.
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

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[inline]
/// # Safety
///
/// The active platform must provide the matching ML-KEM AArch64 assembly
/// backend. `a` and `b` must each reference at least `2 * N` readable,
/// contiguous coefficients and must not overlap the uniquely borrowed `acc`.
pub(super) unsafe fn basemul_accumulate_k2_asm_ptr(acc: &mut Poly, a: *const u16, b: *const u16) {
  #[cfg(target_os = "macos")]
  {
    // SAFETY: ML-KEM aarch64 Darwin K=2 dot-product assembly call because:
    // 1. `acc` is a fixed 256-coefficient polynomial matching the assembly ABI.
    // 2. `a` and `b` point to contiguous `[[u16; 256]; 2]` polynomial vectors supplied by production
    //    `PolyVec<2>` storage.
    // 3. `GAMMAS_MONT` is a fixed 128-lane i16 table matching the 128 ML-KEM base-case products.
    // 4. This module is compiled only for aarch64 macOS with baseline Advanced SIMD support.
    // 5. The assembly performs fixed public loops with no coefficient-dependent memory addresses.
    unsafe {
      rscrypto_mlkem_basemul_accumulate_k2_aarch64_apple_darwin(acc.as_mut_ptr(), a, b, GAMMAS_MONT.as_ptr());
    }
  }

  #[cfg(target_os = "linux")]
  {
    // SAFETY: ML-KEM aarch64 Linux K=2 dot-product assembly call because:
    // 1. `acc` is a fixed 256-coefficient polynomial matching the assembly ABI.
    // 2. `a` and `b` point to contiguous `[[u16; 256]; 2]` polynomial vectors supplied by production
    //    `PolyVec<2>` storage.
    // 3. `GAMMAS_MONT` is a fixed 128-lane i16 table matching the 128 ML-KEM base-case products.
    // 4. This module is compiled only for aarch64 Linux with baseline Advanced SIMD support.
    // 5. The assembly performs fixed public loops with no coefficient-dependent memory addresses.
    unsafe {
      rscrypto_mlkem_basemul_accumulate_k2_aarch64_linux(acc.as_mut_ptr(), a, b, GAMMAS_MONT.as_ptr());
    }
  }
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[inline]
/// # Safety
///
/// The active platform must provide the matching ML-KEM AArch64 assembly
/// backend. `a` and `b` must each reference at least `3 * N` readable,
/// contiguous coefficients and must not overlap the uniquely borrowed `acc`.
pub(super) unsafe fn basemul_accumulate_k3_asm_ptr(acc: &mut Poly, a: *const u16, b: *const u16) {
  #[cfg(target_os = "macos")]
  {
    // SAFETY: ML-KEM aarch64 Darwin K=3 dot-product assembly call because:
    // 1. `acc` is a fixed 256-coefficient polynomial matching the assembly ABI.
    // 2. `a` and `b` point to contiguous `[[u16; 256]; 3]` polynomial vectors supplied by production
    //    `PolyVec<3>` storage.
    // 3. `GAMMAS_MONT` is a fixed 128-lane i16 table matching the 128 ML-KEM base-case products.
    // 4. This module is compiled only for aarch64 macOS with baseline Advanced SIMD support.
    // 5. The assembly performs fixed public loops with no coefficient-dependent memory addresses.
    unsafe {
      rscrypto_mlkem_basemul_accumulate_k3_aarch64_apple_darwin(acc.as_mut_ptr(), a, b, GAMMAS_MONT.as_ptr());
    }
  }

  #[cfg(target_os = "linux")]
  {
    // SAFETY: ML-KEM aarch64 Linux K=3 dot-product assembly call because:
    // 1. `acc` is a fixed 256-coefficient polynomial matching the assembly ABI.
    // 2. `a` and `b` point to contiguous `[[u16; 256]; 3]` polynomial vectors supplied by production
    //    `PolyVec<3>` storage.
    // 3. `GAMMAS_MONT` is a fixed 128-lane i16 table matching the 128 ML-KEM base-case products.
    // 4. This module is compiled only for aarch64 Linux with baseline Advanced SIMD support.
    // 5. The assembly performs fixed public loops with no coefficient-dependent memory addresses.
    unsafe {
      rscrypto_mlkem_basemul_accumulate_k3_aarch64_linux(acc.as_mut_ptr(), a, b, GAMMAS_MONT.as_ptr());
    }
  }
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[inline]
/// # Safety
///
/// The active platform must provide the matching ML-KEM AArch64 assembly
/// backend. `a` and `b` must each reference at least `4 * N` readable,
/// contiguous coefficients and must not overlap the uniquely borrowed `acc`.
pub(super) unsafe fn basemul_accumulate_k4_asm_ptr(acc: &mut Poly, a: *const u16, b: *const u16) {
  #[cfg(target_os = "macos")]
  {
    // SAFETY: ML-KEM aarch64 Darwin K=4 dot-product assembly call because:
    // 1. `acc` is a fixed 256-coefficient polynomial matching the assembly ABI.
    // 2. `a` and `b` point to contiguous `[[u16; 256]; 4]` polynomial vectors supplied by production
    //    `PolyVec<4>` storage.
    // 3. `GAMMAS_MONT` is a fixed 128-lane i16 table matching the 128 ML-KEM base-case products.
    // 4. This module is compiled only for aarch64 macOS with baseline Advanced SIMD support.
    // 5. The assembly performs fixed public loops with no coefficient-dependent memory addresses.
    unsafe {
      rscrypto_mlkem_basemul_accumulate_k4_aarch64_apple_darwin(acc.as_mut_ptr(), a, b, GAMMAS_MONT.as_ptr());
    }
  }

  #[cfg(target_os = "linux")]
  {
    // SAFETY: ML-KEM aarch64 Linux K=4 dot-product assembly call because:
    // 1. `acc` is a fixed 256-coefficient polynomial matching the assembly ABI.
    // 2. `a` and `b` point to contiguous `[[u16; 256]; 4]` polynomial vectors supplied by production
    //    `PolyVec<4>` storage.
    // 3. `GAMMAS_MONT` is a fixed 128-lane i16 table matching the 128 ML-KEM base-case products.
    // 4. This module is compiled only for aarch64 Linux with baseline Advanced SIMD support.
    // 5. The assembly performs fixed public loops with no coefficient-dependent memory addresses.
    unsafe {
      rscrypto_mlkem_basemul_accumulate_k4_aarch64_linux(acc.as_mut_ptr(), a, b, GAMMAS_MONT.as_ptr());
    }
  }
}
