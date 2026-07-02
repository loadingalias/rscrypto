#![allow(clippy::indexing_slicing)] // Fixed-size ML-KEM native tables and assembly ABI.
#![allow(unsafe_code)]

use core::arch::global_asm;

#[cfg(any(feature = "diag", all(test, target_os = "linux")))]
use super::PolyVec;
#[cfg(all(test, target_os = "linux"))]
use super::SAMPLE_NTT_ACC_CHUNK_COEFFS;
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
  #[cfg(any(test, feature = "diag"))]
  fn rscrypto_mlkem_rej_uniform_3blocks_aarch64_linux(out: *mut u16, input: *const u8) -> usize;
  #[cfg(any(test, feature = "diag"))]
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

#[cfg(all(any(test, feature = "diag"), target_os = "linux"))]
#[inline]
pub(super) unsafe fn sample_ntt_rej_uniform_3blocks_asm(out: *mut u16, input: *const u8) -> usize {
  // SAFETY: Linux aarch64 three-block SampleNTT rejection parser call because:
  // 1. `input` points to three contiguous readable SHAKE128 rate blocks.
  // 2. `out` points to writable capacity for one full 256-coefficient ML-KEM polynomial.
  // 3. The assembly caps writes at 256 accepted candidates and returns the accepted count written.
  // 4. Advanced SIMD is baseline for supported aarch64 Linux targets.
  // 5. Rejection branches and write positions depend only on public matrix-A XOF bytes.
  unsafe { rscrypto_mlkem_rej_uniform_3blocks_aarch64_linux(out, input) }
}

#[inline]
#[cfg(any(test, feature = "diag", target_os = "macos"))]
pub(super) unsafe fn basemul_accumulate_asm(acc: &mut Poly, a: &Poly, b: &Poly) {
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

#[cfg(all(test, target_os = "linux"))]
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

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[inline]
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

#[inline]
#[cfg(all(test, target_os = "linux"))]
unsafe fn basemul_accumulate_k2_asm(acc: &mut Poly, a: &PolyVec<2>, b: &PolyVec<2>) {
  // SAFETY: ML-KEM aarch64 K=2 dot-product assembly call because:
  // 1. `acc`, `a`, and `b` are fixed-size ML-KEM polynomial arrays matching the assembly ABI.
  // 2. `a` and `b` are contiguous `[[u16; 256]; 2]` values, so the assembly's fixed 512-byte
  //    polynomial strides stay in bounds.
  // 3. The assembly performs fixed public loops with no coefficient-dependent memory addresses.
  unsafe {
    basemul_accumulate_k2_asm_ptr(acc, a.as_ptr().cast::<u16>(), b.as_ptr().cast::<u16>());
  }
}

#[inline]
#[cfg(any(feature = "diag", all(test, target_os = "linux")))]
unsafe fn basemul_accumulate_k3_asm(acc: &mut Poly, a: &PolyVec<3>, b: &PolyVec<3>) {
  // SAFETY: forwards contiguous `PolyVec<3>` storage to the target K=3 pointer ABI.
  unsafe {
    basemul_accumulate_k3_asm_ptr(acc, a.as_ptr().cast::<u16>(), b.as_ptr().cast::<u16>());
  }
}

#[inline]
#[cfg(any(feature = "diag", all(test, target_os = "linux")))]
unsafe fn basemul_accumulate_k4_asm(acc: &mut Poly, a: &PolyVec<4>, b: &PolyVec<4>) {
  // SAFETY: forwards contiguous `PolyVec<4>` storage to the target K=4 pointer ABI.
  unsafe {
    basemul_accumulate_k4_asm_ptr(acc, a.as_ptr().cast::<u16>(), b.as_ptr().cast::<u16>());
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

#[cfg(all(test, target_os = "linux"))]
pub(super) unsafe fn test_basemul_accumulate_k2_asm(acc: &mut Poly, a: &PolyVec<2>, b: &PolyVec<2>) {
  // SAFETY: test-only direct access to the K=2 dot-product assembly entry point because:
  // 1. `acc`, `a`, and `b` are fixed-size ML-KEM polynomial arrays.
  // 2. This module is compiled only on aarch64 Linux/macOS targets with the assembly backend.
  // 3. Tests compare the output against the scalar/FIPS accumulator before production dispatch.
  unsafe {
    basemul_accumulate_k2_asm(acc, a, b);
  }
}

#[cfg(all(test, target_os = "linux"))]
pub(super) unsafe fn test_basemul_accumulate_k3_asm(acc: &mut Poly, a: &PolyVec<3>, b: &PolyVec<3>) {
  // SAFETY: test-only direct access to the K=3 dot-product assembly entry point because:
  // 1. `acc`, `a`, and `b` are fixed-size ML-KEM polynomial arrays.
  // 2. This module is compiled only on aarch64 Linux/macOS targets with the assembly backend.
  // 3. Tests compare the output against the scalar/FIPS accumulator before production dispatch.
  unsafe {
    basemul_accumulate_k3_asm(acc, a, b);
  }
}

#[cfg(all(test, target_os = "linux"))]
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

#[cfg(all(test, target_os = "linux"))]
pub(super) unsafe fn test_basemul_accumulate_chunk_asm(
  acc: &mut Poly,
  a: &[u16; SAMPLE_NTT_ACC_CHUNK_COEFFS],
  b: &Poly,
  coeff_offset: usize,
) {
  debug_assert_eq!(coeff_offset % SAMPLE_NTT_ACC_CHUNK_COEFFS, 0);
  debug_assert!(coeff_offset.strict_add(SAMPLE_NTT_ACC_CHUNK_COEFFS) <= acc.len());
  let gamma_offset = coeff_offset / 2;

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
