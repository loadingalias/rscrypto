//! AArch64 assembly backends for ECDSA curve operations.
//!
//! The embedded routines are adapted from s2n-bignum AArch64 P-256/P-384
//! Montgomery-Jacobian scalar multiplication backends. This module owns the
//! ABI boundary; `ecdsa.rs` owns scalar validation, blinding, affine
//! conversion, and public API semantics.

#![allow(unsafe_code)]

use core::arch::global_asm;

#[path = "ecdsa_aarch64_tables.rs"]
mod ecdsa_aarch64_tables;

#[cfg(target_os = "macos")]
global_asm!(include_str!(
  "asm/rscrypto_p256_scalarmulbase_alt_aarch64_apple_darwin.s"
));
#[cfg(target_os = "macos")]
global_asm!(include_str!("asm/rscrypto_bignum_mod_n256_aarch64_apple_darwin.s"));
#[cfg(target_os = "macos")]
global_asm!(include_str!("asm/rscrypto_bignum_mod_n384_aarch64_apple_darwin.s"));
#[cfg(target_os = "macos")]
global_asm!(include_str!("asm/rscrypto_bignum_montmul_p384_aarch64_apple_darwin.s"));
#[cfg(target_os = "macos")]
global_asm!(include_str!("asm/rscrypto_bignum_montsqr_p384_aarch64_apple_darwin.s"));
#[cfg(target_os = "macos")]
global_asm!(include_str!("asm/rscrypto_bignum_montinv_p384_aarch64_apple_darwin.s"));
#[cfg(target_os = "macos")]
global_asm!(include_str!("asm/rscrypto_bignum_modinv_aarch64_apple_darwin.s"));
#[cfg(target_os = "macos")]
global_asm!(include_str!("asm/rscrypto_p384_montjdouble_alt_aarch64_apple_darwin.s"));
#[cfg(target_os = "macos")]
global_asm!(include_str!("asm/rscrypto_p384_montjmixadd_alt_aarch64_apple_darwin.s"));
#[cfg(target_os = "linux")]
global_asm!(include_str!(
  "asm/rscrypto_p256_scalarmulbase_alt_aarch64_unknown_linux.s"
));
#[cfg(target_os = "linux")]
global_asm!(include_str!("asm/rscrypto_bignum_mod_n256_aarch64_unknown_linux.s"));
#[cfg(target_os = "linux")]
global_asm!(include_str!("asm/rscrypto_bignum_mod_n384_aarch64_unknown_linux.s"));
#[cfg(target_os = "linux")]
global_asm!(include_str!("asm/rscrypto_bignum_montmul_p384_aarch64_unknown_linux.s"));
#[cfg(target_os = "linux")]
global_asm!(include_str!("asm/rscrypto_bignum_montsqr_p384_aarch64_unknown_linux.s"));
#[cfg(target_os = "linux")]
global_asm!(include_str!("asm/rscrypto_bignum_montinv_p384_aarch64_unknown_linux.s"));
#[cfg(target_os = "linux")]
global_asm!(include_str!("asm/rscrypto_bignum_modinv_aarch64_unknown_linux.s"));
#[cfg(target_os = "linux")]
global_asm!(include_str!(
  "asm/rscrypto_p384_montjdouble_alt_aarch64_unknown_linux.s"
));
#[cfg(target_os = "linux")]
global_asm!(include_str!(
  "asm/rscrypto_p384_montjmixadd_alt_aarch64_unknown_linux.s"
));

unsafe extern "C" {
  fn rscrypto_p256_scalarmulbase_alt(out: *mut u64, scalar: *const u64, blocksize: u64, table: *const u64);
  fn rscrypto_bignum_mod_n256(out: *mut u64, len: u64, input: *const u64);
  fn rscrypto_bignum_mod_n384(out: *mut u64, len: u64, input: *const u64);
  fn rscrypto_bignum_montmul_p384(out: *mut u64, lhs: *const u64, rhs: *const u64);
  fn rscrypto_bignum_montsqr_p384(out: *mut u64, value: *const u64);
  fn rscrypto_bignum_montinv_p384(out: *mut u64, value: *const u64);
  fn rscrypto_bignum_modinv(k: u64, out: *mut u64, value: *const u64, modulus: *const u64, tmp: *mut u64);
  fn rscrypto_p384_montjdouble_alt(out: *mut u64, point: *const u64);
  fn rscrypto_p384_montjmixadd_alt(out: *mut u64, lhs: *const u64, rhs: *const u64);
}

#[inline]
pub(super) fn p256_scalarmulbase_generator(scalar: &[u64; 4]) -> [u64; 8] {
  let mut out = [0u64; 8];
  // SAFETY: P-256 fixed-base scalar multiplication call because:
  // 1. This module is compiled only for supported AArch64, matching the embedded assembly ABI.
  // 2. `out` and `scalar` are fixed-size arrays with the exact limb counts required by s2n-bignum.
  // 3. The assembly routine is constant-time with respect to the scalar; ECDSA signing uses secret
  //    nonce material.
  // 4. The table is generated for `P256_AARCH64_BASEPOINT_BLOCKSIZE` and contains every entry the
  //    s2n-bignum fixed-base routine reads for that block size.
  unsafe {
    rscrypto_p256_scalarmulbase_alt(
      out.as_mut_ptr(),
      scalar.as_ptr(),
      ecdsa_aarch64_tables::P256_AARCH64_BASEPOINT_BLOCKSIZE,
      ecdsa_aarch64_tables::P256_AARCH64_BASEPOINT_TABLE.as_ptr(),
    )
  };
  out
}

#[inline]
pub(super) fn p256_reduce_order_64(bytes: &[u8; 64]) -> [u64; 4] {
  let input = words_from_be_bytes_reversed::<8, 64>(bytes);
  let mut out = [0u64; 4];
  // SAFETY: P-256 order reduction call because:
  // 1. This module is compiled only for supported AArch64, matching the embedded assembly ABI.
  // 2. `out` has four `u64` limbs, which is the exact P-256 group-order output size.
  // 3. `input` has eight `u64` limbs and `len` is 8, so the assembly reads exactly the provided
  //    input.
  // 4. The routine runs a fixed-size reduction for public length 8; the reduced value may be secret.
  unsafe { rscrypto_bignum_mod_n256(out.as_mut_ptr(), 8, input.as_ptr()) };
  out
}

#[inline]
pub(super) fn p384_reduce_order_96(bytes: &[u8; 96]) -> [u64; 6] {
  let input = words_from_be_bytes_reversed::<12, 96>(bytes);
  let mut out = [0u64; 6];
  // SAFETY: P-384 order reduction call because:
  // 1. This module is compiled only for supported AArch64, matching the embedded assembly ABI.
  // 2. `out` has six `u64` limbs, which is the exact P-384 group-order output size.
  // 3. `input` has twelve `u64` limbs and `len` is 12, so the assembly reads exactly the provided
  //    input.
  // 4. The routine runs a fixed-size reduction for public length 12; the reduced value may be secret.
  unsafe { rscrypto_bignum_mod_n384(out.as_mut_ptr(), 12, input.as_ptr()) };
  out
}

#[inline]
pub(super) fn p384_field_mul(lhs: &[u64; 6], rhs: &[u64; 6]) -> [u64; 6] {
  let mut out = [0u64; 6];
  // SAFETY: P-384 field Montgomery multiplication call because:
  // 1. This module is compiled only for supported AArch64, matching the embedded assembly ABI.
  // 2. `out`, `lhs`, and `rhs` are six-limb arrays, which is the exact P-384 field size.
  // 3. Callers route only P-384 field elements in Montgomery form through this wrapper.
  // 4. The routine is constant-time with respect to the field-element values.
  unsafe { rscrypto_bignum_montmul_p384(out.as_mut_ptr(), lhs.as_ptr(), rhs.as_ptr()) };
  out
}

#[inline]
pub(super) fn p384_field_square(value: &[u64; 6]) -> [u64; 6] {
  let mut out = [0u64; 6];
  // SAFETY: P-384 field Montgomery square call because:
  // 1. This module is compiled only for supported AArch64, matching the embedded assembly ABI.
  // 2. `out` and `value` are six-limb arrays, which is the exact P-384 field size.
  // 3. Callers route only P-384 field elements in Montgomery form through this wrapper.
  // 4. The routine is constant-time with respect to the field-element value.
  unsafe { rscrypto_bignum_montsqr_p384(out.as_mut_ptr(), value.as_ptr()) };
  out
}

#[inline]
pub(super) fn p384_field_inverse(value: &[u64; 6]) -> [u64; 6] {
  let mut out = [0u64; 6];
  // SAFETY: P-384 field Montgomery inverse call because:
  // 1. This module is compiled only for supported AArch64, matching the embedded assembly ABI.
  // 2. `out` and `value` are six-limb arrays, which is the exact P-384 field size.
  // 3. Callers route only P-384 field elements in Montgomery form through this wrapper.
  // 4. The routine has fixed public iteration structure and returns zero for zero input.
  unsafe { rscrypto_bignum_montinv_p384(out.as_mut_ptr(), value.as_ptr()) };
  out
}

#[inline]
pub(super) fn scalar_inverse<const L: usize>(value: &[u64; L], modulus: &[u64; L]) -> [u64; L] {
  let mut out = [0u64; L];
  let mut tmp = [0u64; 18];
  debug_assert!(L == 4 || L == 6);
  // SAFETY: scalar-order inverse call because:
  // 1. This module is compiled only for supported AArch64, matching the embedded assembly ABI.
  // 2. The wrapper is called only with L = 4 or L = 6, and `tmp` has 3 * 6 limbs, enough for both
  //    sizes.
  // 3. `value`, `modulus`, and `out` have the same fixed limb count passed as `k`.
  // 4. Callers use odd prime ECDSA group orders and non-zero nonce scalars.
  unsafe {
    rscrypto_bignum_modinv(
      L as u64,
      out.as_mut_ptr(),
      value.as_ptr(),
      modulus.as_ptr(),
      tmp.as_mut_ptr(),
    )
  };
  out
}

#[inline]
pub(super) fn p384_point_double(point: &[u64; 18]) -> [u64; 18] {
  let mut out = [0u64; 18];
  // SAFETY: P-384 Jacobian doubling call because:
  // 1. This module is compiled only for supported AArch64, matching the embedded assembly ABI.
  // 2. `out` and `point` are 18-limb arrays laid out as x/y/z P-384 Montgomery coordinates.
  // 3. Callers preserve the Rust-level infinity flag and select the infinity result when required.
  // 4. The routine is constant-time with respect to the point coordinate values.
  unsafe { rscrypto_p384_montjdouble_alt(out.as_mut_ptr(), point.as_ptr()) };
  out
}

#[inline]
pub(super) fn p384_point_mixadd(lhs: &[u64; 18], rhs: &[u64; 12]) -> [u64; 18] {
  let mut out = [0u64; 18];
  // SAFETY: P-384 mixed-addition call because:
  // 1. This module is compiled only for supported AArch64, matching the embedded assembly ABI.
  // 2. `out` and `lhs` are 18-limb Jacobian arrays; `rhs` is a 12-limb affine x/y array.
  // 3. Callers preserve the Rust-level infinity and zero-digit handling around this raw point
  //    operation.
  // 4. The routine is constant-time with respect to the point coordinate values.
  unsafe { rscrypto_p384_montjmixadd_alt(out.as_mut_ptr(), lhs.as_ptr(), rhs.as_ptr()) };
  out
}

#[inline]
fn words_from_be_bytes_reversed<const WORDS: usize, const BYTES: usize>(bytes: &[u8; BYTES]) -> [u64; WORDS] {
  let mut out = [0u64; WORDS];
  for (word, chunk) in out.iter_mut().zip(bytes.rchunks_exact(8)) {
    let mut limb = [0u8; 8];
    limb.copy_from_slice(chunk);
    *word = u64::from_be_bytes(limb);
  }
  out
}
