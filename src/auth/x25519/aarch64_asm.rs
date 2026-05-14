//! AArch64 X25519 assembly backend.
//!
//! The embedded routines are adapted from the s2n-bignum AArch64 Curve25519
//! byte backends. This module owns the ABI boundary and keeps the public
//! rscrypto API semantics in `x25519.rs`.

#![allow(unsafe_code)]

use core::arch::global_asm;

use super::POINT_LENGTH;

global_asm!(include_str!("asm/rscrypto_x25519_aarch64_apple_darwin.s"));

unsafe extern "C" {
  fn rscrypto_curve25519_x25519_byte_alt(out: *mut u8, scalar: *const u8, point: *const u8);
  fn rscrypto_curve25519_x25519base_byte_alt(out: *mut u8, scalar: *const u8);
}

#[inline]
pub(super) fn x25519(scalar: &[u8; POINT_LENGTH], point: &[u8; POINT_LENGTH]) -> [u8; POINT_LENGTH] {
  let mut out = [0u8; POINT_LENGTH];
  // SAFETY: X25519 byte backend call because:
  // 1. This module is compiled only for macOS aarch64, matching the embedded assembly target.
  // 2. `out`, `scalar`, and `point` are fixed 32-byte arrays and valid for the assembly ABI.
  // 3. The assembly clamps the scalar and masks the point top bit per RFC 7748.
  // 4. The routine does not implement low-order all-zero rejection; the Rust caller checks that.
  unsafe { rscrypto_curve25519_x25519_byte_alt(out.as_mut_ptr(), scalar.as_ptr(), point.as_ptr()) };
  out
}

#[inline]
pub(super) fn x25519_base(scalar: &[u8; POINT_LENGTH]) -> [u8; POINT_LENGTH] {
  let mut out = [0u8; POINT_LENGTH];
  // SAFETY: X25519 fixed-base byte backend call because:
  // 1. This module is compiled only for macOS aarch64, matching the embedded assembly target.
  // 2. `out` and `scalar` are fixed 32-byte arrays and valid for the assembly ABI.
  // 3. The assembly applies the RFC 7748 scalar clamping internally.
  unsafe { rscrypto_curve25519_x25519base_byte_alt(out.as_mut_ptr(), scalar.as_ptr()) };
  out
}
