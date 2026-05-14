//! x86_64 Linux X25519 assembly backend.
//!
//! The embedded routines are adapted from the s2n-bignum x86_64 Curve25519
//! byte backends. This module owns the ABI boundary and keeps the public
//! rscrypto API semantics in `x25519.rs`.

#![allow(unsafe_code)]

use core::arch::global_asm;

use super::POINT_LENGTH;
use crate::platform::{self, caps::x86};

global_asm!(include_str!("asm/rscrypto_x25519_x86_64_linux.s"), options(att_syntax));

unsafe extern "C" {
  fn rscrypto_curve25519_x25519_byte(out: *mut u8, scalar: *const u8, point: *const u8);
  fn rscrypto_curve25519_x25519_byte_alt(out: *mut u8, scalar: *const u8, point: *const u8);
  fn rscrypto_curve25519_x25519base_byte(out: *mut u8, scalar: *const u8);
  fn rscrypto_curve25519_x25519base_byte_alt(out: *mut u8, scalar: *const u8);
}

#[inline]
fn has_bmi2_adx() -> bool {
  platform::caps().has(x86::BMI2.union(x86::ADX))
}

#[inline]
pub(super) fn x25519(scalar: &[u8; POINT_LENGTH], point: &[u8; POINT_LENGTH]) -> [u8; POINT_LENGTH] {
  let mut out = [0u8; POINT_LENGTH];
  if has_bmi2_adx() {
    // SAFETY: X25519 BMI2/ADX byte backend call because:
    // 1. This module is compiled only for Linux x86_64, matching the embedded System V assembly target.
    // 2. `out`, `scalar`, and `point` are fixed 32-byte arrays and valid for the ABI.
    // 3. Runtime capabilities prove BMI2 and ADX before selecting this backend.
    // 4. The assembly clamps the scalar and masks the point top bit per RFC 7748.
    // 5. The routine does not implement low-order all-zero rejection; the Rust caller checks that.
    unsafe { rscrypto_curve25519_x25519_byte(out.as_mut_ptr(), scalar.as_ptr(), point.as_ptr()) };
  } else {
    // SAFETY: X25519 generic byte backend call because:
    // 1. This module is compiled only for Linux x86_64, matching the embedded System V assembly target.
    // 2. `out`, `scalar`, and `point` are fixed 32-byte arrays and valid for the ABI.
    // 3. The `_alt` routine is the baseline x86_64 backend and does not require BMI2 or ADX.
    // 4. The assembly clamps the scalar and masks the point top bit per RFC 7748.
    // 5. The routine does not implement low-order all-zero rejection; the Rust caller checks that.
    unsafe { rscrypto_curve25519_x25519_byte_alt(out.as_mut_ptr(), scalar.as_ptr(), point.as_ptr()) };
  }
  out
}

#[inline]
pub(super) fn x25519_base(scalar: &[u8; POINT_LENGTH]) -> [u8; POINT_LENGTH] {
  let mut out = [0u8; POINT_LENGTH];
  if has_bmi2_adx() {
    // SAFETY: X25519 BMI2/ADX fixed-base byte backend call because:
    // 1. This module is compiled only for Linux x86_64, matching the embedded System V assembly target.
    // 2. `out` and `scalar` are fixed 32-byte arrays and valid for the ABI.
    // 3. Runtime capabilities prove BMI2 and ADX before selecting this backend.
    // 4. The assembly applies the RFC 7748 scalar clamping internally.
    unsafe { rscrypto_curve25519_x25519base_byte(out.as_mut_ptr(), scalar.as_ptr()) };
  } else {
    // SAFETY: X25519 generic fixed-base byte backend call because:
    // 1. This module is compiled only for Linux x86_64, matching the embedded System V assembly target.
    // 2. `out` and `scalar` are fixed 32-byte arrays and valid for the ABI.
    // 3. The `_alt` routine is the baseline x86_64 backend and does not require BMI2 or ADX.
    // 4. The assembly applies the RFC 7748 scalar clamping internally.
    unsafe { rscrypto_curve25519_x25519base_byte_alt(out.as_mut_ptr(), scalar.as_ptr()) };
  }
  out
}
