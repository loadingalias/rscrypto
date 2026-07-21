//! Linux x86-64 Ed25519 signing assembly backend.
//!
//! The embedded routine is adapted from the s2n-bignum x86-64 Ed25519
//! fixed-base multiplication backend. This module owns the ABI boundary;
//! `ed25519.rs` owns signing semantics.

#![allow(unsafe_code)]

use core::arch::global_asm;

use super::{
  constants::{PUBLIC_KEY_LENGTH, SECRET_KEY_LENGTH},
  field, point,
};
use crate::platform::{self, caps::x86};

const AFFINE_POINT_LIMBS: usize = 8;
const FIELD_LIMBS: usize = 4;

global_asm!(
  include_str!("asm/rscrypto_ed25519_scalarmulbase_x86_64_unknown_linux.s"),
  options(att_syntax)
);
global_asm!(
  include_str!("asm/rscrypto_ed25519_scalarmulbase_alt_x86_64_unknown_linux.s"),
  options(att_syntax)
);

unsafe extern "C" {
  fn rscrypto_edwards25519_scalarmulbase(out: *mut u64, scalar: *const u64);
  fn rscrypto_edwards25519_scalarmulbase_alt(out: *mut u64, scalar: *const u64);
}

#[inline]
fn has_bmi2_adx() -> bool {
  platform::caps().has(x86::BMI2.union(x86::ADX))
}

/// Compute `[s]B` and return the compressed Ed25519 point.
#[inline]
pub(super) fn basepoint_mul_encoded(s: &[u8; SECRET_KEY_LENGTH]) -> [u8; PUBLIC_KEY_LENGTH] {
  let out = basepoint_mul_affine(s);
  encode_affine_point(&out)
}

/// Compute `[s]B` and return both compressed bytes and cached affine point.
#[inline]
pub(super) fn basepoint_mul_public(s: &[u8; SECRET_KEY_LENGTH]) -> ([u8; PUBLIC_KEY_LENGTH], point::ExtendedPoint) {
  let out = basepoint_mul_affine(s);
  (encode_affine_point(&out), extended_point_from_affine(&out))
}

#[inline]
fn basepoint_mul_affine(s: &[u8; SECRET_KEY_LENGTH]) -> [u64; AFFINE_POINT_LIMBS] {
  let s_words = words_from_le_bytes(s);
  let mut out = [0u64; AFFINE_POINT_LIMBS];

  if has_bmi2_adx() {
    // SAFETY: BMI2/ADX fixed-base scalar multiplication call because:
    // 1. This module is compiled only for Linux x86-64 System V, matching the embedded assembly ABI.
    // 2. `out` has space for eight `u64` affine limbs.
    // 3. `s_words` has four little-endian `u64` scalar limbs, matching the s2n-bignum ABI.
    // 4. Runtime capabilities prove BMI2 and ADX before selecting this backend.
    // 5. The scalar may contain secret nonce material. Generated-code timing assurance is
    //    configuration- and release-evidence-bound; see `ct.toml`.
    unsafe { rscrypto_edwards25519_scalarmulbase(out.as_mut_ptr(), s_words.as_ptr()) };
  } else {
    // SAFETY: baseline fixed-base scalar multiplication call because:
    // 1. This module is compiled only for Linux x86-64 System V, matching the embedded assembly ABI.
    // 2. `out` has space for eight `u64` affine limbs.
    // 3. `s_words` has four little-endian `u64` scalar limbs, matching the s2n-bignum ABI.
    // 4. The `_alt` routine is the baseline x86-64 backend and does not require BMI2 or ADX.
    // 5. The scalar may contain secret nonce material. Generated-code timing assurance is
    //    configuration- and release-evidence-bound; see `ct.toml`.
    unsafe { rscrypto_edwards25519_scalarmulbase_alt(out.as_mut_ptr(), s_words.as_ptr()) };
  }

  out
}

#[inline]
fn words_from_le_bytes(bytes: &[u8; SECRET_KEY_LENGTH]) -> [u64; FIELD_LIMBS] {
  let mut words = [0u64; FIELD_LIMBS];
  for (word, chunk) in words.iter_mut().zip(bytes.chunks_exact(8)) {
    let mut limb = [0u8; 8];
    limb.copy_from_slice(chunk);
    *word = u64::from_le_bytes(limb);
  }
  words
}

#[inline]
fn encode_affine_point(point: &[u64; AFFINE_POINT_LIMBS]) -> [u8; PUBLIC_KEY_LENGTH] {
  let mut encoded = [0u8; PUBLIC_KEY_LENGTH];

  for (dst, word) in encoded
    .chunks_exact_mut(8)
    .zip(point[FIELD_LIMBS..AFFINE_POINT_LIMBS].iter().copied())
  {
    dst.copy_from_slice(&word.to_le_bytes());
  }

  encoded[PUBLIC_KEY_LENGTH - 1] &= 0x7f;
  encoded[PUBLIC_KEY_LENGTH - 1] |= ((point[0] & 1) as u8) << 7;
  encoded
}

#[inline]
fn extended_point_from_affine(words: &[u64; AFFINE_POINT_LIMBS]) -> point::ExtendedPoint {
  let x = field_element_from_words(&words[..FIELD_LIMBS]);
  let y = field_element_from_words(&words[FIELD_LIMBS..AFFINE_POINT_LIMBS]);
  point::ExtendedPoint::from_affine(x, y)
}

#[inline]
fn field_element_from_words(words: &[u64]) -> field::FieldElement {
  let mut bytes = [0u8; PUBLIC_KEY_LENGTH];
  for (dst, word) in bytes.chunks_exact_mut(8).zip(words.iter().copied()) {
    dst.copy_from_slice(&word.to_le_bytes());
  }
  field::FieldElement::from_bytes(&bytes).expect("Ed25519 assembly must return canonical affine coordinates")
}
