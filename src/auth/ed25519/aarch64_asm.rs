//! Apple AArch64 Ed25519 assembly backend.
//!
//! The embedded routines are adapted from the s2n-bignum AArch64 Ed25519
//! decode, fixed-base multiplication, and double-scalar multiplication
//! backends. This module owns the ABI boundary; `ed25519.rs` owns
//! public-key/signature validation semantics.

#![allow(unsafe_code)]

use core::arch::global_asm;

use super::constants::{PUBLIC_KEY_LENGTH, SECRET_KEY_LENGTH};

const AFFINE_POINT_LIMBS: usize = 8;
const FIELD_LIMBS: usize = 4;

#[cfg(target_os = "macos")]
global_asm!(include_str!("asm/rscrypto_ed25519_aarch64_apple_darwin.s"));
#[cfg(target_os = "linux")]
global_asm!(include_str!("asm/rscrypto_ed25519_aarch64_unknown_linux.s"));
#[cfg(target_os = "macos")]
global_asm!(include_str!(
  "asm/rscrypto_ed25519_scalarmulbase_aarch64_apple_darwin.s"
));
#[cfg(target_os = "linux")]
global_asm!(include_str!(
  "asm/rscrypto_ed25519_scalarmulbase_aarch64_unknown_linux.s"
));

unsafe extern "C" {
  fn rscrypto_edwards25519_decode_alt(point: *mut u64, encoded: *const u8) -> u64;
  fn rscrypto_edwards25519_scalarmulbase_alt(out: *mut u64, scalar: *const u64);
  fn rscrypto_edwards25519_scalarmuldouble_alt(
    out: *mut u64,
    scalar: *const u64,
    point: *const u64,
    basepoint_scalar: *const u64,
  );
}

/// Compute `[s]B` and return the compressed Ed25519 point.
#[inline]
pub(super) fn basepoint_mul_encoded(s: &[u8; SECRET_KEY_LENGTH]) -> [u8; PUBLIC_KEY_LENGTH] {
  let s_words = words_from_le_bytes(s);
  let mut out = [0u64; AFFINE_POINT_LIMBS];

  // SAFETY: fixed-base scalar multiplication call because:
  // 1. This module is compiled only for supported AArch64 OS ABIs, matching the embedded assembly
  //    target.
  // 2. `out` has space for eight `u64` affine limbs.
  // 3. `s_words` has four little-endian `u64` scalar limbs, matching the s2n-bignum ABI.
  // 4. The scalar may contain secret nonce material. Generated-code timing assurance is
  //    configuration- and release-evidence-bound; see `ct.toml`.
  unsafe { rscrypto_edwards25519_scalarmulbase_alt(out.as_mut_ptr(), s_words.as_ptr()) };

  encode_affine_point(&out)
}

/// Compute `[s]B + [h]A` and return the compressed Ed25519 point.
///
/// The s2n-bignum ABI takes its scalars as four little-endian `u64` limbs and
/// its point as affine `(x, y)` field limbs. Callers pass `h = -H(R,A,M)` for
/// strict Ed25519 verification.
#[inline]
pub(super) fn double_scalar_basepoint_encoded(
  s: &[u8; SECRET_KEY_LENGTH],
  h: &[u8; SECRET_KEY_LENGTH],
  public_key: &[u8; PUBLIC_KEY_LENGTH],
) -> Option<[u8; PUBLIC_KEY_LENGTH]> {
  let s_words = words_from_le_bytes(s);
  let h_words = words_from_le_bytes(h);
  let mut public_point = [0u64; AFFINE_POINT_LIMBS];

  // SAFETY: public-key decode call because:
  // 1. This module is compiled only for supported AArch64 OS ABIs, matching the embedded assembly
  //    target.
  // 2. `public_point` has space for eight `u64` affine limbs and `public_key` is a fixed 32-byte
  //    input.
  // 3. The routine only reads the input bytes and writes the fixed output buffer.
  let decode_failed = unsafe { rscrypto_edwards25519_decode_alt(public_point.as_mut_ptr(), public_key.as_ptr()) };
  if decode_failed != 0 {
    return None;
  }

  let mut out = [0u64; AFFINE_POINT_LIMBS];
  // SAFETY: double-scalar multiplication call because:
  // 1. This module is compiled only for supported AArch64 OS ABIs, matching the embedded assembly
  //    target.
  // 2. `out` has space for eight `u64` affine limbs.
  // 3. `h_words`, `public_point`, and `s_words` match the s2n-bignum ABI: scalar[4], point[8],
  //    basepoint_scalar[4].
  // 4. The assembly routine is variable-time; Ed25519 verification inputs are public.
  unsafe {
    rscrypto_edwards25519_scalarmuldouble_alt(
      out.as_mut_ptr(),
      h_words.as_ptr(),
      public_point.as_ptr(),
      s_words.as_ptr(),
    )
  };

  Some(encode_affine_point(&out))
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
