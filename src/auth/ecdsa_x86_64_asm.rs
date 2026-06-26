//! Linux x86-64 assembly backends for ECDSA curve operations.
//!
//! The embedded routines are adapted from s2n-bignum x86-64 P-256 fixed-base
//! and P-384 bignum backends. This module owns the ABI boundary; `ecdsa.rs`
//! owns scalar validation, blinding, affine conversion, and public API
//! semantics.

#![allow(unsafe_code)]

use core::arch::global_asm;

use crate::platform::{self, caps::x86};

#[path = "ecdsa_aarch64_tables.rs"]
mod ecdsa_x86_64_tables;

global_asm!(
  include_str!("asm/rscrypto_p256_scalarmulbase_x86_64_unknown_linux.S"),
  options(att_syntax)
);
global_asm!(
  include_str!("asm/rscrypto_p256_scalarmulbase_alt_x86_64_unknown_linux.S"),
  options(att_syntax)
);
global_asm!(
  include_str!("asm/rscrypto_bignum_mod_n256_x86_64_unknown_linux.S"),
  options(att_syntax)
);
global_asm!(
  include_str!("asm/rscrypto_bignum_mod_n384_x86_64_unknown_linux.S"),
  options(att_syntax)
);
global_asm!(
  include_str!("asm/rscrypto_bignum_modinv_x86_64_unknown_linux.S"),
  options(att_syntax)
);
global_asm!(
  include_str!("asm/rscrypto_bignum_montmul_p384_x86_64_unknown_linux.S"),
  options(att_syntax)
);
global_asm!(
  include_str!("asm/rscrypto_bignum_montmul_p384_alt_x86_64_unknown_linux.S"),
  options(att_syntax)
);
global_asm!(
  include_str!("asm/rscrypto_bignum_montsqr_p384_x86_64_unknown_linux.S"),
  options(att_syntax)
);
global_asm!(
  include_str!("asm/rscrypto_bignum_montsqr_p384_alt_x86_64_unknown_linux.S"),
  options(att_syntax)
);
global_asm!(
  include_str!("asm/rscrypto_bignum_montinv_p384_x86_64_unknown_linux.S"),
  options(att_syntax)
);
unsafe extern "C" {
  fn rscrypto_p256_scalarmulbase(out: *mut u64, scalar: *const u64, blocksize: u64, table: *const u64);
  fn rscrypto_p256_scalarmulbase_alt(out: *mut u64, scalar: *const u64, blocksize: u64, table: *const u64);
  fn rscrypto_bignum_mod_n256(out: *mut u64, len: u64, input: *const u64);
  fn rscrypto_bignum_mod_n384(out: *mut u64, len: u64, input: *const u64);
  fn rscrypto_bignum_modinv(k: u64, out: *mut u64, value: *const u64, modulus: *const u64, tmp: *mut u64);
  fn rscrypto_bignum_montmul_p384(out: *mut u64, lhs: *const u64, rhs: *const u64);
  fn rscrypto_bignum_montmul_p384_alt(out: *mut u64, lhs: *const u64, rhs: *const u64);
  fn rscrypto_bignum_montsqr_p384(out: *mut u64, value: *const u64);
  fn rscrypto_bignum_montsqr_p384_alt(out: *mut u64, value: *const u64);
  fn rscrypto_bignum_montinv_p384(out: *mut u64, value: *const u64);
}

#[inline]
fn has_bmi2_adx() -> bool {
  platform::caps().has(x86::BMI2.union(x86::ADX))
}

#[inline]
pub(super) fn p256_scalarmulbase_generator(scalar: &[u64; 4]) -> [u64; 8] {
  let mut out = [0u64; 8];

  if has_bmi2_adx() {
    // SAFETY: BMI2/ADX P-256 fixed-base scalar multiplication call because:
    // 1. This module is compiled only for Linux x86-64 System V, matching the embedded assembly ABI.
    // 2. `out` and `scalar` are fixed-size arrays with the exact limb counts required by s2n-bignum.
    // 3. Runtime capabilities prove BMI2 and ADX before selecting this backend.
    // 4. The assembly routine is constant-time with respect to the scalar; ECDSA signing uses secret
    //    nonce material.
    // 5. The table is generated for `P256_AARCH64_BASEPOINT_BLOCKSIZE` and contains every entry the
    //    s2n-bignum fixed-base routine reads for that block size.
    unsafe {
      rscrypto_p256_scalarmulbase(
        out.as_mut_ptr(),
        scalar.as_ptr(),
        ecdsa_x86_64_tables::P256_AARCH64_BASEPOINT_BLOCKSIZE,
        ecdsa_x86_64_tables::P256_AARCH64_BASEPOINT_TABLE.as_ptr(),
      )
    };
  } else {
    // SAFETY: baseline P-256 fixed-base scalar multiplication call because:
    // 1. This module is compiled only for Linux x86-64 System V, matching the embedded assembly ABI.
    // 2. `out` and `scalar` are fixed-size arrays with the exact limb counts required by s2n-bignum.
    // 3. The `_alt` routine is the baseline x86-64 backend and does not require BMI2 or ADX.
    // 4. The assembly routine is constant-time with respect to the scalar; ECDSA signing uses secret
    //    nonce material.
    // 5. The table is generated for `P256_AARCH64_BASEPOINT_BLOCKSIZE` and contains every entry the
    //    s2n-bignum fixed-base routine reads for that block size.
    unsafe {
      rscrypto_p256_scalarmulbase_alt(
        out.as_mut_ptr(),
        scalar.as_ptr(),
        ecdsa_x86_64_tables::P256_AARCH64_BASEPOINT_BLOCKSIZE,
        ecdsa_x86_64_tables::P256_AARCH64_BASEPOINT_TABLE.as_ptr(),
      )
    };
  }

  out
}

#[inline]
pub(super) fn p256_reduce_order_64(bytes: &[u8; 64]) -> [u64; 4] {
  let input = words_from_be_bytes_reversed::<8, 64>(bytes);
  let mut out = [0u64; 4];
  // SAFETY: P-256 order reduction call because:
  // 1. This module is compiled only for Linux x86-64 System V, matching the embedded assembly ABI.
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
  // 1. This module is compiled only for Linux x86-64 System V, matching the embedded assembly ABI.
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

  if has_bmi2_adx() {
    // SAFETY: BMI2/ADX P-384 field Montgomery multiplication call because:
    // 1. This module is compiled only for Linux x86-64 System V, matching the embedded assembly ABI.
    // 2. `out`, `lhs`, and `rhs` are six-limb arrays, which is the exact P-384 field size.
    // 3. Runtime capabilities prove BMI2 and ADX before selecting this backend.
    // 4. Callers route only P-384 field elements in Montgomery form through this wrapper.
    // 5. The routine is constant-time with respect to the field-element values.
    unsafe { rscrypto_bignum_montmul_p384(out.as_mut_ptr(), lhs.as_ptr(), rhs.as_ptr()) };
  } else {
    // SAFETY: baseline P-384 field Montgomery multiplication call because:
    // 1. This module is compiled only for Linux x86-64 System V, matching the embedded assembly ABI.
    // 2. `out`, `lhs`, and `rhs` are six-limb arrays, which is the exact P-384 field size.
    // 3. The `_alt` routine is the baseline x86-64 backend and does not require BMI2 or ADX.
    // 4. Callers route only P-384 field elements in Montgomery form through this wrapper.
    // 5. The routine is constant-time with respect to the field-element values.
    unsafe { rscrypto_bignum_montmul_p384_alt(out.as_mut_ptr(), lhs.as_ptr(), rhs.as_ptr()) };
  }

  out
}

#[inline]
pub(super) fn p384_field_square(value: &[u64; 6]) -> [u64; 6] {
  let mut out = [0u64; 6];

  if has_bmi2_adx() {
    // SAFETY: BMI2/ADX P-384 field Montgomery square call because:
    // 1. This module is compiled only for Linux x86-64 System V, matching the embedded assembly ABI.
    // 2. `out` and `value` are six-limb arrays, which is the exact P-384 field size.
    // 3. Runtime capabilities prove BMI2 and ADX before selecting this backend.
    // 4. Callers route only P-384 field elements in Montgomery form through this wrapper.
    // 5. The routine is constant-time with respect to the field-element value.
    unsafe { rscrypto_bignum_montsqr_p384(out.as_mut_ptr(), value.as_ptr()) };
  } else {
    // SAFETY: baseline P-384 field Montgomery square call because:
    // 1. This module is compiled only for Linux x86-64 System V, matching the embedded assembly ABI.
    // 2. `out` and `value` are six-limb arrays, which is the exact P-384 field size.
    // 3. The `_alt` routine is the baseline x86-64 backend and does not require BMI2 or ADX.
    // 4. Callers route only P-384 field elements in Montgomery form through this wrapper.
    // 5. The routine is constant-time with respect to the field-element value.
    unsafe { rscrypto_bignum_montsqr_p384_alt(out.as_mut_ptr(), value.as_ptr()) };
  }

  out
}

#[inline]
pub(super) fn p384_field_inverse(value: &[u64; 6]) -> [u64; 6] {
  let mut out = [0u64; 6];
  // SAFETY: P-384 field Montgomery inverse call because:
  // 1. This module is compiled only for Linux x86-64 System V, matching the embedded assembly ABI.
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
  // 1. This module is compiled only for Linux x86-64 System V, matching the embedded assembly ABI.
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
fn words_from_be_bytes_reversed<const WORDS: usize, const BYTES: usize>(bytes: &[u8; BYTES]) -> [u64; WORDS] {
  let mut out = [0u64; WORDS];
  for (word, chunk) in out.iter_mut().zip(bytes.rchunks_exact(8)) {
    let mut limb = [0u8; 8];
    limb.copy_from_slice(chunk);
    *word = u64::from_be_bytes(limb);
  }
  out
}
