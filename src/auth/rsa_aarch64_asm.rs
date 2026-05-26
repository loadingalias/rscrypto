//! macOS AArch64 RSA arithmetic assembly backend.
//!
//! This module owns the ABI boundary. `rsa.rs` owns all RSA validation,
//! representative range checks, dispatch, and fallback semantics.

#![allow(unsafe_code)]

use core::arch::global_asm;

global_asm!(include_str!("asm/rscrypto_rsa_aarch64_apple_darwin.s"));
global_asm!(include_str!("asm/rscrypto_rsa_bignum_mont_apple.s"));

unsafe extern "C" {
  fn rscrypto_rsa_mont_reduce_cios_32_aarch64_apple_darwin(
    out: *mut u64,
    value: *const u64,
    modulus: *const u64,
    n0: u64,
    t: *mut u64,
  );

  fn rscrypto_rsa_mont_reduce_cios_words_aarch64_apple_darwin(
    out: *mut u64,
    value: *const u64,
    modulus: *const u64,
    n0: u64,
    t: *mut u64,
    words: usize,
  );

  fn rscrypto_rsa_bn_mul_mont_words_apple(
    out: *mut u64,
    a: *const u64,
    b: *const u64,
    modulus: *const u64,
    n0: *const u64,
    words: usize,
  );
}

#[inline]
pub(super) fn supports_bignum_mont_words(words: usize) -> bool {
  matches!(words, 32 | 48 | 64)
}

#[inline]
pub(super) fn bignum_mont_scratch_words(words: usize) -> usize {
  words.strict_mul(2).strict_add(2)
}

#[inline]
pub(super) fn mont_mul_cios_words(
  out: &mut [u64],
  a: &[u64],
  b: &[u64],
  modulus: &[u64],
  n0: u64,
  words: usize,
  t: &mut [u64],
) {
  debug_assert!(supports_bignum_mont_words(words));
  debug_assert_eq!(out.len(), words);
  debug_assert_eq!(a.len(), words);
  debug_assert_eq!(b.len(), words);
  debug_assert_eq!(modulus.len(), words);
  debug_assert!(t.len() >= bignum_mont_scratch_words(words));

  // SAFETY: RSA Montgomery assembly call because:
  // 1. This module is compiled only for macOS AArch64 and embeds the matching Darwin symbol.
  // 2. The caller checks all slices have one of the supported public RSA limb widths: 32, 48, or 64
  //    `u64` limbs. The `t` capacity check preserves the shared RSA helper contract; this kernel uses
  //    stack scratch internally.
  // 3. All pointers are derived from live Rust slices and are valid for the assembly's fixed
  //    read/write ranges. The assembly does not retain pointers after returning.
  // 4. `out` does not alias `a`, `b`, or `modulus` in current non-in-place callers.
  unsafe {
    rscrypto_rsa_bn_mul_mont_words_apple(out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), modulus.as_ptr(), &n0, words)
  };
}

#[inline]
pub(super) fn mont_square_cios_words_in_place(
  value: &mut [u64],
  modulus: &[u64],
  n0: u64,
  words: usize,
  t: &mut [u64],
) {
  debug_assert!(supports_bignum_mont_words(words));
  debug_assert_eq!(value.len(), words);
  debug_assert_eq!(modulus.len(), words);
  debug_assert!(t.len() >= bignum_mont_scratch_words(words));

  // SAFETY: RSA in-place Montgomery square assembly call because:
  // 1. This module is compiled only for macOS AArch64 and embeds the matching Darwin symbol.
  // 2. The caller checks `value` and `modulus` are 32, 48, or 64 `u64` limbs. The `t` capacity check
  //    preserves the shared RSA helper contract; this kernel uses stack scratch internally.
  // 3. `value.as_mut_ptr()` is intentionally passed as `out`, `a`, and `b`; the assembly consumes all
  //    input limbs before writing `out`.
  // 4. All pointers are derived from live Rust slices and the assembly does not retain them.
  unsafe {
    rscrypto_rsa_bn_mul_mont_words_apple(
      value.as_mut_ptr(),
      value.as_ptr(),
      value.as_ptr(),
      modulus.as_ptr(),
      &n0,
      words,
    )
  };
}

#[inline]
pub(super) fn mont_mul_cios_words_in_place_left(
  left: &mut [u64],
  right: &[u64],
  modulus: &[u64],
  n0: u64,
  words: usize,
  t: &mut [u64],
) {
  debug_assert!(supports_bignum_mont_words(words));
  debug_assert_eq!(left.len(), words);
  debug_assert_eq!(right.len(), words);
  debug_assert_eq!(modulus.len(), words);
  debug_assert!(t.len() >= bignum_mont_scratch_words(words));

  // SAFETY: RSA in-place Montgomery multiply assembly call because:
  // 1. This module is compiled only for macOS AArch64 and embeds the matching Darwin symbol.
  // 2. The caller checks `left`, `right`, and `modulus` are 32, 48, or 64 `u64` limbs. The `t`
  //    capacity check preserves the shared RSA helper contract; this kernel uses stack scratch
  //    internally.
  // 3. `left.as_mut_ptr()` is intentionally passed as both `out` and `a`; the assembly consumes all
  //    input limbs before writing `out`.
  // 4. All pointers are derived from live Rust slices and the assembly does not retain them.
  unsafe {
    rscrypto_rsa_bn_mul_mont_words_apple(
      left.as_mut_ptr(),
      left.as_ptr(),
      right.as_ptr(),
      modulus.as_ptr(),
      &n0,
      words,
    )
  };
}

#[inline]
pub(super) fn mont_reduce_cios_words(
  out: &mut [u64],
  value: &[u64],
  modulus: &[u64],
  n0: u64,
  words: usize,
  t: &mut [u64],
) {
  debug_assert!(matches!(words, 48 | 64));
  debug_assert_eq!(out.len(), words);
  debug_assert_eq!(value.len(), words);
  debug_assert_eq!(modulus.len(), words);
  debug_assert!(t.len() >= bignum_mont_scratch_words(words));

  // SAFETY: RSA Montgomery REDC assembly call because:
  // 1. This module is compiled only for macOS AArch64 and embeds the matching Darwin symbol.
  // 2. The caller checks `out`, `value`, and `modulus` are 48 or 64 `u64` limbs and `t` has `2 *
  //    words + 2` scratch limbs.
  // 3. The assembly writes only caller-provided scratch and performs final conditional subtraction
  //    with mask selection, not data-dependent branches.
  // 4. All pointers are derived from live Rust slices and the assembly does not retain them.
  unsafe {
    rscrypto_rsa_mont_reduce_cios_words_aarch64_apple_darwin(
      out.as_mut_ptr(),
      value.as_ptr(),
      modulus.as_ptr(),
      n0,
      t.as_mut_ptr(),
      words,
    )
  };
}

#[inline]
pub(super) fn public_e65537_mont_words(
  out: &mut [u64],
  input: &[u64],
  r2: &[u64],
  acc: &mut [u64],
  modulus: &[u64],
  n0: u64,
  words: usize,
  t: &mut [u64],
) {
  debug_assert!(supports_bignum_mont_words(words));
  debug_assert_eq!(out.len(), words);
  debug_assert_eq!(input.len(), words);
  debug_assert_eq!(r2.len(), words);
  debug_assert_eq!(acc.len(), words);
  debug_assert_eq!(modulus.len(), words);
  debug_assert!(t.len() >= bignum_mont_scratch_words(words));

  // SAFETY: RSA public e=65537 Montgomery chain because:
  // 1. This module is compiled only for macOS AArch64 and embeds the matching Darwin symbols.
  // 2. The caller checks all slices have one of the supported public RSA limb widths: 32, 48, or 64
  //    `u64` limbs, with `t` sized to the shared scratch contract.
  // 3. `out` and `acc` are distinct caller-owned scratch limbs. `out` first preserves the Montgomery
  //    base, while `acc` is squared in place and multiplied by that preserved base.
  // 4. All pointers are derived from live Rust slices and the assembly does not retain them.
  unsafe {
    rscrypto_rsa_bn_mul_mont_words_apple(
      out.as_mut_ptr(),
      input.as_ptr(),
      r2.as_ptr(),
      modulus.as_ptr(),
      &n0,
      words,
    );
    acc.copy_from_slice(out);
    for _ in 0..16 {
      rscrypto_rsa_bn_mul_mont_words_apple(
        acc.as_mut_ptr(),
        acc.as_ptr(),
        acc.as_ptr(),
        modulus.as_ptr(),
        &n0,
        words,
      );
    }
    rscrypto_rsa_bn_mul_mont_words_apple(
      acc.as_mut_ptr(),
      acc.as_ptr(),
      out.as_ptr(),
      modulus.as_ptr(),
      &n0,
      words,
    );

    if words == 32 {
      rscrypto_rsa_mont_reduce_cios_32_aarch64_apple_darwin(
        out.as_mut_ptr(),
        acc.as_ptr(),
        modulus.as_ptr(),
        n0,
        t.as_mut_ptr(),
      );
    } else {
      rscrypto_rsa_mont_reduce_cios_words_aarch64_apple_darwin(
        out.as_mut_ptr(),
        acc.as_ptr(),
        modulus.as_ptr(),
        n0,
        t.as_mut_ptr(),
        words,
      );
    }
  }
}

#[inline]
pub(super) fn mont_reduce_cios_32(out: &mut [u64], value: &[u64], modulus: &[u64], n0: u64, t: &mut [u64]) {
  debug_assert_eq!(out.len(), 32);
  debug_assert_eq!(value.len(), 32);
  debug_assert_eq!(modulus.len(), 32);
  debug_assert!(t.len() >= 66);

  // SAFETY: RSA-2048 Montgomery REDC assembly call because:
  // 1. This module is compiled only for macOS AArch64 and embeds the matching Darwin symbol.
  // 2. The caller checks `out`, `value`, and `modulus` are 32 `u64` limbs and `t` has at least 66
  //    `u64` scratch limbs.
  // 3. All pointers are derived from live Rust slices and are valid for the assembly's fixed
  //    read/write ranges. The assembly does not retain pointers after returning.
  // 4. `t` is dedicated caller-owned scratch. Current callers pass distinct `out` and `value` slices
  //    through the RSA arithmetic scratch layout.
  unsafe {
    rscrypto_rsa_mont_reduce_cios_32_aarch64_apple_darwin(
      out.as_mut_ptr(),
      value.as_ptr(),
      modulus.as_ptr(),
      n0,
      t.as_mut_ptr(),
    )
  };
}
