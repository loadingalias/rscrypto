//! Linux x86-64 RSA arithmetic assembly backend.
//!
//! This module owns the ABI boundary. `rsa.rs` owns all RSA validation,
//! representative range checks, dispatch, and fallback semantics.

#![allow(unsafe_code)]

use core::arch::global_asm;

use crate::platform::{caps, caps::x86};

global_asm!(include_str!("asm/rscrypto_rsa_x86_64_elf.S"), options(att_syntax));

unsafe extern "C" {
  fn rscrypto_rsa_bn_mulx4x_mont_x86_64_elf(
    out: *mut u64,
    a: *const u64,
    b: *const u64,
    modulus: *const u64,
    n0: *const u64,
    words: usize,
  ) -> i32;

  fn rscrypto_rsa_bn_sqr8x_mont_x86_64_elf(
    out: *mut u64,
    a: *const u64,
    mulx_adx_capable: u64,
    modulus: *const u64,
    n0: *const u64,
    words: usize,
  ) -> i32;
}

#[inline]
pub(super) fn supports_bignum_mont_words(words: usize) -> bool {
  matches!(words, 32 | 48 | 64) && caps().has(x86::BMI2) && caps().has(x86::ADX)
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

  // SAFETY: RSA Montgomery ADX/BMI2 assembly call because:
  // 1. This module is compiled only for Linux x86-64 and embeds the matching ELF symbol.
  // 2. `supports_bignum_mont_words` checks BMI2 and ADX before this call, matching the kernel's
  //    `mulx/adcx/adox` instruction requirements.
  // 3. The caller checks all slices have one of the supported public RSA limb widths: 32, 48, or 64
  //    `u64` limbs. The `t` capacity check preserves the shared RSA helper contract; this kernel uses
  //    stack scratch internally.
  // 4. All pointers are derived from live Rust slices and are valid for the assembly's fixed
  //    read/write ranges. The assembly does not retain pointers after returning.
  let _ = unsafe {
    rscrypto_rsa_bn_mulx4x_mont_x86_64_elf(out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), modulus.as_ptr(), &n0, words)
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

  // SAFETY: RSA in-place Montgomery square ADX/BMI2 assembly call because:
  // 1. This module is compiled only for Linux x86-64 and embeds the matching ELF symbol.
  // 2. `supports_bignum_mont_words` checks BMI2 and ADX before this call, matching the kernel's
  //    `mulx/adcx/adox` instruction requirements.
  // 3. The caller checks `value` and `modulus` are 32, 48, or 64 `u64` limbs, satisfying the square
  //    kernel's `num >= 8 && num % 8 == 0` precondition.
  // 4. Passing `mulx_adx_capable = 1` is valid because item 2 proves the ADX/BMI2 instruction set.
  //    The kernel writes its result after consuming operands into internal stack scratch, so passing
  //    `value` as both `out` and `a` preserves the in-place helper contract.
  // 5. All pointers are derived from live Rust slices and the assembly does not retain them.
  let _ = unsafe {
    rscrypto_rsa_bn_sqr8x_mont_x86_64_elf(value.as_mut_ptr(), value.as_ptr(), 1, modulus.as_ptr(), &n0, words)
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

  // SAFETY: RSA in-place Montgomery multiply ADX/BMI2 assembly call because:
  // 1. This module is compiled only for Linux x86-64 and embeds the matching ELF symbol.
  // 2. `supports_bignum_mont_words` checks BMI2 and ADX before this call, matching the kernel's
  //    `mulx/adcx/adox` instruction requirements.
  // 3. The caller checks `left`, `right`, and `modulus` are 32, 48, or 64 `u64` limbs. The kernel
  //    writes its result after consuming operands into internal stack scratch, so passing `left` as
  //    both `out` and `a` preserves the in-place helper contract.
  // 4. All pointers are derived from live Rust slices and the assembly does not retain them.
  let _ = unsafe {
    rscrypto_rsa_bn_mulx4x_mont_x86_64_elf(
      left.as_mut_ptr(),
      left.as_ptr(),
      right.as_ptr(),
      modulus.as_ptr(),
      &n0,
      words,
    )
  };
}
