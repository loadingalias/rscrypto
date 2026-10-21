//! Linux AArch64 RSA arithmetic assembly backend.
//!
//! This module owns the ABI boundary. `rsa.rs` owns all RSA validation,
//! representative range checks, dispatch, and fallback semantics.

use core::arch::global_asm;

global_asm!(include_str!("asm/rscrypto_rsa_aarch64_linux_elf.s"));

unsafe extern "C" {
  fn rscrypto_rsa_mont_reduce_cios_32_aarch64_elf(
    out: *mut u64,
    value: *const u64,
    modulus: *const u64,
    n0: u64,
    t: *mut u64,
  );

  fn rscrypto_rsa_mont_reduce_cios_words_aarch64_elf(
    out: *mut u64,
    value: *const u64,
    modulus: *const u64,
    n0: u64,
    t: *mut u64,
    words: usize,
  );

}

#[inline]
pub(super) fn bignum_mont_scratch_words(words: usize) -> usize {
  words.strict_mul(2).strict_add(2)
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
  debug_assert!(matches!(words, 48 | 64 | 128));
  debug_assert_eq!(out.len(), words);
  debug_assert_eq!(value.len(), words);
  debug_assert_eq!(modulus.len(), words);
  debug_assert!(t.len() >= bignum_mont_scratch_words(words));

  // SAFETY: RSA Montgomery REDC assembly call because:
  // 1. This module is compiled only for Linux AArch64 and embeds the matching ELF symbol.
  // 2. The caller checks `out`, `value`, and `modulus` are 48, 64, or 128 `u64` limbs and `t` has
  //    `2 * words + 2` scratch limbs.
  // 3. The assembly writes only caller-provided scratch and performs final conditional subtraction
  //    with mask selection, not data-dependent branches.
  // 4. All pointers are derived from live Rust slices and the assembly does not retain them.
  unsafe {
    rscrypto_rsa_mont_reduce_cios_words_aarch64_elf(
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
pub(super) fn mont_reduce_cios_32(out: &mut [u64], value: &[u64], modulus: &[u64], n0: u64, t: &mut [u64]) {
  debug_assert_eq!(out.len(), 32);
  debug_assert_eq!(value.len(), 32);
  debug_assert_eq!(modulus.len(), 32);
  debug_assert!(t.len() >= 66);

  // SAFETY: RSA-2048 Montgomery REDC assembly call because:
  // 1. This module is compiled only for Linux AArch64 and embeds the matching ELF symbol.
  // 2. The caller checks `out`, `value`, and `modulus` are 32 `u64` limbs and `t` has at least 66
  //    `u64` scratch limbs.
  // 3. All pointers are derived from live Rust slices and are valid for the assembly's fixed
  //    read/write ranges. The assembly does not retain pointers after returning.
  // 4. `t` is dedicated caller-owned scratch. Current callers pass distinct `out` and `value` slices
  //    through the RSA arithmetic scratch layout.
  unsafe {
    rscrypto_rsa_mont_reduce_cios_32_aarch64_elf(out.as_mut_ptr(), value.as_ptr(), modulus.as_ptr(), n0, t.as_mut_ptr())
  };
}
