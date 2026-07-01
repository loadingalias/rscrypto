//! x86_64 Linux integrated ChaCha20-Poly1305 seal assembly.
//!
//! The embedded assembly is adapted from the generated AWS-LC/BoringSSL x86_64
//! ChaCha20-Poly1305 assembly. This Rust module owns the ABI boundary and
//! keeps runtime feature dispatch in the parent AEAD implementation.

#![allow(unsafe_code)]

use core::{arch::global_asm, mem};

use super::KEY_SIZE;

global_asm!(
  include_str!("asm/rscrypto_chacha20_poly1305_x86_64_linux.s"),
  options(att_syntax)
);

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct SealDataIn {
  key: [u8; KEY_SIZE],
  counter: u32,
  nonce: [u8; 12],
  extra_ciphertext: *const u8,
  extra_ciphertext_len: usize,
}

#[repr(C)]
union SealData {
  input: SealDataIn,
  out: TagOut,
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct TagOut {
  tag: [u8; 16],
}

const _: () = assert!(mem::size_of::<SealData>() == 64);
const _: () = assert!(mem::align_of::<SealData>() == 16);

unsafe extern "C" {
  fn rscrypto_chacha20_poly1305_seal_x86_64(
    out_ciphertext: *mut u8,
    plaintext: *const u8,
    plaintext_len: usize,
    aad: *const u8,
    aad_len: usize,
    data: *mut SealData,
  );
}

#[inline]
pub(super) fn seal_in_place(key: &[u8; KEY_SIZE], nonce: &[u8; 12], aad: &[u8], buffer: &mut [u8]) -> [u8; 16] {
  debug_assert!(!buffer.is_empty());

  let extra_ciphertext = [0u8; 16];
  let mut data = SealData {
    input: SealDataIn {
      key: *key,
      counter: 0,
      nonce: *nonce,
      extra_ciphertext: extra_ciphertext.as_ptr(),
      extra_ciphertext_len: 0,
    },
  };

  // SAFETY: integrated x86_64 Linux seal call because:
  // 1. The caller gates this function on AVX2 and BMI2 before crossing the assembly ABI boundary.
  // 2. `buffer.as_ptr()` and `buffer.as_mut_ptr()` are valid for `buffer.len()` bytes and may alias;
  //    the assembly routine supports in-place seal, matching the AWS-LC ABI.
  // 3. `aad.as_ptr()` is valid for `aad.len()` bytes, including the conventional dangling pointer for
  //    empty slices because the length is zero.
  // 4. `extra_ciphertext` points at 16 initialized bytes and `extra_ciphertext_len` is zero, matching
  //    the AWS-LC seal ABI for callers without extra trailing ciphertext.
  // 5. `data` is 16-byte aligned and matches the assembly input/output union layout.
  unsafe {
    rscrypto_chacha20_poly1305_seal_x86_64(
      buffer.as_mut_ptr(),
      buffer.as_ptr(),
      buffer.len(),
      aad.as_ptr(),
      aad.len(),
      &mut data,
    );
    data.out.tag
  }
}
