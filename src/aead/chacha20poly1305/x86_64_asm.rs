//! x86_64 Linux integrated ChaCha20-Poly1305 seal/open assembly.
//!
//! The embedded assembly is adapted from the generated AWS-LC/BoringSSL x86_64
//! ChaCha20-Poly1305 assembly. This Rust module owns the ABI boundary and
//! keeps runtime feature dispatch in the parent AEAD implementation.

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
struct OpenDataIn {
  key: [u8; KEY_SIZE],
  counter: u32,
  nonce: [u8; 12],
}

#[repr(C)]
union OpenData {
  input: OpenDataIn,
  out: TagOut,
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct TagOut {
  tag: [u8; 16],
}

const _: () = assert!(mem::size_of::<OpenData>() == 48);
const _: () = assert!(mem::align_of::<OpenData>() == 16);
const _: () = assert!(mem::size_of::<SealData>() == 64);
const _: () = assert!(mem::align_of::<SealData>() == 16);

unsafe extern "C" {
  fn rscrypto_chacha20_poly1305_open_x86_64(
    out_plaintext: *mut u8,
    ciphertext: *const u8,
    plaintext_len: usize,
    aad: *const u8,
    aad_len: usize,
    data: *mut OpenData,
  );

  fn rscrypto_chacha20_poly1305_seal_x86_64(
    out_ciphertext: *mut u8,
    plaintext: *const u8,
    plaintext_len: usize,
    aad: *const u8,
    aad_len: usize,
    data: *mut SealData,
  );
}

/// Seal one nonempty buffer with the integrated x86-64 assembly kernel.
///
/// # Safety
///
/// The caller must ensure that AVX2 and BMI2 are available and that `buffer` is nonempty and within ChaCha20's
/// 2³²-block limit.
#[inline]
pub(super) unsafe fn seal_in_place(key: &[u8; KEY_SIZE], nonce: &[u8; 12], aad: &[u8], buffer: &mut [u8]) -> [u8; 16] {
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
  // 5. `data` is 16-byte aligned and matches the assembly input/output union layout; `from_mut` preserves that
  //    allocation's provenance and exclusive writability for the call, which initializes all 16 tag bytes before
  //    the union output is read.
  // 6. The tag replaces the first 16 copied key bytes; the second half remains initialized key material and is
  //    volatile-zeroed before the stack allocation expires.
  unsafe {
    rscrypto_chacha20_poly1305_seal_x86_64(
      buffer.as_mut_ptr(),
      buffer.as_ptr(),
      buffer.len(),
      aad.as_ptr(),
      aad.len(),
      core::ptr::from_mut(&mut data),
    );
    crate::traits::ct::zeroize(&mut data.input.key[16..]);
    data.out.tag
  }
}

/// Open one nonempty buffer with the integrated x86-64 assembly kernel.
///
/// # Safety
///
/// The caller must ensure that AVX2 and BMI2 are available and that `buffer` is nonempty and within ChaCha20's
/// 2³²-block limit.
#[inline]
pub(super) unsafe fn open_in_place(key: &[u8; KEY_SIZE], nonce: &[u8; 12], aad: &[u8], buffer: &mut [u8]) -> [u8; 16] {
  debug_assert!(!buffer.is_empty());

  let mut data = OpenData {
    input: OpenDataIn {
      key: *key,
      counter: 0,
      nonce: *nonce,
    },
  };

  // SAFETY: integrated x86_64 Linux open call because:
  // 1. The caller gates this function on AVX2 and BMI2 before crossing the assembly ABI boundary.
  // 2. `buffer.as_ptr()` and `buffer.as_mut_ptr()` are valid for `buffer.len()` bytes and may alias;
  //    the assembly routine supports in-place open, matching the AWS-LC ABI.
  // 3. `aad.as_ptr()` is valid for `aad.len()` bytes, including the conventional dangling pointer for
  //    empty slices because the length is zero.
  // 4. `data` is 16-byte aligned and matches the assembly input/output union layout; `from_mut` preserves that
  //    allocation's provenance and exclusive writability for the call, which initializes all 16 tag bytes before
  //    the union output is read.
  // 5. The tag replaces the first 16 copied key bytes; the second half remains initialized key material and is
  //    volatile-zeroed before the stack allocation expires.
  unsafe {
    rscrypto_chacha20_poly1305_open_x86_64(
      buffer.as_mut_ptr(),
      buffer.as_ptr(),
      buffer.len(),
      aad.as_ptr(),
      aad.len(),
      core::ptr::from_mut(&mut data),
    );
    crate::traits::ct::zeroize(&mut data.input.key[16..]);
    data.out.tag
  }
}
