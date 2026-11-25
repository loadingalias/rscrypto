//! aarch64 integrated ChaCha20-Poly1305 seal/open assembly.
//!
//! The embedded assembly is adapted from the generated AWS-LC/BoringSSL aarch64
//! ChaCha20-Poly1305 assembly. This Rust module owns the ABI boundary and
//! keeps runtime feature dispatch in the parent AEAD implementation.

use core::{arch::global_asm, mem};

use super::KEY_SIZE;
use crate::aead::segments::Span;

#[cfg(any(target_os = "linux", target_os = "macos"))]
global_asm!(include_str!("asm/rscrypto_chacha20_poly1305_aarch64.s"), macho = const cfg!(target_os = "macos") as u8);

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
  #[cfg(target_os = "macos")]
  fn rscrypto_chacha20_poly1305_seal_aarch64_apple_darwin(
    out_ciphertext: *mut u8,
    plaintext: *const u8,
    plaintext_len: usize,
    aad: *const u8,
    aad_len: usize,
    data: *mut SealData,
  );

  #[cfg(target_os = "linux")]
  fn rscrypto_chacha20_poly1305_seal_aarch64(
    out_ciphertext: *mut u8,
    plaintext: *const u8,
    plaintext_len: usize,
    aad: *const u8,
    aad_len: usize,
    data: *mut SealData,
  );

  #[cfg(target_os = "macos")]
  fn rscrypto_chacha20_poly1305_open_aarch64_apple_darwin(
    out_plaintext: *mut u8,
    ciphertext: *const u8,
    plaintext_len: usize,
    aad: *const u8,
    aad_len: usize,
    data: *mut OpenData,
  );

  #[cfg(target_os = "linux")]
  fn rscrypto_chacha20_poly1305_open_aarch64(
    out_plaintext: *mut u8,
    ciphertext: *const u8,
    plaintext_len: usize,
    aad: *const u8,
    aad_len: usize,
    data: *mut OpenData,
  );

  #[cfg(target_os = "macos")]
  fn rscrypto_chacha20_poly1305_open_scatter_aarch64_apple_darwin(
    out_spans: *const Span,
    in_spans: *const Span,
    len: usize,
    aad: *const u8,
    aad_len: usize,
    data: *mut OpenData,
  );

  #[cfg(target_os = "linux")]
  fn rscrypto_chacha20_poly1305_open_scatter_aarch64(
    out_spans: *const Span,
    in_spans: *const Span,
    len: usize,
    aad: *const u8,
    aad_len: usize,
    data: *mut OpenData,
  );
}

#[inline]
pub(super) fn seal_in_place(key: &[u8; KEY_SIZE], nonce: &[u8; 12], aad: &[u8], buffer: &mut [u8]) -> [u8; 16] {
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

  // SAFETY: integrated aarch64 seal call because:
  // 1. This module is only compiled for macOS/Linux aarch64, where ASIMD/NEON is part of the target
  //    baseline ABI.
  // 2. `buffer.as_ptr()` and `buffer.as_mut_ptr()` are valid for `buffer.len()` bytes and may alias;
  //    the assembly routine supports in-place seal, matching the AWS-LC ABI.
  // 3. `aad.as_ptr()` is valid for `aad.len()` bytes, including the conventional dangling pointer for
  //    empty slices because the length is zero.
  // 4. `extra_ciphertext` points at 16 initialized bytes and `extra_ciphertext_len` is zero, matching
  //    the AWS-LC seal ABI for callers without extra trailing ciphertext.
  // 5. `data` is 16-byte aligned and matches the assembly input/output union layout; the assembly initializes all
  //    16 bytes of `data.out.tag` before the union field is read.
  // 6. After that write, the tag replaces the first 16 copied key bytes; the second half remains initialized key
  //    material and is volatile-zeroed before the stack allocation expires.
  unsafe {
    #[cfg(target_os = "macos")]
    rscrypto_chacha20_poly1305_seal_aarch64_apple_darwin(
      buffer.as_mut_ptr(),
      buffer.as_ptr(),
      buffer.len(),
      aad.as_ptr(),
      aad.len(),
      core::ptr::from_mut(&mut data),
    );
    #[cfg(target_os = "linux")]
    rscrypto_chacha20_poly1305_seal_aarch64(
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

/// Seal borrowed nonempty spans directly into separate ciphertext output.
#[cfg(target_endian = "little")]
pub(super) fn seal_gather<'input, 'output>(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; 12],
  aad: &[u8],
  input: &[super::super::gather::Slice<'input>],
  output: &'output mut [core::mem::MaybeUninit<u8>],
) -> ([u8; 16], &'output mut [u8]) {
  let extra_ciphertext = [0u8; 16];
  let mut total = 0usize;
  for part in input {
    let bytes = part.as_slice();
    assert!(!bytes.is_empty());
    total = total.strict_add(bytes.len());
  }
  assert_eq!(total, output.len());
  assert!(!input.is_empty());
  const {
    assert!(mem::size_of::<super::super::gather::Slice<'static>>() == 16);
  }
  let mut data = SealData {
    input: SealDataIn {
      key: *key,
      counter: 0,
      nonce: *nonce,
      extra_ciphertext: extra_ciphertext.as_ptr(),
      extra_ciphertext_len: 0,
    },
  };
  unsafe extern "C" {
    #[cfg(target_os = "macos")]
    fn rscrypto_chacha20_poly1305_seal_gather_aarch64_apple_darwin(
      out: *mut u8,
      input: *const super::super::gather::Slice<'_>,
      len: usize,
      aad: *const u8,
      aad_len: usize,
      data: *mut SealData,
    );
    #[cfg(target_os = "linux")]
    fn rscrypto_chacha20_poly1305_seal_gather_aarch64(
      out: *mut u8,
      input: *const super::super::gather::Slice<'_>,
      len: usize,
      aad: *const u8,
      aad_len: usize,
      data: *mut SealData,
    );
  }
  // SAFETY: the repr(C) spans retain live, initialized, nonempty input borrows.
  // Their checked total equals exclusive output. The public entry validated
  // counters and lengths. AArch64 NEON is baseline on these existing targets.
  // The native cursor loads only the current span's bytes; seams stay in SIMD
  // registers. It initializes all output once and writes the complete tag.
  // data has the unchanged aligned ABI; its copied key's remaining half clears
  // after the tag overwrites the first half, as in seal_in_place.
  unsafe {
    #[cfg(target_os = "macos")]
    rscrypto_chacha20_poly1305_seal_gather_aarch64_apple_darwin(
      output.as_mut_ptr().cast(),
      input.as_ptr(),
      total,
      aad.as_ptr(),
      aad.len(),
      core::ptr::from_mut(&mut data),
    );
    #[cfg(target_os = "linux")]
    rscrypto_chacha20_poly1305_seal_gather_aarch64(
      output.as_mut_ptr().cast(),
      input.as_ptr(),
      total,
      aad.as_ptr(),
      aad.len(),
      core::ptr::from_mut(&mut data),
    );
    crate::traits::ct::zeroize(&mut data.input.key[16..]);
    let tag = data.out.tag;
    (tag, core::slice::from_raw_parts_mut(output.as_mut_ptr().cast(), total))
  }
}

#[inline]
pub(super) fn open_in_place(key: &[u8; KEY_SIZE], nonce: &[u8; 12], aad: &[u8], buffer: &mut [u8]) -> [u8; 16] {
  let mut data = OpenData {
    input: OpenDataIn {
      key: *key,
      counter: 0,
      nonce: *nonce,
    },
  };

  // SAFETY: integrated aarch64 open call because:
  // 1. This module is only compiled for macOS/Linux aarch64, where ASIMD/NEON is part of the target
  //    baseline ABI.
  // 2. `buffer.as_ptr()` and `buffer.as_mut_ptr()` are valid for `buffer.len()` bytes and may alias;
  //    the assembly routine supports in-place open, matching the AWS-LC ABI.
  // 3. `aad.as_ptr()` is valid for `aad.len()` bytes, including the conventional dangling pointer for
  //    empty slices because the length is zero.
  // 4. `data` is 16-byte aligned and matches the assembly input/output union layout; the assembly initializes all
  //    16 bytes of `data.out.tag` before the union field is read.
  // 5. After that write, the tag replaces the first 16 copied key bytes; the second half remains initialized key
  //    material and is volatile-zeroed before the stack allocation expires.
  unsafe {
    #[cfg(target_os = "macos")]
    rscrypto_chacha20_poly1305_open_aarch64_apple_darwin(
      buffer.as_mut_ptr(),
      buffer.as_ptr(),
      buffer.len(),
      aad.as_ptr(),
      aad.len(),
      core::ptr::from_mut(&mut data),
    );
    #[cfg(target_os = "linux")]
    rscrypto_chacha20_poly1305_open_aarch64(
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

/// Open the ciphertext of `spans` in place, in order, and return the computed
/// tag. The caller compares it, and on a mismatch clears every span.
pub(super) fn open_scatter(key: &[u8; KEY_SIZE], nonce: &[u8; 12], aad: &[u8], spans: &[Span]) -> [u8; 16] {
  assert!(!spans.is_empty());
  let len = spans.iter().fold(0usize, |len, span| len.strict_add(span.len()));
  let mut data = OpenData {
    input: OpenDataIn {
      key: *key,
      counter: 0,
      nonce: *nonce,
    },
  };
  // SAFETY: integrated aarch64 span open because:
  // 1. NEON is part of the macOS/Linux aarch64 baseline ABI.
  // 2. Each repr(C) span holds the pointer and length of a nonempty segment
  //    borrowed exclusively for this call; the spans do not overlap, and their
  //    checked total is `len`, which the public entry validated against
  //    ChaCha20's block bound. The read, hash, and write cursors walk the same
  //    spans in order and touch only their bytes; seams stay in registers.
  // 3. `aad` is valid for `aad.len()` bytes; `data` has the aligned union
  //    layout of `open_in_place`, and the assembly writes all 16 tag bytes.
  // 4. The tag overwrites the first half of the copied key; the second half
  //    is cleared before the stack slot expires.
  unsafe {
    #[cfg(target_os = "macos")]
    rscrypto_chacha20_poly1305_open_scatter_aarch64_apple_darwin(
      spans.as_ptr(),
      spans.as_ptr(),
      len,
      aad.as_ptr(),
      aad.len(),
      core::ptr::from_mut(&mut data),
    );
    #[cfg(target_os = "linux")]
    rscrypto_chacha20_poly1305_open_scatter_aarch64(
      spans.as_ptr(),
      spans.as_ptr(),
      len,
      aad.as_ptr(),
      aad.len(),
      core::ptr::from_mut(&mut data),
    );
    crate::traits::ct::zeroize(&mut data.input.key[16..]);
    data.out.tag
  }
}
