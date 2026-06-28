//! rscrypto-owned aarch64 ChaCha20 assembly kernels.
//!
//! This module is diagnostic-only while the owned assembly competes against the
//! current Rust NEON backend. Production dispatch should not use a hand-written
//! kernel until the equivalence tests pass and benches show a real win.

#![allow(unsafe_code)]

use core::arch::global_asm;

use super::{BLOCK_SIZE, KEY_SIZE, NONCE_SIZE, aarch64_neon};

const BLOCKS_PER_CHUNK: usize = 8;
const CHUNK_SIZE: usize = BLOCK_SIZE * BLOCKS_PER_CHUNK;

#[cfg(target_os = "macos")]
global_asm!(include_str!("asm/rscrypto_chacha20_xor_aarch64_apple_darwin.s"));
#[cfg(target_os = "linux")]
global_asm!(include_str!("asm/rscrypto_chacha20_xor_aarch64_linux.s"));

unsafe extern "C" {
  #[cfg(target_os = "macos")]
  fn rscrypto_chacha20_xor_8block_aarch64_apple_darwin(
    buffer: *mut u8,
    chunks: usize,
    key: *const u8,
    counter: u32,
    nonce: *const u8,
  );

  #[cfg(target_os = "linux")]
  fn rscrypto_chacha20_xor_8block_aarch64_linux(
    buffer: *mut u8,
    chunks: usize,
    key: *const u8,
    counter: u32,
    nonce: *const u8,
  );
}

#[inline]
pub(super) fn xor_keystream_8block(
  key: &[u8; KEY_SIZE],
  initial_counter: u32,
  nonce: &[u8; NONCE_SIZE],
  buffer: &mut [u8],
) {
  let full_len = buffer.len().strict_div(CHUNK_SIZE).strict_mul(CHUNK_SIZE);
  if full_len != 0 {
    let chunks = full_len.strict_div(CHUNK_SIZE);
    // SAFETY: aarch64 ChaCha20 assembly call because:
    // 1. This module is compiled only for aarch64 macOS/Linux with baseline Advanced SIMD support.
    // 2. `buffer` is valid for `full_len == chunks * 512` writable bytes.
    // 3. `key` and `nonce` are fixed-size initialized arrays matching the assembly ABI.
    // 4. The assembly branches and memory addresses depend only on public buffer length.
    unsafe {
      #[cfg(target_os = "macos")]
      rscrypto_chacha20_xor_8block_aarch64_apple_darwin(
        buffer.as_mut_ptr(),
        chunks,
        key.as_ptr(),
        initial_counter,
        nonce.as_ptr(),
      );
      #[cfg(target_os = "linux")]
      rscrypto_chacha20_xor_8block_aarch64_linux(
        buffer.as_mut_ptr(),
        chunks,
        key.as_ptr(),
        initial_counter,
        nonce.as_ptr(),
      );
    }
  }

  let remainder = &mut buffer[full_len..];
  if !remainder.is_empty() {
    let tail_counter = initial_counter.wrapping_add(full_len.strict_div(BLOCK_SIZE) as u32);
    aarch64_neon::xor_keystream(key, tail_counter, nonce, remainder);
  }
}
