use super::{KEY_SIZE, NONCE_SIZE, xor_keystream_u32x4_impl};

/// Generate and XOR a ChaCha20 stream with the RISC-V vector kernel.
///
/// # Safety
///
/// The caller must ensure that the RISC-V vector extension is available and that `buffer`'s 64-byte block count fits
/// the counter range starting at `initial_counter`.
#[inline]
pub(super) unsafe fn xor_keystream(
  key: &[u8; KEY_SIZE],
  initial_counter: u32,
  nonce: &[u8; NONCE_SIZE],
  buffer: &mut [u8],
) {
  // SAFETY: the caller established the vector capability and counter range required by the shared implementation.
  unsafe { xor_keystream_u32x4_impl(key, initial_counter, nonce, buffer) }
}
