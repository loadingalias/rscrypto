//! Diagnostic x86_64 short-message ChaCha20-Poly1305 seal path.

use core::sync::atomic::{Ordering, compiler_fence};

use crate::{
  aead::{AeadByteLengths, SealError, chacha20, poly1305},
  traits::ct,
};

pub(super) const SHORT_MAX_LEN: usize = 256;

pub(super) fn encrypt_in_place_short(
  key: &[u8; chacha20::KEY_SIZE],
  nonce: &[u8; chacha20::NONCE_SIZE],
  aad: &[u8],
  buffer: &mut [u8],
) -> Option<Result<[u8; 16], SealError>> {
  if buffer.is_empty() || buffer.len() > SHORT_MAX_LEN {
    return None;
  }

  let lengths = match AeadByteLengths::try_new(aad.len(), buffer.len()) {
    Ok(lengths) => lengths,
    Err(_) => return Some(Err(SealError::too_large())),
  };

  let mut block0 = chacha20::block(key, 0, nonce);
  let mut poly_key = [0u8; chacha20::POLY1305_KEY_SIZE];
  poly_key.copy_from_slice(&block0[..chacha20::POLY1305_KEY_SIZE]);
  ct::zeroize_no_fence(&mut block0);

  let mut authenticator = poly1305::AeadScalar::new(&poly_key);
  ct::zeroize_no_fence(&mut poly_key);
  authenticator.update_padded_segment(aad);

  if chacha20::try_xor_keystream_x86_ssse3_x4(key, 1, nonce, buffer) {
    authenticator.update_padded_segment(buffer);
  } else {
    encrypt_ciphertext_into_authenticator(key, nonce, buffer, &mut authenticator);
  }

  let tag = authenticator.finalize(lengths);
  compiler_fence(Ordering::SeqCst);
  Some(Ok(tag))
}

fn encrypt_ciphertext_into_authenticator(
  key: &[u8; chacha20::KEY_SIZE],
  nonce: &[u8; chacha20::NONCE_SIZE],
  buffer: &mut [u8],
  authenticator: &mut poly1305::AeadScalar,
) {
  let mut counter = 1u32;
  for chunk in buffer.chunks_mut(chacha20::BLOCK_SIZE) {
    let mut keystream = chacha20::block(key, counter, nonce);
    encrypt_chunk_into_authenticator(chunk, &keystream, authenticator);
    ct::zeroize_no_fence(&mut keystream);
    counter = counter.wrapping_add(1);
  }
}

fn encrypt_chunk_into_authenticator(
  chunk: &mut [u8],
  keystream: &[u8; chacha20::BLOCK_SIZE],
  authenticator: &mut poly1305::AeadScalar,
) {
  let mut offset = 0usize;
  while offset < chunk.len() {
    let take = 16usize.min(chunk.len().strict_sub(offset));
    let mut block = [0u8; 16];
    let input = &mut chunk[offset..offset.strict_add(take)];
    let key_bytes = &keystream[offset..offset.strict_add(take)];

    for ((dst, key_byte), auth_byte) in input.iter_mut().zip(key_bytes).zip(block.iter_mut()) {
      *dst ^= *key_byte;
      *auth_byte = *dst;
    }

    authenticator.update_padded_block(&block);
    offset = offset.strict_add(take);
  }
}
