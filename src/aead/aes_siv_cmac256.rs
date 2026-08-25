//! AES-SIV-CMAC-256 nonce-based authenticated encryption (RFC 5297).
//!
//! This module implements only the registered 32-byte-key RFC 5116 profile. It does not expose
//! deterministic SIV, CMAC, S2V, raw AES, or a vector-of-associated-data interface.

use core::fmt;

use super::{AeadBufferError, OpenError, SealError, aes};
use crate::traits::ct;

const BLOCK_SIZE: usize = 16;
const KEY_SIZE: usize = 32;
const TAG_SIZE: usize = 16;
const CTR_BATCH_BLOCKS: usize = 8;
#[cfg(target_arch = "x86_64")]
const CTR_WIDE_BATCH_BLOCKS: usize = 16;

define_aead_key_type!(AesSivCmac256Key, KEY_SIZE, "AES-SIV-CMAC-256 secret key (32 bytes).");

define_aead_tag_type!(
  AesSivCmac256Tag,
  TAG_SIZE,
  "AES-SIV-CMAC-256 synthetic-IV authentication tag (16 bytes)."
);

define_unit_error! {
  /// Empty nonce supplied to the AES-SIV-CMAC-256 nonce-based profile.
  pub struct AesSivCmac256NonceError;
  "AES-SIV-CMAC-256 nonce must not be empty"
}

/// Borrowed non-empty nonce for the AES-SIV-CMAC-256 RFC 5116 profile.
///
/// RFC 5297 permits any nonce length of at least one byte. The nonce is the final S2V
/// associated-data component immediately before the plaintext.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct AesSivCmac256Nonce<'a>(&'a [u8]);

impl<'a> AesSivCmac256Nonce<'a> {
  /// Borrow the nonce bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(self) -> &'a [u8] {
    self.0
  }
}

impl<'a> TryFrom<&'a [u8]> for AesSivCmac256Nonce<'a> {
  type Error = AesSivCmac256NonceError;

  #[inline]
  fn try_from(bytes: &'a [u8]) -> Result<Self, Self::Error> {
    if bytes.is_empty() {
      Err(AesSivCmac256NonceError::new())
    } else {
      Ok(Self(bytes))
    }
  }
}

impl AsRef<[u8]> for AesSivCmac256Nonce<'_> {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    self.0
  }
}

impl fmt::Debug for AesSivCmac256Nonce<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("AesSivCmac256Nonce(")?;
    crate::hex::fmt_hex_lower(self.0, f)?;
    f.write_str(")")
  }
}

/// AES-SIV-CMAC-256 nonce-based authenticated encryption (RFC 5297 section 6.1).
///
/// The 32-byte key is split into independent AES-128 CMAC/S2V and CTR keys. The public profile
/// accepts one RFC 5116 associated-data string and a distinct non-empty variable-length nonce.
/// Detached operations return or accept the 16-byte synthetic IV as a typed tag. Combined
/// operations use the RFC-defined `synthetic_iv || ciphertext` layout.
///
/// Authentication failure is opaque and clears the complete unauthenticated plaintext buffer
/// before returning. The context is intentionally neither `Clone` nor `Copy`.
///
/// # Security
///
/// SIV preserves authenticity under nonce reuse, while repeated key/nonce/AAD/plaintext tuples
/// reveal equality. Applications should still issue unique nonces. Constant-time claims remain
/// compiler-, target-, feature-, and release-evidence-bound; see `ct.toml`.
pub struct AesSivCmac256 {
  cmac_key: aes::Aes128EncKey,
  ctr_key: aes::Aes128EncKey,
  cmac_subkey1: [u8; BLOCK_SIZE],
  cmac_subkey2: [u8; BLOCK_SIZE],
}

impl fmt::Debug for AesSivCmac256 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("AesSivCmac256").finish_non_exhaustive()
  }
}

impl Drop for AesSivCmac256 {
  fn drop(&mut self) {
    ct::zeroize_no_fence(&mut self.cmac_subkey1);
    ct::zeroize_no_fence(&mut self.cmac_subkey2);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

impl AesSivCmac256 {
  /// Key length in bytes.
  pub const KEY_SIZE: usize = KEY_SIZE;

  /// Minimum nonce length in bytes.
  pub const MIN_NONCE_SIZE: usize = 1;

  /// Synthetic-IV tag length in bytes.
  pub const TAG_SIZE: usize = TAG_SIZE;

  /// Construct an AES-SIV-CMAC-256 context.
  #[must_use]
  pub fn new(key: &AesSivCmac256Key) -> Self {
    let (cmac_key_bytes, ctr_key_bytes) = key
      .as_bytes()
      .split_first_chunk::<16>()
      .expect("AES-SIV-CMAC-256 key has two fixed 16-byte halves");
    let ctr_key_bytes = ctr_key_bytes
      .first_chunk::<16>()
      .expect("AES-SIV-CMAC-256 key has two fixed 16-byte halves");
    let cmac_key = aes::aes128_expand_key(cmac_key_bytes);
    let ctr_key = aes::aes128_expand_key(ctr_key_bytes);

    let mut l = [0u8; BLOCK_SIZE];
    aes::aes128_encrypt_block(&cmac_key, &mut l);
    let mut cmac_subkey1 = l;
    double_block(&mut cmac_subkey1);
    let mut cmac_subkey2 = cmac_subkey1;
    double_block(&mut cmac_subkey2);
    ct::zeroize(&mut l);

    Self {
      cmac_key,
      ctr_key,
      cmac_subkey1,
      cmac_subkey2,
    }
  }

  #[cfg(feature = "diag")]
  fn new_forced_portable(key: &AesSivCmac256Key) -> Self {
    let (cmac_key_bytes, ctr_key_bytes) = key
      .as_bytes()
      .split_first_chunk::<16>()
      .expect("AES-SIV-CMAC-256 key has two fixed 16-byte halves");
    let ctr_key_bytes = ctr_key_bytes
      .first_chunk::<16>()
      .expect("AES-SIV-CMAC-256 key has two fixed 16-byte halves");

    let cmac_key = aes::aes128_expand_key_forced_portable(cmac_key_bytes);
    let ctr_key = aes::aes128_expand_key_forced_portable(ctr_key_bytes);

    let mut l = [0u8; BLOCK_SIZE];
    aes::aes128_encrypt_block(&cmac_key, &mut l);
    let mut cmac_subkey1 = l;
    double_block(&mut cmac_subkey1);
    let mut cmac_subkey2 = cmac_subkey1;
    double_block(&mut cmac_subkey2);
    ct::zeroize(&mut l);

    Self {
      cmac_key,
      ctr_key,
      cmac_subkey1,
      cmac_subkey2,
    }
  }

  /// Rebuild a typed synthetic-IV tag from raw bytes.
  #[inline]
  pub fn tag_from_slice(bytes: &[u8]) -> Result<AesSivCmac256Tag, AeadBufferError> {
    if bytes.len() != TAG_SIZE {
      return Err(AeadBufferError::new());
    }
    let mut tag = [0u8; TAG_SIZE];
    tag.copy_from_slice(bytes);
    Ok(AesSivCmac256Tag::from_bytes(tag))
  }

  /// Encrypt `buffer` in place and return its detached synthetic-IV tag.
  #[must_use]
  pub fn seal_in_place(&self, nonce: AesSivCmac256Nonce<'_>, aad: &[u8], buffer: &mut [u8]) -> AesSivCmac256Tag {
    let tag = self.s2v(&[aad, nonce.as_bytes()], buffer);
    self.ctr_xor(&tag, buffer);
    AesSivCmac256Tag::from_bytes(tag)
  }

  /// Decrypt `buffer` in place and authenticate its detached synthetic-IV tag.
  ///
  /// On authentication failure the entire buffer is cleared before one opaque verification error
  /// is returned.
  pub fn open_in_place(
    &self,
    nonce: AesSivCmac256Nonce<'_>,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &AesSivCmac256Tag,
  ) -> Result<(), OpenError> {
    self.ctr_xor(tag.as_bytes(), buffer);
    let mut computed = self.s2v(&[aad, nonce.as_bytes()], buffer);
    let accepted = ct::fixed_eq(&computed, tag.as_bytes());
    ct::zeroize(&mut computed);

    if accepted.declassify() {
      Ok(())
    } else {
      ct::zeroize(buffer);
      Err(OpenError::verification())
    }
  }

  /// Encrypt `plaintext` into `out` as `synthetic_iv || ciphertext`.
  ///
  /// `out` must be exactly 16 bytes longer than `plaintext`.
  pub fn seal(
    &self,
    nonce: AesSivCmac256Nonce<'_>,
    aad: &[u8],
    plaintext: &[u8],
    out: &mut [u8],
  ) -> Result<(), SealError> {
    let expected = plaintext.len().checked_add(TAG_SIZE).ok_or_else(SealError::too_large)?;
    if out.len() != expected {
      return Err(SealError::buffer());
    }

    let (tag_out, ciphertext) = out.split_at_mut(TAG_SIZE);
    ciphertext.copy_from_slice(plaintext);
    let tag = self.seal_in_place(nonce, aad, ciphertext);
    tag_out.copy_from_slice(tag.as_bytes());
    Ok(())
  }

  /// Decrypt RFC-layout `synthetic_iv || ciphertext` into `out`.
  ///
  /// `out` must be exactly 16 bytes shorter than `tag_and_ciphertext`. On authentication failure
  /// every byte of `out` is cleared before one opaque verification error is returned.
  pub fn open(
    &self,
    nonce: AesSivCmac256Nonce<'_>,
    aad: &[u8],
    tag_and_ciphertext: &[u8],
    out: &mut [u8],
  ) -> Result<(), OpenError> {
    if tag_and_ciphertext.len() < TAG_SIZE {
      return Err(OpenError::buffer());
    }
    let plaintext_len = tag_and_ciphertext.len().strict_sub(TAG_SIZE);
    if out.len() != plaintext_len {
      return Err(OpenError::buffer());
    }

    let (tag_bytes, ciphertext) = tag_and_ciphertext.split_at(TAG_SIZE);
    let mut raw_tag = [0u8; TAG_SIZE];
    raw_tag.copy_from_slice(tag_bytes);
    let tag = AesSivCmac256Tag::from_bytes(raw_tag);
    out.copy_from_slice(ciphertext);
    self.open_in_place(nonce, aad, out, &tag)
  }

  fn cmac_step(&self, state: &mut [u8; BLOCK_SIZE], block: &[u8]) {
    for (state_byte, input_byte) in state.iter_mut().zip(block) {
      *state_byte ^= *input_byte;
    }
    aes::aes128_encrypt_block(&self.cmac_key, state);
  }

  fn cmac(&self, input: &[u8]) -> [u8; BLOCK_SIZE] {
    let mut state = [0u8; BLOCK_SIZE];
    let mut final_block = [0u8; BLOCK_SIZE];

    if input.is_empty() {
      final_block[0] = 0x80;
      xor_block(&mut final_block, &self.cmac_subkey2);
    } else {
      let remainder = input.len().strict_rem(BLOCK_SIZE);
      let final_len = if remainder == 0 { BLOCK_SIZE } else { remainder };
      let prefix_len = input.len().strict_sub(final_len);
      let (prefix, final_input) = input.split_at(prefix_len);
      let (prefix_blocks, prefix_tail) = prefix.as_chunks::<BLOCK_SIZE>();
      debug_assert!(prefix_tail.is_empty());
      if !prefix_blocks.is_empty() {
        aes::aes128_xor_encrypt_blocks(&self.cmac_key, &mut state, prefix_blocks);
      }
      final_block[..final_input.len()].copy_from_slice(final_input);
      if final_input.len() == BLOCK_SIZE {
        xor_block(&mut final_block, &self.cmac_subkey1);
      } else {
        final_block[final_input.len()] = 0x80;
        xor_block(&mut final_block, &self.cmac_subkey2);
      }
    }

    self.cmac_step(&mut state, &final_block);
    ct::zeroize(&mut final_block);
    state
  }

  fn cmac_xorend(&self, input: &[u8], suffix: &[u8; BLOCK_SIZE]) -> [u8; BLOCK_SIZE] {
    debug_assert!(input.len() >= BLOCK_SIZE);
    let suffix_start = input.len().strict_sub(BLOCK_SIZE);
    let prefix_len = suffix_start.strict_sub(suffix_start.strict_rem(BLOCK_SIZE));
    let (prefix, affected) = input.split_at(prefix_len);
    let mut state = [0u8; BLOCK_SIZE];
    let (prefix_blocks, prefix_tail) = prefix.as_chunks::<BLOCK_SIZE>();
    debug_assert!(prefix_tail.is_empty());
    if !prefix_blocks.is_empty() {
      aes::aes128_xor_encrypt_blocks(&self.cmac_key, &mut state, prefix_blocks);
    }

    let mut offset = 0usize;
    while affected.len().strict_sub(offset) > BLOCK_SIZE {
      let mut block = [0u8; BLOCK_SIZE];
      block.copy_from_slice(&affected[offset..offset.strict_add(BLOCK_SIZE)]);
      xor_suffix_window(&mut block, prefix_len.strict_add(offset), suffix_start, suffix);
      self.cmac_step(&mut state, &block);
      ct::zeroize(&mut block);
      offset = offset.strict_add(BLOCK_SIZE);
    }

    let final_input = &affected[offset..];
    let mut final_block = [0u8; BLOCK_SIZE];
    final_block[..final_input.len()].copy_from_slice(final_input);
    xor_suffix_window(&mut final_block, prefix_len.strict_add(offset), suffix_start, suffix);
    if final_input.len() == BLOCK_SIZE {
      xor_block(&mut final_block, &self.cmac_subkey1);
    } else {
      final_block[final_input.len()] = 0x80;
      xor_block(&mut final_block, &self.cmac_subkey2);
    }
    self.cmac_step(&mut state, &final_block);
    ct::zeroize(&mut final_block);
    state
  }

  fn s2v(&self, components: &[&[u8]], plaintext: &[u8]) -> [u8; BLOCK_SIZE] {
    let mut d = self.cmac(&[0u8; BLOCK_SIZE]);
    for component in components {
      double_block(&mut d);
      let mut component_tag = self.cmac(component);
      xor_block(&mut d, &component_tag);
      ct::zeroize(&mut component_tag);
    }

    let tag = if plaintext.len() >= BLOCK_SIZE {
      self.cmac_xorend(plaintext, &d)
    } else {
      double_block(&mut d);
      for (dst, src) in d.iter_mut().zip(plaintext) {
        *dst ^= *src;
      }
      d[plaintext.len()] ^= 0x80;
      self.cmac(&d)
    };
    ct::zeroize(&mut d);
    tag
  }

  fn ctr_xor(&self, tag: &[u8; TAG_SIZE], data: &mut [u8]) {
    if data.is_empty() {
      return;
    }

    #[cfg(target_arch = "x86_64")]
    if data.len() >= CTR_WIDE_BATCH_BLOCKS.strict_mul(BLOCK_SIZE) {
      self.ctr_xor_batched::<CTR_WIDE_BATCH_BLOCKS>(tag, data);
      return;
    }

    self.ctr_xor_batched::<CTR_BATCH_BLOCKS>(tag, data);
  }

  fn ctr_xor_batched<const BATCH_BLOCKS: usize>(&self, tag: &[u8; TAG_SIZE], data: &mut [u8]) {
    debug_assert!(BATCH_BLOCKS > 0);
    let mut initial = *tag;
    initial[8] &= 0x7f;
    initial[12] &= 0x7f;
    let mut counter = u128::from_be_bytes(initial);
    let mut keystream = [[0u8; BLOCK_SIZE]; BATCH_BLOCKS];
    let mut offset = 0usize;

    while offset < data.len() {
      let remaining = data.len().strict_sub(offset);
      let whole_blocks = remaining.strict_div(BLOCK_SIZE);
      let partial_block = usize::from(remaining.strict_rem(BLOCK_SIZE) != 0);
      let block_count = whole_blocks.strict_add(partial_block).min(BATCH_BLOCKS);

      for (lane, block) in keystream[..block_count].iter_mut().enumerate() {
        *block = counter.wrapping_add(lane as u128).to_be_bytes();
      }
      aes::aes128_encrypt_blocks_ecb(&self.ctr_key, &mut keystream[..block_count]);

      for block in &keystream[..block_count] {
        let take = data.len().strict_sub(offset).min(BLOCK_SIZE);
        for (output, mask) in data[offset..offset.strict_add(take)].iter_mut().zip(block) {
          *output ^= *mask;
        }
        offset = offset.strict_add(take);
      }

      counter = counter.wrapping_add(block_count as u128);
      ct::zeroize_no_fence(keystream.as_flattened_mut());
    }

    ct::zeroize_no_fence(&mut initial);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

/// Evaluate the private CMAC/S2V state through the forced-portable AES authority.
///
/// This diagnostic exists only for constant-time and backend-equivalence evidence. It is not a
/// supported CMAC, S2V, or deterministic-SIV product API.
#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline(never)]
#[must_use]
pub fn diag_aes_siv_cmac256_s2v_portable(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; BLOCK_SIZE],
  aad: &[u8; 32],
  plaintext: &[u8; 48],
) -> [u8; TAG_SIZE] {
  let key = AesSivCmac256Key::from_bytes(*key);
  let cipher = AesSivCmac256::new_forced_portable(&key);
  cipher.s2v(&[aad, nonce], plaintext)
}

/// Run a complete fixed-shape open through the forced-portable AES authority.
///
/// This diagnostic exists only for generated-code and timing evidence. The returned byte is an
/// opaque success indicator; authentication failure still clears the complete plaintext buffer.
#[cfg(feature = "diag")]
#[doc(hidden)]
#[inline(never)]
#[must_use]
pub fn diag_aes_siv_cmac256_open_portable(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; BLOCK_SIZE],
  aad: &[u8; 32],
  ciphertext: &[u8; 48],
  tag: &[u8; TAG_SIZE],
) -> u8 {
  let key = AesSivCmac256Key::from_bytes(*key);
  let cipher = AesSivCmac256::new_forced_portable(&key);
  let nonce = AesSivCmac256Nonce::try_from(nonce.as_slice()).expect("fixed diagnostic nonce is non-empty");
  let tag = AesSivCmac256Tag::from_bytes(*tag);
  let mut buffer = *ciphertext;
  let accepted = cipher.open_in_place(nonce, aad, &mut buffer, &tag).is_ok();
  let digest = buffer.iter().copied().fold(0u8, |acc, byte| acc ^ byte);
  digest ^ u8::from(accepted)
}

#[cfg(feature = "diag")]
#[doc(hidden)]
/// Exercise AES-SIV construction, seal, open, local cleanup, and retained-owner drop.
#[unsafe(no_mangle)]
#[inline(never)]
#[must_use]
pub fn diag_zeroize_aes_siv_cmac256(key: [u8; KEY_SIZE], nonce: [u8; BLOCK_SIZE], input: [u8; 48]) -> u8 {
  let key = AesSivCmac256Key::from_bytes(key);
  let cipher = AesSivCmac256::new(&key);
  let nonce = AesSivCmac256Nonce::try_from(nonce.as_slice()).expect("fixed diagnostic nonce is non-empty");
  let mut buffer = input;
  let tag = cipher.seal_in_place(nonce, b"zeroize evidence", &mut buffer);
  cipher
    .open_in_place(nonce, b"zeroize evidence", &mut buffer, &tag)
    .expect("fresh diagnostic ciphertext authenticates");
  buffer[0] ^ tag.as_bytes()[0]
}

#[inline]
fn xor_block(left: &mut [u8; BLOCK_SIZE], right: &[u8; BLOCK_SIZE]) {
  for (left_byte, right_byte) in left.iter_mut().zip(right) {
    *left_byte ^= *right_byte;
  }
}

#[inline]
fn double_block(block: &mut [u8; BLOCK_SIZE]) {
  let mut carry = 0u8;
  for byte in block.iter_mut().rev() {
    let next_carry = *byte >> 7;
    *byte = (*byte << 1) | carry;
    carry = next_carry;
  }
  block[BLOCK_SIZE.strict_sub(1)] ^= 0x87 & 0u8.wrapping_sub(carry);
}

fn xor_suffix_window(block: &mut [u8; BLOCK_SIZE], block_start: usize, suffix_start: usize, suffix: &[u8; BLOCK_SIZE]) {
  for (index, byte) in block.iter_mut().enumerate() {
    let position = block_start.strict_add(index);
    if position >= suffix_start && position < suffix_start.strict_add(BLOCK_SIZE) {
      *byte ^= suffix[position.strict_sub(suffix_start)];
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::aead::test_vectors::{hex_vec, hex16, hex32};

  fn context(key_hex: &str) -> AesSivCmac256 {
    AesSivCmac256::new(&AesSivCmac256Key::from_bytes(hex32(key_hex)))
  }

  #[test]
  fn rfc5297_appendix_a1_deterministic_construction() {
    let cipher = context("fffefdfcfbfaf9f8f7f6f5f4f3f2f1f0f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff");
    let aad = hex_vec("101112131415161718191a1b1c1d1e1f2021222324252627");
    let mut plaintext = hex_vec("112233445566778899aabbccddee");
    let expected = hex_vec("85632d07c6e8f37f950acd320a2ecc9340c02b9690c4dc04daef7f6afe5c");

    let tag = cipher.s2v(&[&aad], &plaintext);
    assert_eq!(tag, expected[..TAG_SIZE]);
    cipher.ctr_xor(&tag, &mut plaintext);
    assert_eq!(plaintext, expected[TAG_SIZE..]);
  }

  #[test]
  fn rfc5297_appendix_a2_nonce_based_construction() {
    let cipher = context("7f7e7d7c7b7a79787776757473727170404142434445464748494a4b4c4d4e4f");
    let aad1 = hex_vec("00112233445566778899aabbccddeeffdeaddadadeaddadaffeeddccbbaa99887766554433221100");
    let aad2 = hex_vec("102030405060708090a0");
    let nonce = hex_vec("09f911029d74e35bd84156c5635688c0");
    let mut plaintext =
      hex_vec("7468697320697320736f6d6520706c61696e7465787420746f20656e6372797074207573696e67205349562d414553");
    let expected = hex_vec(
      "7bdb6e3b432667eb06f4d14bff2fbd0fcb900f2fddbe404326601965c889bf17dba77ceb094fa663b7a3f748ba8af829ea64ad544a272e9c485b62a3fd5c0d",
    );

    let tag = cipher.s2v(&[&aad1, &aad2, &nonce], &plaintext);
    assert_eq!(tag, expected[..TAG_SIZE]);
    cipher.ctr_xor(&tag, &mut plaintext);
    assert_eq!(plaintext, expected[TAG_SIZE..]);
  }

  #[test]
  fn ctr_uses_full_big_endian_counter_and_clears_required_bits() {
    use ::aes::cipher::{Array, BlockCipherEncrypt as _, KeyInit as _};

    let key = hex32("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f");
    let cipher = AesSivCmac256::new(&AesSivCmac256Key::from_bytes(key));
    let tag = hex16("fffffffffffffffffffffffffffffffe");
    let mut actual = [0u8; 48];
    cipher.ctr_xor(&tag, &mut actual);

    let ctr_key: [u8; 16] = key[16..].try_into().expect("CTR key half has fixed length");
    let aes = ::aes::Aes128::new(&Array::from(ctr_key));
    let mut counter_bytes = tag;
    counter_bytes[8] &= 0x7f;
    counter_bytes[12] &= 0x7f;
    let counter = u128::from_be_bytes(counter_bytes);
    let mut expected = [0u8; 48];
    for (index, chunk) in expected.as_chunks_mut::<16>().0.iter_mut().enumerate() {
      let mut block = Array::from(counter.wrapping_add(index as u128).to_be_bytes());
      aes.encrypt_block(&mut block);
      chunk.copy_from_slice(&block);
    }
    assert_eq!(actual, expected);
  }

  #[cfg(feature = "diag")]
  #[test]
  fn forced_portable_cmac_s2v_and_ctr_match_selected_backend() {
    const LENGTHS: &[usize] = &[0, 1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 127, 128, 129];

    for case in 0u8..32 {
      let mut key_bytes = [0u8; KEY_SIZE];
      for (index, byte) in key_bytes.iter_mut().enumerate() {
        let index = u8::try_from(index).expect("AES-SIV key index fits in u8");
        *byte = case.wrapping_mul(29).wrapping_add(index).rotate_left(1);
      }
      let key = AesSivCmac256Key::from_bytes(key_bytes);
      let selected = AesSivCmac256::new(&key);
      let portable = AesSivCmac256::new_forced_portable(&key);

      let mut storage = [0u8; 131];
      for (index, byte) in storage.iter_mut().enumerate() {
        let index = u8::try_from(index).expect("AES-SIV test input index fits in u8");
        *byte = case.wrapping_mul(17).wrapping_add(index).rotate_left(3);
      }
      let offset = usize::from(case & 1);
      let nonce = &storage[offset..offset.strict_add(16)];
      let aad = &storage[offset..offset.strict_add(32)];

      for &len in LENGTHS {
        let input = &storage[offset..offset.strict_add(len)];
        assert_eq!(selected.cmac(input), portable.cmac(input), "CMAC case={case} len={len}");
        assert_eq!(
          selected.s2v(&[aad, nonce], input),
          portable.s2v(&[aad, nonce], input),
          "S2V case={case} len={len}"
        );

        let tag = selected.s2v(&[aad, nonce], input);
        let mut selected_ctr = input.to_vec();
        let mut portable_ctr = selected_ctr.clone();
        selected.ctr_xor(&tag, &mut selected_ctr);
        portable.ctr_xor(&tag, &mut portable_ctr);
        assert_eq!(selected_ctr, portable_ctr, "CTR case={case} len={len}");
      }
    }
  }
}
