#![allow(clippy::indexing_slicing)]

//! AES-256-GCM-SIV public AEAD surface (RFC 8452).

use core::fmt;

#[cfg(target_arch = "x86_64")]
use super::polyval::{accumulate_4blocks, precompute_powers};
use super::{AeadBufferError, Nonce96, OpenError, aes, polyval};
use crate::traits::{Aead, VerificationError, ct};

const KEY_SIZE: usize = 32;
const TAG_SIZE: usize = 16;
const NONCE_SIZE: usize = Nonce96::LENGTH;

/// Maximum plaintext length: 2^36 - 32 bytes (per RFC 8452 §5).
/// Beyond this the 32-bit CTR counter wraps, causing keystream reuse.
const MAX_PLAINTEXT_LEN: u64 = (1u64 << 36).strict_sub(32);

/// AES-256-GCM-SIV secret key (32 bytes).
#[derive(Clone)]
pub struct Aes256GcmSivKey([u8; Self::LENGTH]);

impl PartialEq for Aes256GcmSivKey {
  fn eq(&self, other: &Self) -> bool {
    ct::constant_time_eq(&self.0, &other.0)
  }
}

impl Eq for Aes256GcmSivKey {}

impl Aes256GcmSivKey {
  /// Key length in bytes.
  pub const LENGTH: usize = KEY_SIZE;

  /// Construct a typed key from raw bytes.
  #[inline]
  #[must_use]
  pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
    Self(bytes)
  }

  /// Return the key bytes.
  #[inline]
  #[must_use]
  pub fn to_bytes(&self) -> [u8; Self::LENGTH] {
    self.0
  }

  /// Borrow the key bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }
}

impl Default for Aes256GcmSivKey {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl AsRef<[u8]> for Aes256GcmSivKey {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

impl fmt::Debug for Aes256GcmSivKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("Aes256GcmSivKey(****)")
  }
}

impl Aes256GcmSivKey {
  /// Construct a key by filling bytes from the provided closure.
  ///
  /// ```ignore
  /// let key = Aes256GcmSivKey::generate(|buf| getrandom::fill(buf).unwrap());
  /// ```
  #[inline]
  #[must_use]
  pub fn generate(fill: impl FnOnce(&mut [u8; Self::LENGTH])) -> Self {
    let mut bytes = [0u8; Self::LENGTH];
    fill(&mut bytes);
    Self(bytes)
  }
}

impl_hex_fmt_secret!(Aes256GcmSivKey);

impl Drop for Aes256GcmSivKey {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

/// AES-256-GCM-SIV authentication tag (16 bytes).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Aes256GcmSivTag([u8; Self::LENGTH]);

impl Aes256GcmSivTag {
  /// Tag length in bytes.
  pub const LENGTH: usize = TAG_SIZE;

  /// Construct a typed tag from raw bytes.
  #[inline]
  #[must_use]
  pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
    Self(bytes)
  }

  /// Return the tag bytes.
  #[inline]
  #[must_use]
  pub const fn to_bytes(self) -> [u8; Self::LENGTH] {
    self.0
  }

  /// Borrow the tag bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }
}

impl Default for Aes256GcmSivTag {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl AsRef<[u8]> for Aes256GcmSivTag {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

impl fmt::Debug for Aes256GcmSivTag {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Aes256GcmSivTag(")?;
    crate::hex::fmt_hex_lower(&self.0, f)?;
    write!(f, ")")
  }
}

impl_hex_fmt!(Aes256GcmSivTag);

/// AES-256-GCM-SIV AEAD (RFC 8452).
///
/// Nonce-misuse resistant authenticated encryption. On nonce reuse, only
/// the authentication guarantee degrades — confidentiality is preserved
/// up to a multi-message distinguishing bound.
#[derive(Clone)]
pub struct Aes256GcmSiv {
  master_ek: aes::Aes256EncKey,
}

impl fmt::Debug for Aes256GcmSiv {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Aes256GcmSiv").finish_non_exhaustive()
  }
}

impl Aes256GcmSiv {
  /// Key length in bytes.
  pub const KEY_SIZE: usize = KEY_SIZE;

  /// Nonce length in bytes.
  pub const NONCE_SIZE: usize = NONCE_SIZE;

  /// Tag length in bytes.
  pub const TAG_SIZE: usize = TAG_SIZE;

  /// Construct a new AES-256-GCM-SIV instance from `key`.
  #[inline]
  #[must_use]
  pub fn new(key: &Aes256GcmSivKey) -> Self {
    <Self as Aead>::new(key)
  }

  /// Rebuild a typed tag from raw tag bytes.
  #[inline]
  pub fn tag_from_slice(bytes: &[u8]) -> Result<Aes256GcmSivTag, AeadBufferError> {
    <Self as Aead>::tag_from_slice(bytes)
  }

  /// Encrypt `buffer` in place and return the detached authentication tag.
  #[inline]
  #[must_use]
  pub fn encrypt_in_place(&self, nonce: &Nonce96, aad: &[u8], buffer: &mut [u8]) -> Aes256GcmSivTag {
    <Self as Aead>::encrypt_in_place(self, nonce, aad, buffer)
  }

  /// Decrypt `buffer` in place and verify the detached authentication tag.
  #[inline]
  pub fn decrypt_in_place(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Aes256GcmSivTag,
  ) -> Result<(), VerificationError> {
    <Self as Aead>::decrypt_in_place(self, nonce, aad, buffer, tag)
  }

  /// Encrypt `plaintext` into `out` as `ciphertext || tag`.
  #[inline]
  pub fn encrypt(&self, nonce: &Nonce96, aad: &[u8], plaintext: &[u8], out: &mut [u8]) -> Result<(), AeadBufferError> {
    <Self as Aead>::encrypt(self, nonce, aad, plaintext, out)
  }

  /// Decrypt a combined `ciphertext || tag` into `out`.
  #[inline]
  pub fn decrypt(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    ciphertext_and_tag: &[u8],
    out: &mut [u8],
  ) -> Result<(), OpenError> {
    <Self as Aead>::decrypt(self, nonce, aad, ciphertext_and_tag, out)
  }
}

// ---------------------------------------------------------------------------
// RFC 8452 construction internals
// ---------------------------------------------------------------------------

/// Derive per-message authentication and encryption keys from the cached
/// master-key schedule and nonce (RFC 8452 §4, AES-256 variant).
///
/// Returns (auth_key [16 bytes], enc_key [32 bytes]).
///
/// Prepares all 6 counter blocks up front and encrypts them in a single
/// batch call. On s390x this collapses 6 individual KM invocations into
/// one, eliminating per-block overhead that dominates at small sizes.
#[inline]
fn derive_keys(master_ek: &aes::Aes256EncKey, nonce: &Nonce96) -> ([u8; 16], [u8; 32]) {
  let nonce_bytes = nonce.as_bytes();

  // Build all 6 counter blocks: counter (LE32) || nonce (96 bits).
  let mut blocks = [[0u8; 16]; 6];
  let mut i = 0u32;
  while i < 6 {
    blocks[i as usize][0..4].copy_from_slice(&i.to_le_bytes());
    blocks[i as usize][4..16].copy_from_slice(nonce_bytes);
    i = i.strict_add(1);
  }

  // Encrypt all 6 blocks (single KM call on s390x, per-block elsewhere).
  aes::aes256_encrypt_blocks_ecb(master_ek, &mut blocks);

  // Extract the first 8 bytes of each encrypted block.
  let mut auth_key = [0u8; 16];
  let mut enc_key = [0u8; 32];
  auth_key[0..8].copy_from_slice(&blocks[0][0..8]);
  auth_key[8..16].copy_from_slice(&blocks[1][0..8]);
  enc_key[0..8].copy_from_slice(&blocks[2][0..8]);
  enc_key[8..16].copy_from_slice(&blocks[3][0..8]);
  enc_key[16..24].copy_from_slice(&blocks[4][0..8]);
  enc_key[24..32].copy_from_slice(&blocks[5][0..8]);

  ct::zeroize(blocks.as_flattened_mut());
  (auth_key, enc_key)
}

/// Compute the POLYVAL-based authentication tag (RFC 8452 §5 steps 1-3).
///
/// Accepts the pre-expanded encryption key to avoid redundant key expansion
/// (the same expanded key is reused for AES-CTR in the caller).
#[inline]
fn compute_tag(
  auth_key: &[u8; 16],
  enc_ek: &aes::Aes256EncKey,
  nonce: &Nonce96,
  aad: &[u8],
  plaintext: &[u8],
) -> [u8; TAG_SIZE] {
  // POLYVAL over: padded AAD || padded plaintext || length block
  let mut pv = polyval::Polyval::new(auth_key);

  // Process AAD (padded to 16-byte blocks).
  pv.update_padded(aad);

  // Process plaintext (padded to 16-byte blocks).
  pv.update_padded(plaintext);

  // Length block: [aad_bits as u64 LE || plaintext_bits as u64 LE].
  let aad_bits = (aad.len() as u64).strict_mul(8);
  let pt_bits = (plaintext.len() as u64).strict_mul(8);
  let mut length_block = [0u8; 16];
  length_block[0..8].copy_from_slice(&aad_bits.to_le_bytes());
  length_block[8..16].copy_from_slice(&pt_bits.to_le_bytes());
  pv.update_block(&length_block);

  let mut s = pv.finalize();

  // XOR nonce into the first 12 bytes.
  let nonce_bytes = nonce.as_bytes();
  let mut j = 0usize;
  while j < 12 {
    s[j] ^= nonce_bytes[j];
    j = j.strict_add(1);
  }

  // Clear the MSB of the last byte.
  s[15] &= 0x7f;

  // Encrypt with AES to get the tag.
  aes::aes256_encrypt_block(enc_ek, &mut s);

  s
}

/// Compute the POLYVAL-based authentication tag using 4-block wide processing.
///
/// Same semantics as `compute_tag` but processes data in 4-block (64-byte)
/// chunks via `accumulate_4blocks`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn compute_tag_wide(
  auth_key: &[u8; 16],
  enc_ek: &aes::Aes256EncKey,
  nonce: &Nonce96,
  aad: &[u8],
  plaintext: &[u8],
) -> [u8; TAG_SIZE] {
  let h = u128::from_le_bytes(*auth_key);
  let powers = precompute_powers(h);
  let h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];

  let mut acc: u128 = 0;

  // Process AAD in 4-block wide chunks.
  let mut offset = 0usize;
  while offset.strict_add(64) <= aad.len() {
    let mut blocks = [0u128; 4];
    let mut i = 0usize;
    while i < 4 {
      let base = offset.strict_add(i.strict_mul(16));
      let mut block = [0u8; 16];
      block.copy_from_slice(&aad[base..base.strict_add(16)]);
      blocks[i] = u128::from_le_bytes(block);
      i = i.strict_add(1);
    }
    acc = accumulate_4blocks(acc, h, &h_powers_rev, &blocks);
    offset = offset.strict_add(64);
  }
  // Remaining AAD single blocks.
  while offset.strict_add(16) <= aad.len() {
    let mut block = [0u8; 16];
    block.copy_from_slice(&aad[offset..offset.strict_add(16)]);
    acc ^= u128::from_le_bytes(block);
    acc = polyval::clmul128_reduce(acc, h);
    offset = offset.strict_add(16);
  }
  let remaining_aad = aad.len().strict_sub(offset);
  if remaining_aad > 0 {
    let mut block = [0u8; 16];
    block[..remaining_aad].copy_from_slice(&aad[offset..]);
    acc ^= u128::from_le_bytes(block);
    acc = polyval::clmul128_reduce(acc, h);
  }

  // Process plaintext in 4-block wide chunks.
  offset = 0;
  while offset.strict_add(64) <= plaintext.len() {
    let mut blocks = [0u128; 4];
    let mut i = 0usize;
    while i < 4 {
      let base = offset.strict_add(i.strict_mul(16));
      let mut block = [0u8; 16];
      block.copy_from_slice(&plaintext[base..base.strict_add(16)]);
      blocks[i] = u128::from_le_bytes(block);
      i = i.strict_add(1);
    }
    acc = accumulate_4blocks(acc, h, &h_powers_rev, &blocks);
    offset = offset.strict_add(64);
  }
  // Remaining plaintext single blocks.
  while offset.strict_add(16) <= plaintext.len() {
    let mut block = [0u8; 16];
    block.copy_from_slice(&plaintext[offset..offset.strict_add(16)]);
    acc ^= u128::from_le_bytes(block);
    acc = polyval::clmul128_reduce(acc, h);
    offset = offset.strict_add(16);
  }
  let remaining_pt = plaintext.len().strict_sub(offset);
  if remaining_pt > 0 {
    let mut block = [0u8; 16];
    block[..remaining_pt].copy_from_slice(&plaintext[offset..]);
    acc ^= u128::from_le_bytes(block);
    acc = polyval::clmul128_reduce(acc, h);
  }

  // Length block: [aad_bits as u64 LE || pt_bits as u64 LE].
  let aad_bits = (aad.len() as u64).strict_mul(8);
  let pt_bits = (plaintext.len() as u64).strict_mul(8);
  let mut length_block = [0u8; 16];
  length_block[0..8].copy_from_slice(&aad_bits.to_le_bytes());
  length_block[8..16].copy_from_slice(&pt_bits.to_le_bytes());
  acc ^= u128::from_le_bytes(length_block);
  acc = polyval::clmul128_reduce(acc, h);

  // Finalize: convert to bytes, XOR nonce, clear MSB, encrypt.
  let mut s = acc.to_le_bytes();
  let nonce_bytes = nonce.as_bytes();
  let mut j = 0usize;
  while j < 12 {
    s[j] ^= nonce_bytes[j];
    j = j.strict_add(1);
  }
  s[15] &= 0x7f;
  aes::aes256_encrypt_block(enc_ek, &mut s);
  s
}

// ---------------------------------------------------------------------------
// aarch64 fused encrypt/decrypt (single #[target_feature] scope)
// ---------------------------------------------------------------------------
//
// On aarch64 the "scalar path" crosses a #[target_feature] boundary for every
// AES-CE encrypt_block and PMULL clmul128_reduce call, causing register spills
// that dominate at small message sizes. The fused paths below inline the entire
// GCM-SIV construction — key derivation, POLYVAL, tag finalization, AES-CTR —
// into one #[target_feature(enable = "aes,neon")] scope, matching the AEGIS
// pattern. The compiler keeps AES round keys and POLYVAL state in NEON
// registers across the whole operation.

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes,neon")]
unsafe fn encrypt_fused_aarch64(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 32],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  // SAFETY: caller has verified AES-CE availability.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    // --- 1. Expand derived encryption key ---
    let enc_ek = aes::aarch64_expand_key_inline(enc_key_bytes);
    ct::zeroize(enc_key_bytes);

    // --- 2. POLYVAL tag computation (RFC 8452 §5) ---
    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    // Precompute H powers only when at least one 4-block chunk will run.
    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

    // Process AAD in 4-block (64-byte) chunks.
    let mut offset = 0usize;
    while offset.strict_add(64) <= aad.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&aad[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::aarch64_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining AAD single blocks.
    while offset.strict_add(16) <= aad.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&aad[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::aarch64_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < aad.len() {
      let mut block = [0u8; 16];
      block[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::aarch64_clmul128_reduce_inline(acc, h);
    }

    // Process plaintext in 4-block (64-byte) chunks.
    offset = 0;
    while offset.strict_add(64) <= buffer.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&buffer[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::aarch64_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining plaintext single blocks.
    while offset.strict_add(16) <= buffer.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::aarch64_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < buffer.len() {
      let mut block = [0u8; 16];
      block[..buffer.len().strict_sub(offset)].copy_from_slice(&buffer[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::aarch64_clmul128_reduce_inline(acc, h);
    }

    // Length block: [aad_bits as u64 LE || pt_bits as u64 LE].
    let aad_bits = (aad.len() as u64).strict_mul(8);
    let pt_bits = (buffer.len() as u64).strict_mul(8);
    let mut length_block = [0u8; 16];
    length_block[0..8].copy_from_slice(&aad_bits.to_le_bytes());
    length_block[8..16].copy_from_slice(&pt_bits.to_le_bytes());
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::aarch64_clmul128_reduce_inline(acc, h);

    // --- 4. Tag finalization ---
    let mut tag = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      tag[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    tag[15] &= 0x7f;
    aes::aarch64_encrypt_block_inline(&enc_ek, &mut tag);

    // --- 5. AES-CTR encryption ---
    let mut counter_block = tag;
    counter_block[15] |= 0x80;
    let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
    offset = 0;
    while offset < buffer.len() {
      counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
      let mut keystream = counter_block;
      aes::aarch64_encrypt_block_inline(&enc_ek, &mut keystream);

      let remaining = buffer.len().strict_sub(offset);
      if remaining >= 16 {
        let mut d = [0u8; 16];
        d.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
        let xored = u128::from_ne_bytes(d) ^ u128::from_ne_bytes(keystream);
        buffer[offset..offset.strict_add(16)].copy_from_slice(&xored.to_ne_bytes());
        offset = offset.strict_add(16);
      } else {
        let mut i = 0usize;
        while i < remaining {
          buffer[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    tag
  }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes,neon")]
unsafe fn decrypt_fused_aarch64(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 32],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  tag: &Aes256GcmSivTag,
) -> Result<(), VerificationError> {
  // SAFETY: caller has verified AES-CE availability.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    // --- 1. Expand derived encryption key ---
    let enc_ek = aes::aarch64_expand_key_inline(enc_key_bytes);
    ct::zeroize(enc_key_bytes);

    // --- 2. AES-CTR decryption (SIV: decrypt before verify) ---
    let mut counter_block = tag.0;
    counter_block[15] |= 0x80;
    let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
    let mut offset = 0usize;
    while offset < buffer.len() {
      counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
      let mut keystream = counter_block;
      aes::aarch64_encrypt_block_inline(&enc_ek, &mut keystream);

      let remaining = buffer.len().strict_sub(offset);
      if remaining >= 16 {
        let mut d = [0u8; 16];
        d.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
        let xored = u128::from_ne_bytes(d) ^ u128::from_ne_bytes(keystream);
        buffer[offset..offset.strict_add(16)].copy_from_slice(&xored.to_ne_bytes());
        offset = offset.strict_add(16);
      } else {
        let mut i = 0usize;
        while i < remaining {
          buffer[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    // --- 3. POLYVAL tag computation over decrypted plaintext ---
    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    // Precompute H powers only when at least one 4-block chunk will run.
    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

    // Process AAD in 4-block (64-byte) chunks.
    offset = 0;
    while offset.strict_add(64) <= aad.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&aad[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::aarch64_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining AAD single blocks.
    while offset.strict_add(16) <= aad.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&aad[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::aarch64_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < aad.len() {
      let mut block = [0u8; 16];
      block[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::aarch64_clmul128_reduce_inline(acc, h);
    }

    // Process decrypted plaintext in 4-block (64-byte) chunks.
    offset = 0;
    while offset.strict_add(64) <= buffer.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&buffer[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::aarch64_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining plaintext single blocks.
    while offset.strict_add(16) <= buffer.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::aarch64_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < buffer.len() {
      let mut block = [0u8; 16];
      block[..buffer.len().strict_sub(offset)].copy_from_slice(&buffer[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::aarch64_clmul128_reduce_inline(acc, h);
    }

    // Length block.
    let aad_bits = (aad.len() as u64).strict_mul(8);
    let pt_bits = (buffer.len() as u64).strict_mul(8);
    let mut length_block = [0u8; 16];
    length_block[0..8].copy_from_slice(&aad_bits.to_le_bytes());
    length_block[8..16].copy_from_slice(&pt_bits.to_le_bytes());
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::aarch64_clmul128_reduce_inline(acc, h);

    // --- 4. Tag finalization + verification ---
    let mut expected = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      expected[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    expected[15] &= 0x7f;
    aes::aarch64_encrypt_block_inline(&enc_ek, &mut expected);

    if !ct::constant_time_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Err(VerificationError::new());
    }
    Ok(())
  }
}

// ---------------------------------------------------------------------------
// powerpc64 fused encrypt/decrypt (single #[target_feature] scope)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "powerpc64")]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
unsafe fn encrypt_fused_ppc(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 32],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  // SAFETY: caller has verified POWER8 crypto availability.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    // --- 1. Expand derived encryption key ---
    let enc_ek = aes::ppc_expand_key_inline(enc_key_bytes);
    ct::zeroize(enc_key_bytes);

    // --- 2. POLYVAL tag computation (RFC 8452 §5) ---
    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    // Precompute H powers only when at least one 4-block chunk will run.
    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

    // Process AAD in 4-block (64-byte) chunks.
    let mut offset = 0usize;
    while offset.strict_add(64) <= aad.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&aad[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::ppc_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining AAD single blocks.
    while offset.strict_add(16) <= aad.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&aad[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::ppc_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < aad.len() {
      let mut block = [0u8; 16];
      block[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::ppc_clmul128_reduce_inline(acc, h);
    }

    // Process plaintext in 4-block (64-byte) chunks.
    offset = 0;
    while offset.strict_add(64) <= buffer.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&buffer[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::ppc_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining plaintext single blocks.
    while offset.strict_add(16) <= buffer.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::ppc_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < buffer.len() {
      let mut block = [0u8; 16];
      block[..buffer.len().strict_sub(offset)].copy_from_slice(&buffer[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::ppc_clmul128_reduce_inline(acc, h);
    }

    // Length block: [aad_bits as u64 LE || pt_bits as u64 LE].
    let aad_bits = (aad.len() as u64).strict_mul(8);
    let pt_bits = (buffer.len() as u64).strict_mul(8);
    let mut length_block = [0u8; 16];
    length_block[0..8].copy_from_slice(&aad_bits.to_le_bytes());
    length_block[8..16].copy_from_slice(&pt_bits.to_le_bytes());
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::ppc_clmul128_reduce_inline(acc, h);

    // --- 4. Tag finalization ---
    let mut tag = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      tag[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    tag[15] &= 0x7f;
    aes::ppc_encrypt_block_inline(&enc_ek, &mut tag);

    // --- 5. AES-CTR encryption ---
    let mut counter_block = tag;
    counter_block[15] |= 0x80;
    let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
    offset = 0;
    while offset < buffer.len() {
      counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
      let mut keystream = counter_block;
      aes::ppc_encrypt_block_inline(&enc_ek, &mut keystream);

      let remaining = buffer.len().strict_sub(offset);
      if remaining >= 16 {
        let mut d = [0u8; 16];
        d.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
        let xored = u128::from_ne_bytes(d) ^ u128::from_ne_bytes(keystream);
        buffer[offset..offset.strict_add(16)].copy_from_slice(&xored.to_ne_bytes());
        offset = offset.strict_add(16);
      } else {
        let mut i = 0usize;
        while i < remaining {
          buffer[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    tag
  }
}

#[cfg(target_arch = "powerpc64")]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
unsafe fn decrypt_fused_ppc(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 32],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  tag: &Aes256GcmSivTag,
) -> Result<(), VerificationError> {
  // SAFETY: caller has verified POWER8 crypto availability.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    // --- 1. Expand derived encryption key ---
    let enc_ek = aes::ppc_expand_key_inline(enc_key_bytes);
    ct::zeroize(enc_key_bytes);

    // --- 2. AES-CTR decryption (SIV: decrypt before verify) ---
    let mut counter_block = tag.0;
    counter_block[15] |= 0x80;
    let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
    let mut offset = 0usize;
    while offset < buffer.len() {
      counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
      let mut keystream = counter_block;
      aes::ppc_encrypt_block_inline(&enc_ek, &mut keystream);

      let remaining = buffer.len().strict_sub(offset);
      if remaining >= 16 {
        let mut d = [0u8; 16];
        d.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
        let xored = u128::from_ne_bytes(d) ^ u128::from_ne_bytes(keystream);
        buffer[offset..offset.strict_add(16)].copy_from_slice(&xored.to_ne_bytes());
        offset = offset.strict_add(16);
      } else {
        let mut i = 0usize;
        while i < remaining {
          buffer[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    // --- 3. POLYVAL tag computation over decrypted plaintext ---
    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    // Precompute H powers only when at least one 4-block chunk will run.
    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

    // Process AAD in 4-block (64-byte) chunks.
    offset = 0;
    while offset.strict_add(64) <= aad.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&aad[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::ppc_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining AAD single blocks.
    while offset.strict_add(16) <= aad.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&aad[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::ppc_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < aad.len() {
      let mut block = [0u8; 16];
      block[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::ppc_clmul128_reduce_inline(acc, h);
    }

    // Process decrypted plaintext in 4-block (64-byte) chunks.
    offset = 0;
    while offset.strict_add(64) <= buffer.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&buffer[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::ppc_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining plaintext single blocks.
    while offset.strict_add(16) <= buffer.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::ppc_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < buffer.len() {
      let mut block = [0u8; 16];
      block[..buffer.len().strict_sub(offset)].copy_from_slice(&buffer[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::ppc_clmul128_reduce_inline(acc, h);
    }

    // Length block.
    let aad_bits = (aad.len() as u64).strict_mul(8);
    let pt_bits = (buffer.len() as u64).strict_mul(8);
    let mut length_block = [0u8; 16];
    length_block[0..8].copy_from_slice(&aad_bits.to_le_bytes());
    length_block[8..16].copy_from_slice(&pt_bits.to_le_bytes());
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::ppc_clmul128_reduce_inline(acc, h);

    // --- 4. Tag finalization + verification ---
    let mut expected = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      expected[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    expected[15] &= 0x7f;
    aes::ppc_encrypt_block_inline(&enc_ek, &mut expected);

    if !ct::constant_time_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Err(VerificationError::new());
    }
    Ok(())
  }
}

// ---------------------------------------------------------------------------
// s390x fused encrypt/decrypt (#[target_feature(enable = "vector")] for POLYVAL)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "s390x")]
#[target_feature(enable = "vector")]
unsafe fn encrypt_fused_s390x(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 32],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  // SAFETY: caller has verified z/Vector + MSA availability.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    // --- 1. Expand derived encryption key ---
    let enc_km = aes::s390x_expand_key_inline(enc_key_bytes);
    ct::zeroize(enc_key_bytes);

    // --- 2. POLYVAL tag computation (RFC 8452 §5) ---
    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    // Precompute H powers only when at least one 4-block chunk will run.
    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

    // Process AAD in 4-block (64-byte) chunks.
    let mut offset = 0usize;
    while offset.strict_add(64) <= aad.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&aad[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::s390x_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining AAD single blocks.
    while offset.strict_add(16) <= aad.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&aad[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::s390x_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < aad.len() {
      let mut block = [0u8; 16];
      block[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::s390x_clmul128_reduce_inline(acc, h);
    }

    // Process plaintext in 4-block (64-byte) chunks.
    offset = 0;
    while offset.strict_add(64) <= buffer.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&buffer[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::s390x_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining plaintext single blocks.
    while offset.strict_add(16) <= buffer.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::s390x_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < buffer.len() {
      let mut block = [0u8; 16];
      block[..buffer.len().strict_sub(offset)].copy_from_slice(&buffer[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::s390x_clmul128_reduce_inline(acc, h);
    }

    // Length block: [aad_bits as u64 LE || pt_bits as u64 LE].
    let aad_bits = (aad.len() as u64).strict_mul(8);
    let pt_bits = (buffer.len() as u64).strict_mul(8);
    let mut length_block = [0u8; 16];
    length_block[0..8].copy_from_slice(&aad_bits.to_le_bytes());
    length_block[8..16].copy_from_slice(&pt_bits.to_le_bytes());
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::s390x_clmul128_reduce_inline(acc, h);

    // --- 4. Tag finalization ---
    let mut tag = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      tag[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    tag[15] &= 0x7f;
    aes::s390x_encrypt_block_inline(&enc_km, &mut tag);

    // --- 5. AES-CTR encryption ---
    let mut counter_block = tag;
    counter_block[15] |= 0x80;
    let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
    offset = 0;
    while offset < buffer.len() {
      counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
      let mut keystream = counter_block;
      aes::s390x_encrypt_block_inline(&enc_km, &mut keystream);

      let remaining = buffer.len().strict_sub(offset);
      if remaining >= 16 {
        let mut d = [0u8; 16];
        d.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
        let xored = u128::from_ne_bytes(d) ^ u128::from_ne_bytes(keystream);
        buffer[offset..offset.strict_add(16)].copy_from_slice(&xored.to_ne_bytes());
        offset = offset.strict_add(16);
      } else {
        let mut i = 0usize;
        while i < remaining {
          buffer[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    tag
  }
}

#[cfg(target_arch = "s390x")]
#[target_feature(enable = "vector")]
unsafe fn decrypt_fused_s390x(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 32],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  tag: &Aes256GcmSivTag,
) -> Result<(), VerificationError> {
  // SAFETY: caller has verified z/Vector + MSA availability.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    // --- 1. Expand derived encryption key ---
    let enc_km = aes::s390x_expand_key_inline(enc_key_bytes);
    ct::zeroize(enc_key_bytes);

    // --- 2. AES-CTR decryption (SIV: decrypt before verify) ---
    let mut counter_block = tag.0;
    counter_block[15] |= 0x80;
    let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
    let mut offset = 0usize;
    while offset < buffer.len() {
      counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
      let mut keystream = counter_block;
      aes::s390x_encrypt_block_inline(&enc_km, &mut keystream);

      let remaining = buffer.len().strict_sub(offset);
      if remaining >= 16 {
        let mut d = [0u8; 16];
        d.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
        let xored = u128::from_ne_bytes(d) ^ u128::from_ne_bytes(keystream);
        buffer[offset..offset.strict_add(16)].copy_from_slice(&xored.to_ne_bytes());
        offset = offset.strict_add(16);
      } else {
        let mut i = 0usize;
        while i < remaining {
          buffer[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    // --- 3. POLYVAL tag computation over decrypted plaintext ---
    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    // Precompute H powers only when at least one 4-block chunk will run.
    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

    // Process AAD in 4-block (64-byte) chunks.
    offset = 0;
    while offset.strict_add(64) <= aad.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&aad[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::s390x_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining AAD single blocks.
    while offset.strict_add(16) <= aad.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&aad[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::s390x_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < aad.len() {
      let mut block = [0u8; 16];
      block[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::s390x_clmul128_reduce_inline(acc, h);
    }

    // Process decrypted plaintext in 4-block (64-byte) chunks.
    offset = 0;
    while offset.strict_add(64) <= buffer.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&buffer[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::s390x_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining plaintext single blocks.
    while offset.strict_add(16) <= buffer.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::s390x_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < buffer.len() {
      let mut block = [0u8; 16];
      block[..buffer.len().strict_sub(offset)].copy_from_slice(&buffer[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::s390x_clmul128_reduce_inline(acc, h);
    }

    // Length block.
    let aad_bits = (aad.len() as u64).strict_mul(8);
    let pt_bits = (buffer.len() as u64).strict_mul(8);
    let mut length_block = [0u8; 16];
    length_block[0..8].copy_from_slice(&aad_bits.to_le_bytes());
    length_block[8..16].copy_from_slice(&pt_bits.to_le_bytes());
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::s390x_clmul128_reduce_inline(acc, h);

    // --- 4. Tag finalization + verification ---
    let mut expected = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      expected[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    expected[15] &= 0x7f;
    aes::s390x_encrypt_block_inline(&enc_km, &mut expected);

    if !ct::constant_time_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Err(VerificationError::new());
    }
    Ok(())
  }
}

// ---------------------------------------------------------------------------
// Aead trait implementation
// ---------------------------------------------------------------------------

impl Aead for Aes256GcmSiv {
  const KEY_SIZE: usize = KEY_SIZE;
  const NONCE_SIZE: usize = NONCE_SIZE;
  const TAG_SIZE: usize = TAG_SIZE;

  type Key = Aes256GcmSivKey;
  type Nonce = Nonce96;
  type Tag = Aes256GcmSivTag;

  fn new(key: &Self::Key) -> Self {
    Self {
      master_ek: aes::aes256_expand_key(key.as_bytes()),
    }
  }

  fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
    if bytes.len() != TAG_SIZE {
      return Err(AeadBufferError::new());
    }
    let mut tag = [0u8; TAG_SIZE];
    tag.copy_from_slice(bytes);
    Ok(Aes256GcmSivTag::from_bytes(tag))
  }

  fn encrypt_in_place(&self, nonce: &Self::Nonce, aad: &[u8], buffer: &mut [u8]) -> Self::Tag {
    assert!(
      (buffer.len() as u64) <= MAX_PLAINTEXT_LEN,
      "AES-256-GCM-SIV plaintext exceeds 2^36 - 32 bytes"
    );

    // Wide path: VPCLMULQDQ POLYVAL + VAES-512 CTR when available.
    #[cfg(target_arch = "x86_64")]
    {
      use crate::platform::caps;
      let c = crate::platform::caps();
      if c.has(caps::x86::VAES_READY) && c.has(caps::x86::VPCLMUL_READY) {
        let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
        let ek = aes::aes256_expand_key(&enc_key);
        let tag_bytes = compute_tag_wide(&auth_key, &ek, nonce, aad, buffer);
        let mut counter_block = tag_bytes;
        counter_block[15] |= 0x80;
        // SAFETY: VAES availability verified via CPUID.
        unsafe { aes::aes256_ctr32_encrypt_wide(&ek, &counter_block, buffer) };
        ct::zeroize(&mut auth_key);
        ct::zeroize(&mut enc_key);
        return Aes256GcmSivTag::from_bytes(tag_bytes);
      }
    }

    // Fused path: entire encrypt in a single #[target_feature] scope.
    #[cfg(target_arch = "aarch64")]
    {
      if crate::platform::caps().has(crate::platform::caps::aarch64::AES) {
        let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
        // SAFETY: AES-CE availability verified via HWCAP.
        let tag_bytes = unsafe { encrypt_fused_aarch64(&mut auth_key, &mut enc_key, nonce, aad, buffer) };
        return Aes256GcmSivTag::from_bytes(tag_bytes);
      }
    }

    // Fused path: POWER8 crypto.
    #[cfg(target_arch = "powerpc64")]
    {
      if crate::platform::caps().has(crate::platform::caps::power::POWER8_CRYPTO) {
        let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
        // SAFETY: POWER8 crypto availability verified via HWCAP.
        let tag_bytes = unsafe { encrypt_fused_ppc(&mut auth_key, &mut enc_key, nonce, aad, buffer) };
        return Aes256GcmSivTag::from_bytes(tag_bytes);
      }
    }

    // Fused path: s390x z/Vector + MSA.
    #[cfg(target_arch = "s390x")]
    {
      let c = crate::platform::caps();
      if c.has(crate::platform::caps::s390x::VECTOR) && c.has(crate::platform::caps::s390x::MSA) {
        let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
        // SAFETY: z/Vector + MSA availability verified via STFLE/HWCAP.
        let tag_bytes = unsafe { encrypt_fused_s390x(&mut auth_key, &mut enc_key, nonce, aad, buffer) };
        return Aes256GcmSivTag::from_bytes(tag_bytes);
      }
    }

    // Scalar path.
    let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
    let ek = aes::aes256_expand_key(&enc_key);
    let tag_bytes = compute_tag(&auth_key, &ek, nonce, aad, buffer);
    let mut counter_block = tag_bytes;
    counter_block[15] |= 0x80;
    aes::aes256_ctr32_encrypt(&ek, &counter_block, buffer);

    ct::zeroize(&mut auth_key);
    ct::zeroize(&mut enc_key);

    Aes256GcmSivTag::from_bytes(tag_bytes)
  }

  fn decrypt_in_place(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), VerificationError> {
    assert!(
      (buffer.len() as u64) <= MAX_PLAINTEXT_LEN,
      "AES-256-GCM-SIV ciphertext exceeds 2^36 - 32 bytes"
    );

    // Wide path: VAES-512 CTR + VPCLMULQDQ POLYVAL when available.
    #[cfg(target_arch = "x86_64")]
    {
      use crate::platform::caps;
      let c = crate::platform::caps();
      if c.has(caps::x86::VAES_READY) && c.has(caps::x86::VPCLMUL_READY) {
        let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
        let ek = aes::aes256_expand_key(&enc_key);
        // Decrypt first (SIV pattern).
        let mut counter_block = tag.0;
        counter_block[15] |= 0x80;
        // SAFETY: VAES availability verified via CPUID.
        unsafe { aes::aes256_ctr32_encrypt_wide(&ek, &counter_block, buffer) };

        // Verify tag over decrypted plaintext.
        let expected = compute_tag_wide(&auth_key, &ek, nonce, aad, buffer);
        ct::zeroize(&mut auth_key);
        ct::zeroize(&mut enc_key);
        if !ct::constant_time_eq(&expected, tag.as_bytes()) {
          ct::zeroize(buffer);
          return Err(VerificationError::new());
        }
        return Ok(());
      }
    }

    // Fused path: entire decrypt in a single #[target_feature] scope.
    #[cfg(target_arch = "aarch64")]
    {
      if crate::platform::caps().has(crate::platform::caps::aarch64::AES) {
        let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
        // SAFETY: AES-CE availability verified via HWCAP.
        return unsafe { decrypt_fused_aarch64(&mut auth_key, &mut enc_key, nonce, aad, buffer, tag) };
      }
    }

    // Fused path: POWER8 crypto.
    #[cfg(target_arch = "powerpc64")]
    {
      if crate::platform::caps().has(crate::platform::caps::power::POWER8_CRYPTO) {
        let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
        // SAFETY: POWER8 crypto availability verified via HWCAP.
        return unsafe { decrypt_fused_ppc(&mut auth_key, &mut enc_key, nonce, aad, buffer, tag) };
      }
    }

    // Fused path: s390x z/Vector + MSA.
    #[cfg(target_arch = "s390x")]
    {
      let c = crate::platform::caps();
      if c.has(crate::platform::caps::s390x::VECTOR) && c.has(crate::platform::caps::s390x::MSA) {
        let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
        // SAFETY: z/Vector + MSA availability verified via STFLE/HWCAP.
        return unsafe { decrypt_fused_s390x(&mut auth_key, &mut enc_key, nonce, aad, buffer, tag) };
      }
    }

    // Scalar path: decrypt then verify.
    let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
    let ek = aes::aes256_expand_key(&enc_key);
    let mut counter_block = tag.0;
    counter_block[15] |= 0x80;
    aes::aes256_ctr32_encrypt(&ek, &counter_block, buffer);

    let expected = compute_tag(&auth_key, &ek, nonce, aad, buffer);
    ct::zeroize(&mut auth_key);
    ct::zeroize(&mut enc_key);

    if !ct::constant_time_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Err(VerificationError::new());
    }

    Ok(())
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  /// RFC 8452 Appendix C.2, test case 1: empty plaintext, empty AAD.
  #[test]
  fn aes256gcmsiv_empty() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let expected_ct_tag = hex_vec("07f5f4169bbf55a8400cd47ea6fd400f");

    let cipher = Aes256GcmSiv::new(&key);
    let mut out = vec![0u8; expected_ct_tag.len()];
    cipher.encrypt(&nonce, &[], &[], &mut out).unwrap();
    assert_eq!(out, expected_ct_tag);

    // Decrypt.
    let mut pt_out = vec![0u8; 0];
    cipher.decrypt(&nonce, &[], &expected_ct_tag, &mut pt_out).unwrap();
    assert!(pt_out.is_empty());
  }

  /// RFC 8452 Appendix C.2: AAD=01, plaintext=0200000000000000.
  #[test]
  fn aes256gcmsiv_aad_and_plaintext() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let aad = hex_vec("01");
    let plaintext = hex_vec("0200000000000000");
    let expected_ct_tag = hex_vec("1de22967237a813291213f267e3b452f02d01ae33e4ec854");

    let cipher = Aes256GcmSiv::new(&key);

    // Encrypt.
    let mut out = vec![0u8; plaintext.len().strict_add(TAG_SIZE)];
    cipher.encrypt(&nonce, &aad, &plaintext, &mut out).unwrap();
    assert_eq!(out, expected_ct_tag);

    // Decrypt.
    let mut pt_out = vec![0u8; plaintext.len()];
    cipher.decrypt(&nonce, &aad, &expected_ct_tag, &mut pt_out).unwrap();
    assert_eq!(pt_out, plaintext);
  }

  /// RFC 8452 Appendix C.2: longer AAD and plaintext.
  #[test]
  fn aes256gcmsiv_longer_aad_and_plaintext() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let aad = hex_vec("010000000000000000000000000000000200");
    let plaintext = hex_vec("0300000000000000000000000000000004000000");
    let expected_ct_tag = hex_vec("43dd0163cdb48f9fe3212bf61b201976067f342bb879ad976d8242acc188ab59cabfe307");

    let cipher = Aes256GcmSiv::new(&key);

    let mut out = vec![0u8; plaintext.len().strict_add(TAG_SIZE)];
    cipher.encrypt(&nonce, &aad, &plaintext, &mut out).unwrap();
    assert_eq!(out, expected_ct_tag);

    let mut pt_out = vec![0u8; plaintext.len()];
    cipher.decrypt(&nonce, &aad, &expected_ct_tag, &mut pt_out).unwrap();
    assert_eq!(pt_out, plaintext);
  }

  /// Decryption with wrong tag should fail.
  #[test]
  fn aes256gcmsiv_bad_tag() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let mut bad_ct_tag = hex_vec("07f5f4169bbf55a8400cd47ea6fd400f");
    // Flip a bit in the tag.
    bad_ct_tag[0] ^= 1;

    let cipher = Aes256GcmSiv::new(&key);
    let mut pt_out = vec![0u8; 0];
    let result = cipher.decrypt(&nonce, &[], &bad_ct_tag, &mut pt_out);
    assert!(result.is_err());
  }

  /// Decryption with wrong AAD should fail.
  #[test]
  fn aes256gcmsiv_wrong_aad_rejected() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let ct_tag = hex_vec("1de22967237a813291213f267e3b452f02d01ae33e4ec854");

    let cipher = Aes256GcmSiv::new(&key);
    let mut pt_out = vec![0u8; 8]; // plaintext was 8 bytes
    // Wrong AAD: 0x02 instead of 0x01.
    let result = cipher.decrypt(&nonce, &[0x02], &ct_tag, &mut pt_out);
    assert!(result.is_err());
  }

  /// Decryption with wrong nonce should fail.
  #[test]
  fn aes256gcmsiv_wrong_nonce_rejected() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let aad = hex_vec("01");
    let ct_tag = hex_vec("1de22967237a813291213f267e3b452f02d01ae33e4ec854");

    let cipher = Aes256GcmSiv::new(&key);
    let mut pt_out = vec![0u8; 8]; // plaintext was 8 bytes
    // Wrong nonce: 0x04 instead of 0x03.
    let wrong_nonce = Nonce96::from_bytes(hex12("040000000000000000000000"));
    let result = cipher.decrypt(&wrong_nonce, &aad, &ct_tag, &mut pt_out);
    assert!(result.is_err());
  }

  /// Ciphertext body tampering should fail verification.
  #[test]
  fn aes256gcmsiv_ciphertext_tampering_rejected() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let aad = hex_vec("01");
    let plaintext = hex_vec("0200000000000000");
    let mut ct_tag = hex_vec("1de22967237a813291213f267e3b452f02d01ae33e4ec854");

    // Flip a bit in the ciphertext body (not the tag).
    ct_tag[0] ^= 1;

    let cipher = Aes256GcmSiv::new(&key);
    let mut pt_out = vec![0u8; plaintext.len()];
    let result = cipher.decrypt(&nonce, &aad, &ct_tag, &mut pt_out);
    assert!(result.is_err());
  }

  /// On authentication failure, the output buffer must be zeroed.
  #[test]
  fn aes256gcmsiv_buffer_zeroed_on_auth_failure() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let aad = hex_vec("01");
    let plaintext = hex_vec("0200000000000000");

    let cipher = Aes256GcmSiv::new(&key);
    let mut out = vec![0u8; plaintext.len().strict_add(TAG_SIZE)];
    cipher.encrypt(&nonce, &aad, &plaintext, &mut out).unwrap();

    // Corrupt the tag.
    let last = out.len().strict_sub(1);
    out[last] ^= 0xff;

    let mut pt_out = vec![0xffu8; plaintext.len()]; // fill with non-zero
    let result = cipher.decrypt(&nonce, &aad, &out, &mut pt_out);
    assert!(result.is_err());
    // Buffer must be zeroed even though decryption was attempted.
    assert!(pt_out.iter().all(|&b| b == 0), "buffer not zeroed on auth failure");
  }

  /// Detached encrypt/decrypt round-trip.
  #[test]
  fn aes256gcmsiv_detached_round_trip() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let aad = hex_vec("01");
    let plaintext = hex_vec("0200000000000000");

    let cipher = Aes256GcmSiv::new(&key);

    let mut buf = plaintext.clone();
    let tag = cipher.encrypt_in_place(&nonce, &aad, &mut buf);

    // buf is now ciphertext, tag is separate.
    assert_ne!(buf, plaintext);

    cipher.decrypt_in_place(&nonce, &aad, &mut buf, &tag).unwrap();
    assert_eq!(buf, plaintext);
  }

  /// `tag_from_slice` rejects wrong-length input.
  #[test]
  fn aes256gcmsiv_tag_from_slice_rejects_bad_length() {
    assert!(Aes256GcmSiv::tag_from_slice(&[0u8; 15]).is_err());
    assert!(Aes256GcmSiv::tag_from_slice(&[0u8; 17]).is_err());
    assert!(Aes256GcmSiv::tag_from_slice(&[0u8; 0]).is_err());
    assert!(Aes256GcmSiv::tag_from_slice(&[0u8; 16]).is_ok());
  }

  /// RFC 8452 Appendix C.2 vector with a different key (empty PT+AAD).
  #[test]
  fn aes256gcmsiv_different_key_vector() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "e66021d5eb8e4f4066d4adb9c33560e4f46e44bb3da0015c94f7088736864200",
    ));
    let nonce = Nonce96::from_bytes(hex12("e0eaf5284d884a0e77d31646"));
    let expected_ct_tag = hex_vec("169fbb2fbf389a995f6390af22228a62");

    let cipher = Aes256GcmSiv::new(&key);

    let mut out = vec![0u8; expected_ct_tag.len()];
    cipher.encrypt(&nonce, &[], &[], &mut out).unwrap();
    assert_eq!(out, expected_ct_tag);

    let mut pt_out = vec![0u8; 0];
    cipher.decrypt(&nonce, &[], &expected_ct_tag, &mut pt_out).unwrap();
  }

  // --- Hex helpers ---

  fn hex32(hex: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..32 {
      out[i] = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
    }
    out
  }

  fn hex12(hex: &str) -> [u8; 12] {
    let mut out = [0u8; 12];
    for i in 0..12 {
      out[i] = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
    }
    out
  }

  fn hex_vec(hex: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(hex.len() / 2);
    let mut i = 0;
    while i < hex.len() {
      out.push(u8::from_str_radix(&hex[i..i + 2], 16).unwrap());
      i += 2;
    }
    out
  }
}
