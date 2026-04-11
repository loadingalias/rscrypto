//! HKDF-SHA256 (RFC 5869).

use core::fmt;

use super::hmac::HmacSha256;
use crate::{
  hashes::crypto::sha256::{H0 as SHA256_H0, dispatch as sha256_dispatch, kernels::CompressBlocksFn},
  traits::ct,
};

const OUTPUT_SIZE: usize = 32;
const MAX_OUTPUT_SIZE: usize = 255 * OUTPUT_SIZE;
const BLOCK_SIZE: usize = 64;

/// HKDF requested more output than RFC 5869 allows for a single expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HkdfOutputLengthError;

impl HkdfOutputLengthError {
  /// Construct a new output-length error.
  #[inline]
  #[must_use]
  pub const fn new() -> Self {
    Self
  }
}

impl Default for HkdfOutputLengthError {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl fmt::Display for HkdfOutputLengthError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("requested HKDF output exceeds 8160 bytes")
  }
}

impl core::error::Error for HkdfOutputLengthError {}

/// HKDF-SHA256 pseudorandom key state.
///
/// `new()` and `extract()` perform HKDF-Extract and store the pseudorandom key
/// for later `expand()` calls.
///
/// # Examples
///
/// ```rust
/// use rscrypto::HkdfSha256;
///
/// let hkdf = HkdfSha256::new(b"salt", b"input key material");
///
/// let mut okm = [0u8; 42];
/// hkdf.expand(b"context", &mut okm)?;
///
/// let oneshot = HkdfSha256::derive_array::<42>(b"salt", b"input key material", b"context")?;
/// assert_eq!(okm, oneshot);
/// # Ok::<(), rscrypto::auth::HkdfOutputLengthError>(())
/// ```
#[derive(Clone)]
pub struct HkdfSha256 {
  prk: [u8; OUTPUT_SIZE],
  /// SHA-256 state after compressing H0 + (PRK ⊕ 0x36…36). Cached to avoid
  /// re-compressing the ipad block on every expand iteration.
  inner_init: [u32; 8],
  /// SHA-256 state after compressing H0 + (PRK ⊕ 0x5c…5c). Cached to avoid
  /// re-compressing the opad block on every expand iteration.
  outer_init: [u32; 8],
  /// Resolved SHA-256 compress function (from dispatch, cached).
  compress: CompressBlocksFn,
}

impl fmt::Debug for HkdfSha256 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("HkdfSha256").finish_non_exhaustive()
  }
}

impl HkdfSha256 {
  /// HKDF-SHA256 pseudorandom key size in bytes.
  pub const OUTPUT_SIZE: usize = OUTPUT_SIZE;

  /// Maximum RFC 5869 expand output size in bytes.
  pub const MAX_OUTPUT_SIZE: usize = MAX_OUTPUT_SIZE;

  /// Perform HKDF-Extract with `salt` and `input_key_material`.
  #[inline]
  #[must_use]
  pub fn new(salt: &[u8], input_key_material: &[u8]) -> Self {
    Self::extract(salt, input_key_material)
  }

  /// Perform HKDF-Extract with `salt` and `input_key_material`.
  #[must_use]
  pub fn extract(salt: &[u8], input_key_material: &[u8]) -> Self {
    let zero_salt = [0u8; OUTPUT_SIZE];
    let salt = if salt.is_empty() { &zero_salt[..] } else { salt };
    let prk = HmacSha256::mac(salt, input_key_material);

    // Build ipad/opad key blocks and compress to cached states.
    // PRK is always OUTPUT_SIZE (32) bytes, which is < BLOCK_SIZE (64).
    let compress = sha256_dispatch::compress_dispatch().select(0);

    let mut key_block = [0u8; BLOCK_SIZE];
    key_block[..OUTPUT_SIZE].copy_from_slice(&prk);

    let mut ipad = [0u8; BLOCK_SIZE];
    let mut opad = [0u8; BLOCK_SIZE];
    for ((ip, op), &kb) in ipad.iter_mut().zip(opad.iter_mut()).zip(key_block.iter()) {
      *ip = kb ^ 0x36;
      *op = kb ^ 0x5c;
    }

    let mut inner_init = SHA256_H0;
    compress(&mut inner_init, &ipad);

    let mut outer_init = SHA256_H0;
    compress(&mut outer_init, &opad);

    ct::zeroize(&mut key_block);
    ct::zeroize(&mut ipad);
    ct::zeroize(&mut opad);

    Self {
      prk,
      inner_init,
      outer_init,
      compress,
    }
  }

  /// Expand the stored pseudorandom key into `okm`.
  ///
  /// Uses raw SHA-256 state arrays and a single cached compress function,
  /// bypassing all `Sha256` struct creation, `Drop` zeroization, and dispatch
  /// overhead in the inner loop.
  #[allow(clippy::indexing_slicing)]
  pub fn expand(&self, info: &[u8], okm: &mut [u8]) -> Result<(), HkdfOutputLengthError> {
    if okm.len() > MAX_OUTPUT_SIZE {
      return Err(HkdfOutputLengthError::new());
    }

    if okm.is_empty() {
      return Ok(());
    }

    let compress = self.compress;
    let inner_init = self.inner_init;
    let outer_init = self.outer_init;

    // Pre-build the outer hash block template. Only bytes [0..32] change
    // per iteration; the padding byte and length field are invariant.
    let mut outer_block = [0u8; BLOCK_SIZE];
    outer_block[OUTPUT_SIZE] = 0x80;
    // Total outer: opad(64) + inner_hash(32) = 96 bytes → 768 bits.
    outer_block[56..BLOCK_SIZE].copy_from_slice(&768u64.to_be_bytes());

    let mut prev_tag = [0u8; OUTPUT_SIZE];
    let mut inner_hash = [0u8; OUTPUT_SIZE];
    let mut state = [0u32; 8];
    let mut counter: u8 = 1;

    let mut chunks = okm.chunks_mut(OUTPUT_SIZE);

    // First block: HMAC(PRK, info ∥ 0x01) — no previous tag.
    let Some(first) = chunks.next() else {
      return Ok(());
    };
    expand_hmac_inner(compress, &inner_init, None, info, counter, &mut state, &mut inner_hash);
    expand_hmac_outer(
      compress,
      &outer_init,
      &inner_hash,
      &mut state,
      &mut outer_block,
      &mut prev_tag,
    );
    first.copy_from_slice(&prev_tag[..first.len()]);
    counter = counter.wrapping_add(1);

    // Subsequent blocks: HMAC(PRK, T(i-1) ∥ info ∥ i).
    for chunk in chunks {
      expand_hmac_inner(
        compress,
        &inner_init,
        Some(&prev_tag),
        info,
        counter,
        &mut state,
        &mut inner_hash,
      );
      expand_hmac_outer(
        compress,
        &outer_init,
        &inner_hash,
        &mut state,
        &mut outer_block,
        &mut prev_tag,
      );
      chunk.copy_from_slice(&prev_tag[..chunk.len()]);
      counter = counter.wrapping_add(1);
    }

    // Zeroize all sensitive temporaries once (not per iteration).
    ct::zeroize(&mut prev_tag);
    ct::zeroize(&mut inner_hash);
    ct::zeroize(&mut outer_block);
    for word in state.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);

    Ok(())
  }

  /// Expand into a fixed-size array.
  pub fn expand_array<const N: usize>(&self, info: &[u8]) -> Result<[u8; N], HkdfOutputLengthError> {
    let mut out = [0u8; N];
    self.expand(info, &mut out)?;
    Ok(out)
  }

  /// Perform HKDF-Extract and HKDF-Expand in one shot.
  #[inline]
  pub fn derive(
    salt: &[u8],
    input_key_material: &[u8],
    info: &[u8],
    okm: &mut [u8],
  ) -> Result<(), HkdfOutputLengthError> {
    Self::new(salt, input_key_material).expand(info, okm)
  }

  /// Perform HKDF-Extract and HKDF-Expand into a fixed-size array.
  #[inline]
  pub fn derive_array<const N: usize>(
    salt: &[u8],
    input_key_material: &[u8],
    info: &[u8],
  ) -> Result<[u8; N], HkdfOutputLengthError> {
    Self::new(salt, input_key_material).expand_array(info)
  }

  /// Return the extracted pseudorandom key bytes.
  #[inline]
  #[must_use]
  pub fn prk(&self) -> &[u8; OUTPUT_SIZE] {
    &self.prk
  }
}

impl Drop for HkdfSha256 {
  fn drop(&mut self) {
    ct::zeroize(&mut self.prk);
    for word in self.inner_init.iter_mut().chain(self.outer_init.iter_mut()) {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Raw HMAC helpers — bypass streaming Sha256/HmacSha256 for zero struct
// allocation and zero Drop overhead in the expand hot loop.
// ─────────────────────────────────────────────────────────────────────────────

/// Inner SHA-256 for one HKDF-Expand HMAC step.
///
/// Computes SHA-256(inner_init_state ∥ \[prev ∥\] info ∥ counter), where
/// `inner_init_state` already includes the ipad block compression.
#[inline(always)]
#[allow(clippy::indexing_slicing)]
fn expand_hmac_inner(
  compress: CompressBlocksFn,
  inner_init: &[u32; 8],
  prev: Option<&[u8; OUTPUT_SIZE]>,
  info: &[u8],
  counter: u8,
  state: &mut [u32; 8],
  out: &mut [u8; OUTPUT_SIZE],
) {
  *state = *inner_init;

  let prev_len = if prev.is_some() { OUTPUT_SIZE } else { 0 };
  let msg_len = prev_len.strict_add(info.len()).strict_add(1);
  let total_bytes = (BLOCK_SIZE as u64).strict_add(msg_len as u64);

  let mut block = [0u8; BLOCK_SIZE];
  let mut pos = 0usize;

  // Feed previous HMAC output (only for blocks after the first).
  if let Some(prev) = prev {
    block[..OUTPUT_SIZE].copy_from_slice(prev);
    pos = OUTPUT_SIZE;
  }

  // Feed info bytes.
  let mut info_off = 0usize;
  while info_off < info.len() {
    let space = BLOCK_SIZE.strict_sub(pos);
    let remaining = info.len().strict_sub(info_off);
    let take = if space < remaining { space } else { remaining };
    block[pos..pos.strict_add(take)].copy_from_slice(&info[info_off..info_off.strict_add(take)]);
    pos = pos.strict_add(take);
    info_off = info_off.strict_add(take);
    if pos == BLOCK_SIZE {
      compress(state, &block);
      block = [0u8; BLOCK_SIZE];
      pos = 0;
    }
  }

  // Feed counter byte.
  block[pos] = counter;
  pos = pos.strict_add(1);
  if pos == BLOCK_SIZE {
    compress(state, &block);
    block = [0u8; BLOCK_SIZE];
    pos = 0;
  }

  // SHA-256 padding.
  block[pos] = 0x80;
  if pos.strict_add(1) > 56 {
    compress(state, &block);
    block = [0u8; BLOCK_SIZE];
  }
  block[56..BLOCK_SIZE].copy_from_slice(&total_bytes.strict_mul(8).to_be_bytes());
  compress(state, &block);

  // Serialize state → bytes.
  for (dst, &word) in out.chunks_exact_mut(4).zip(state.iter()) {
    dst.copy_from_slice(&word.to_be_bytes());
  }
}

/// Outer SHA-256 for one HKDF-Expand HMAC step.
///
/// Fixed structure: SHA-256(outer\_init\_state ∥ inner\_hash) with total
/// 96 bytes. Always exactly one final block since |inner\_hash| = 32 < 56.
#[inline(always)]
#[allow(clippy::indexing_slicing)]
fn expand_hmac_outer(
  compress: CompressBlocksFn,
  outer_init: &[u32; 8],
  inner_hash: &[u8; OUTPUT_SIZE],
  state: &mut [u32; 8],
  outer_block: &mut [u8; BLOCK_SIZE],
  out: &mut [u8; OUTPUT_SIZE],
) {
  *state = *outer_init;
  outer_block[..OUTPUT_SIZE].copy_from_slice(inner_hash);
  // Bytes [32..64] are pre-set by the caller: 0x80 at [32], zeros at
  // [33..56], and 768u64 big-endian at [56..64]. Invariant across iterations.
  compress(state, outer_block);

  for (dst, &word) in out.chunks_exact_mut(4).zip(state.iter()) {
    dst.copy_from_slice(&word.to_be_bytes());
  }
}
