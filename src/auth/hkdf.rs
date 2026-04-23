//! HKDF-SHA2 family (RFC 5869).

use core::fmt;

use super::hmac::{HmacSha256, HmacSha384, hmac_prefix_state};
use crate::{
  hashes::crypto::{
    sha256::{H0 as SHA256_H0, dispatch as sha256_dispatch, kernels::CompressBlocksFn as Sha256CompressBlocksFn},
    sha384::{H0 as SHA384_H0, dispatch as sha384_dispatch, kernels::CompressBlocksFn as Sha512FamilyCompressBlocksFn},
  },
  traits::ct,
};

const SHA256_OUTPUT_SIZE: usize = 32;
const SHA256_MAX_OUTPUT_SIZE: usize = 255 * SHA256_OUTPUT_SIZE;
const SHA256_BLOCK_SIZE: usize = 64;

const SHA384_OUTPUT_SIZE: usize = 48;
const SHA384_MAX_OUTPUT_SIZE: usize = 255 * SHA384_OUTPUT_SIZE;
const SHA384_BLOCK_SIZE: usize = 128;

define_unit_error! {
  /// HKDF requested more output than RFC 5869 allows for a single expansion.
  pub struct HkdfOutputLengthError;
  "requested HKDF output exceeds the algorithm maximum"
}

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
  prk: [u8; SHA256_OUTPUT_SIZE],
  inner_init: [u32; 8],
  outer_init: [u32; 8],
  compress: Sha256CompressBlocksFn,
}

impl fmt::Debug for HkdfSha256 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("HkdfSha256").finish_non_exhaustive()
  }
}

impl HkdfSha256 {
  /// HKDF-SHA256 pseudorandom key size in bytes.
  pub const OUTPUT_SIZE: usize = SHA256_OUTPUT_SIZE;

  /// Maximum RFC 5869 expand output size in bytes.
  pub const MAX_OUTPUT_SIZE: usize = SHA256_MAX_OUTPUT_SIZE;

  /// Perform HKDF-Extract with `salt` and `input_key_material`.
  #[inline]
  #[must_use]
  pub fn new(salt: &[u8], input_key_material: &[u8]) -> Self {
    Self::extract(salt, input_key_material)
  }

  /// Perform HKDF-Extract with `salt` and `input_key_material`.
  #[must_use]
  pub fn extract(salt: &[u8], input_key_material: &[u8]) -> Self {
    let zero_salt = [0u8; SHA256_OUTPUT_SIZE];
    let salt = if salt.is_empty() { &zero_salt[..] } else { salt };
    let prk = HmacSha256::mac(salt, input_key_material);

    let compress = sha256_dispatch::compress_dispatch().select(0);

    let mut key_block = [0u8; SHA256_BLOCK_SIZE];
    key_block[..SHA256_OUTPUT_SIZE].copy_from_slice(&prk);

    let (inner_init, outer_init) = hmac_prefix_state(&mut key_block, |ipad, opad| {
      let mut inner_init = SHA256_H0;
      compress(&mut inner_init, ipad);

      let mut outer_init = SHA256_H0;
      compress(&mut outer_init, opad);

      (inner_init, outer_init)
    });

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
    if okm.len() > SHA256_MAX_OUTPUT_SIZE {
      return Err(HkdfOutputLengthError::new());
    }

    if okm.is_empty() {
      return Ok(());
    }

    let compress = self.compress;
    let inner_init = self.inner_init;
    let outer_init = self.outer_init;

    let mut outer_block = [0u8; SHA256_BLOCK_SIZE];
    outer_block[SHA256_OUTPUT_SIZE] = 0x80;
    outer_block[56..SHA256_BLOCK_SIZE].copy_from_slice(&768u64.to_be_bytes());

    let mut prev_tag = [0u8; SHA256_OUTPUT_SIZE];
    let mut inner_hash = [0u8; SHA256_OUTPUT_SIZE];
    let mut state = [0u32; 8];
    let mut counter: u8 = 1;

    let mut chunks = okm.chunks_mut(SHA256_OUTPUT_SIZE);
    let Some(first) = chunks.next() else {
      return Ok(());
    };

    expand_hmac_sha256_inner(compress, &inner_init, None, info, counter, &mut state, &mut inner_hash);
    expand_hmac_sha256_outer(
      compress,
      &outer_init,
      &inner_hash,
      &mut state,
      &mut outer_block,
      &mut prev_tag,
    );
    first.copy_from_slice(&prev_tag[..first.len()]);
    counter = counter.wrapping_add(1);

    for chunk in chunks {
      expand_hmac_sha256_inner(
        compress,
        &inner_init,
        Some(&prev_tag),
        info,
        counter,
        &mut state,
        &mut inner_hash,
      );
      expand_hmac_sha256_outer(
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
  pub fn prk(&self) -> &[u8; SHA256_OUTPUT_SIZE] {
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

/// HKDF-SHA384 pseudorandom key state.
///
/// `new()` and `extract()` perform HKDF-Extract and store the pseudorandom key
/// for later `expand()` calls.
///
/// # Examples
///
/// ```rust
/// use rscrypto::HkdfSha384;
///
/// let hkdf = HkdfSha384::new(b"salt", b"input key material");
///
/// let mut okm = [0u8; 48];
/// hkdf.expand(b"context", &mut okm)?;
///
/// let oneshot = HkdfSha384::derive_array::<48>(b"salt", b"input key material", b"context")?;
/// assert_eq!(okm, oneshot);
/// # Ok::<(), rscrypto::auth::HkdfOutputLengthError>(())
/// ```
#[derive(Clone)]
pub struct HkdfSha384 {
  prk: [u8; SHA384_OUTPUT_SIZE],
  inner_init: [u64; 8],
  outer_init: [u64; 8],
  compress: Sha512FamilyCompressBlocksFn,
}

impl fmt::Debug for HkdfSha384 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("HkdfSha384").finish_non_exhaustive()
  }
}

impl HkdfSha384 {
  /// HKDF-SHA384 pseudorandom key size in bytes.
  pub const OUTPUT_SIZE: usize = SHA384_OUTPUT_SIZE;

  /// Maximum RFC 5869 expand output size in bytes.
  pub const MAX_OUTPUT_SIZE: usize = SHA384_MAX_OUTPUT_SIZE;

  /// Perform HKDF-Extract with `salt` and `input_key_material`.
  #[inline]
  #[must_use]
  pub fn new(salt: &[u8], input_key_material: &[u8]) -> Self {
    Self::extract(salt, input_key_material)
  }

  /// Perform HKDF-Extract with `salt` and `input_key_material`.
  #[must_use]
  pub fn extract(salt: &[u8], input_key_material: &[u8]) -> Self {
    let zero_salt = [0u8; SHA384_OUTPUT_SIZE];
    let salt = if salt.is_empty() { &zero_salt[..] } else { salt };
    let prk = HmacSha384::mac(salt, input_key_material);

    let compress = sha384_dispatch::compress_dispatch().select(0);

    let mut key_block = [0u8; SHA384_BLOCK_SIZE];
    key_block[..SHA384_OUTPUT_SIZE].copy_from_slice(&prk);

    let (inner_init, outer_init) = hmac_prefix_state(&mut key_block, |ipad, opad| {
      let mut inner_init = SHA384_H0;
      compress(&mut inner_init, ipad);

      let mut outer_init = SHA384_H0;
      compress(&mut outer_init, opad);

      (inner_init, outer_init)
    });

    Self {
      prk,
      inner_init,
      outer_init,
      compress,
    }
  }

  /// Expand the stored pseudorandom key into `okm`.
  ///
  /// Uses raw SHA-384 state arrays and a single cached compress function,
  /// bypassing all `Sha384` struct creation, `Drop` zeroization, and dispatch
  /// overhead in the inner loop.
  #[allow(clippy::indexing_slicing)]
  pub fn expand(&self, info: &[u8], okm: &mut [u8]) -> Result<(), HkdfOutputLengthError> {
    if okm.len() > SHA384_MAX_OUTPUT_SIZE {
      return Err(HkdfOutputLengthError::new());
    }

    if okm.is_empty() {
      return Ok(());
    }

    let compress = self.compress;
    let inner_init = self.inner_init;
    let outer_init = self.outer_init;

    let mut outer_block = [0u8; SHA384_BLOCK_SIZE];
    outer_block[SHA384_OUTPUT_SIZE] = 0x80;
    outer_block[112..SHA384_BLOCK_SIZE].copy_from_slice(&1408u128.to_be_bytes());

    let mut prev_tag = [0u8; SHA384_OUTPUT_SIZE];
    let mut state = [0u64; 8];
    let mut counter: u8 = 1;

    let mut chunks = okm.chunks_mut(SHA384_OUTPUT_SIZE);
    let Some(first) = chunks.next() else {
      return Ok(());
    };

    expand_hmac_sha384_inner(compress, &inner_init, None, info, counter, &mut state, &mut outer_block);
    expand_hmac_sha384_outer(compress, &outer_init, &mut state, &mut outer_block, &mut prev_tag);
    first.copy_from_slice(&prev_tag[..first.len()]);
    counter = counter.wrapping_add(1);

    for chunk in chunks {
      expand_hmac_sha384_inner(
        compress,
        &inner_init,
        Some(&prev_tag),
        info,
        counter,
        &mut state,
        &mut outer_block,
      );
      expand_hmac_sha384_outer(compress, &outer_init, &mut state, &mut outer_block, &mut prev_tag);
      chunk.copy_from_slice(&prev_tag[..chunk.len()]);
      counter = counter.wrapping_add(1);
    }

    ct::zeroize(&mut prev_tag);
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
  pub fn prk(&self) -> &[u8; SHA384_OUTPUT_SIZE] {
    &self.prk
  }

  #[cfg(test)]
  pub(crate) fn extract_with_compress_for_test(
    salt: &[u8],
    input_key_material: &[u8],
    compress: Sha512FamilyCompressBlocksFn,
  ) -> Self {
    let zero_salt = [0u8; SHA384_OUTPUT_SIZE];
    let salt = if salt.is_empty() { &zero_salt[..] } else { salt };
    let prk = HmacSha384::mac_with_compress_for_test(salt, input_key_material, compress);

    let mut key_block = [0u8; SHA384_BLOCK_SIZE];
    key_block[..SHA384_OUTPUT_SIZE].copy_from_slice(&prk);

    let (inner_init, outer_init) = hmac_prefix_state(&mut key_block, |ipad, opad| {
      let mut inner_init = SHA384_H0;
      compress(&mut inner_init, ipad);

      let mut outer_init = SHA384_H0;
      compress(&mut outer_init, opad);

      (inner_init, outer_init)
    });

    Self {
      prk,
      inner_init,
      outer_init,
      compress,
    }
  }
}

impl Drop for HkdfSha384 {
  fn drop(&mut self) {
    ct::zeroize(&mut self.prk);
    for word in self.inner_init.iter_mut().chain(self.outer_init.iter_mut()) {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

#[inline(always)]
#[allow(clippy::indexing_slicing)]
fn expand_hmac_sha256_inner(
  compress: Sha256CompressBlocksFn,
  inner_init: &[u32; 8],
  prev: Option<&[u8; SHA256_OUTPUT_SIZE]>,
  info: &[u8],
  counter: u8,
  state: &mut [u32; 8],
  out: &mut [u8; SHA256_OUTPUT_SIZE],
) {
  *state = *inner_init;

  let prev_len = if prev.is_some() { SHA256_OUTPUT_SIZE } else { 0 };
  let msg_len = prev_len.strict_add(info.len()).strict_add(1);
  let total_bytes = (SHA256_BLOCK_SIZE as u64).strict_add(msg_len as u64);

  let mut block = [0u8; SHA256_BLOCK_SIZE];
  let mut pos = 0usize;

  if let Some(prev) = prev {
    block[..SHA256_OUTPUT_SIZE].copy_from_slice(prev);
    pos = SHA256_OUTPUT_SIZE;
  }

  let mut info_off = 0usize;
  while info_off < info.len() {
    let space = SHA256_BLOCK_SIZE.strict_sub(pos);
    let remaining = info.len().strict_sub(info_off);
    let take = if space < remaining { space } else { remaining };
    block[pos..pos.strict_add(take)].copy_from_slice(&info[info_off..info_off.strict_add(take)]);
    pos = pos.strict_add(take);
    info_off = info_off.strict_add(take);
    if pos == SHA256_BLOCK_SIZE {
      compress(state, &block);
      block = [0u8; SHA256_BLOCK_SIZE];
      pos = 0;
    }
  }

  block[pos] = counter;
  pos = pos.strict_add(1);
  if pos == SHA256_BLOCK_SIZE {
    compress(state, &block);
    block = [0u8; SHA256_BLOCK_SIZE];
    pos = 0;
  }

  block[pos] = 0x80;
  if pos.strict_add(1) > 56 {
    compress(state, &block);
    block = [0u8; SHA256_BLOCK_SIZE];
  }
  block[56..SHA256_BLOCK_SIZE].copy_from_slice(&total_bytes.strict_mul(8).to_be_bytes());
  compress(state, &block);

  for (dst, &word) in out.chunks_exact_mut(4).zip(state.iter()) {
    dst.copy_from_slice(&word.to_be_bytes());
  }
}

#[inline(always)]
#[allow(clippy::indexing_slicing)]
fn expand_hmac_sha256_outer(
  compress: Sha256CompressBlocksFn,
  outer_init: &[u32; 8],
  inner_hash: &[u8; SHA256_OUTPUT_SIZE],
  state: &mut [u32; 8],
  outer_block: &mut [u8; SHA256_BLOCK_SIZE],
  out: &mut [u8; SHA256_OUTPUT_SIZE],
) {
  *state = *outer_init;
  outer_block[..SHA256_OUTPUT_SIZE].copy_from_slice(inner_hash);
  compress(state, outer_block);

  for (dst, &word) in out.chunks_exact_mut(4).zip(state.iter()) {
    dst.copy_from_slice(&word.to_be_bytes());
  }
}

#[inline(always)]
#[allow(clippy::indexing_slicing)]
fn expand_hmac_sha384_inner(
  compress: Sha512FamilyCompressBlocksFn,
  inner_init: &[u64; 8],
  prev: Option<&[u8; SHA384_OUTPUT_SIZE]>,
  info: &[u8],
  counter: u8,
  state: &mut [u64; 8],
  outer_block: &mut [u8; SHA384_BLOCK_SIZE],
) {
  *state = *inner_init;

  let prev_len = if prev.is_some() { SHA384_OUTPUT_SIZE } else { 0 };
  let msg_len = prev_len.strict_add(info.len()).strict_add(1);
  let total_bytes = (SHA384_BLOCK_SIZE as u128).strict_add(msg_len as u128);

  let mut block = [0u8; SHA384_BLOCK_SIZE];
  let mut pos = 0usize;

  if let Some(prev) = prev {
    block[..SHA384_OUTPUT_SIZE].copy_from_slice(prev);
    pos = SHA384_OUTPUT_SIZE;
  }

  let mut info_off = 0usize;
  while info_off < info.len() {
    let space = SHA384_BLOCK_SIZE.strict_sub(pos);
    let remaining = info.len().strict_sub(info_off);
    let take = if space < remaining { space } else { remaining };
    block[pos..pos.strict_add(take)].copy_from_slice(&info[info_off..info_off.strict_add(take)]);
    pos = pos.strict_add(take);
    info_off = info_off.strict_add(take);
    if pos == SHA384_BLOCK_SIZE {
      compress(state, &block);
      block = [0u8; SHA384_BLOCK_SIZE];
      pos = 0;
    }
  }

  block[pos] = counter;
  pos = pos.strict_add(1);
  if pos == SHA384_BLOCK_SIZE {
    compress(state, &block);
    block = [0u8; SHA384_BLOCK_SIZE];
    pos = 0;
  }

  block[pos] = 0x80;
  if pos.strict_add(1) > 112 {
    compress(state, &block);
    block = [0u8; SHA384_BLOCK_SIZE];
  }
  block[112..SHA384_BLOCK_SIZE].copy_from_slice(&total_bytes.strict_mul(8).to_be_bytes());
  compress(state, &block);

  for (dst, &word) in outer_block[..SHA384_OUTPUT_SIZE].chunks_exact_mut(8).zip(state.iter()) {
    dst.copy_from_slice(&word.to_be_bytes());
  }
}

#[inline(always)]
#[allow(clippy::indexing_slicing)]
fn expand_hmac_sha384_outer(
  compress: Sha512FamilyCompressBlocksFn,
  outer_init: &[u64; 8],
  state: &mut [u64; 8],
  outer_block: &mut [u8; SHA384_BLOCK_SIZE],
  out: &mut [u8; SHA384_OUTPUT_SIZE],
) {
  *state = *outer_init;
  compress(state, outer_block);

  for (dst, &word) in out.chunks_exact_mut(8).zip(state.iter()) {
    dst.copy_from_slice(&word.to_be_bytes());
  }
}

#[cfg(test)]
mod tests {
  use alloc::{vec, vec::Vec};

  use hkdf::Hkdf as RustCryptoHkdf;

  use super::*;
  use crate::hashes::crypto::sha384::kernels::{
    ALL as SHA384_KERNELS, Sha384KernelId, compress_blocks_fn as sha384_compress_blocks_fn,
    required_caps as sha384_required_caps,
  };

  type RustCryptoHkdfSha384 = RustCryptoHkdf<sha2::Sha384>;

  fn pattern(len: usize, mul: u8, add: u8) -> Vec<u8> {
    (0..len)
      .map(|i| {
        (i as u8)
          .wrapping_mul(mul)
          .wrapping_add(((i >> 2) as u8).wrapping_add(add))
      })
      .collect()
  }

  fn assert_hkdf_sha384_kernel(id: Sha384KernelId) {
    let compress = sha384_compress_blocks_fn(id);
    let cases = [
      (0usize, 0usize, 0usize, 0usize),
      (16, 32, 8, 48),
      (48, 48, 48, 96),
      (96, 64, 80, 256),
      (160, 96, 128, 1024),
    ];

    for &(salt_len, ikm_len, info_len, out_len) in &cases {
      let salt = pattern(salt_len, 13, 5);
      let ikm = pattern(ikm_len, 19, 9);
      let info = pattern(info_len, 31, 17);
      let mut expected = vec![0u8; out_len];
      RustCryptoHkdfSha384::new(Some(&salt), &ikm)
        .expand(&info, &mut expected)
        .unwrap();

      let public = HkdfSha384::new(&salt, &ikm);
      let forced = HkdfSha384::extract_with_compress_for_test(&salt, &ikm, compress);

      let mut public_out = vec![0u8; out_len];
      public.expand(&info, &mut public_out).unwrap();
      assert_eq!(
        public_out,
        expected,
        "hkdf-sha384 public mismatch kernel={} salt_len={} ikm_len={} info_len={} out_len={}",
        id.as_str(),
        salt_len,
        ikm_len,
        info_len,
        out_len
      );

      let mut forced_out = vec![0u8; out_len];
      forced.expand(&info, &mut forced_out).unwrap();
      assert_eq!(
        forced_out,
        expected,
        "hkdf-sha384 forced mismatch kernel={} salt_len={} ikm_len={} info_len={} out_len={}",
        id.as_str(),
        salt_len,
        ikm_len,
        info_len,
        out_len
      );
      assert_eq!(
        forced.prk(),
        public.prk(),
        "hkdf-sha384 prk mismatch kernel={} salt_len={} ikm_len={}",
        id.as_str(),
        salt_len,
        ikm_len
      );
    }
  }

  #[test]
  fn hkdf_sha384_forced_kernels_match_oracle() {
    let caps = crate::platform::caps();
    for &id in SHA384_KERNELS {
      if caps.has(sha384_required_caps(id)) {
        assert_hkdf_sha384_kernel(id);
      }
    }
  }
}
