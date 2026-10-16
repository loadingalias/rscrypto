//! HKDF-SHA2 family (RFC 5869).

use core::fmt;

use super::hmac::{HmacSha256, HmacSha384, HmacSha512};
use crate::{
  hashes::crypto::{
    sha256::{H0 as SHA256_H0, dispatch as sha256_dispatch, kernels::CompressBlocksFn as Sha256CompressBlocksFn},
    sha384::{H0 as SHA384_H0, dispatch as sha384_dispatch, kernels::CompressBlocksFn as Sha384CompressBlocksFn},
    sha512::{H0 as SHA512_H0, dispatch as sha512_dispatch, kernels::CompressBlocksFn as Sha512CompressBlocksFn},
  },
  traits::ct,
};

const SHA256_OUTPUT_SIZE: usize = 32;
const SHA256_MAX_OUTPUT_SIZE: usize = 255 * SHA256_OUTPUT_SIZE;
const SHA256_BLOCK_SIZE: usize = 64;

const SHA384_OUTPUT_SIZE: usize = 48;
const SHA384_MAX_OUTPUT_SIZE: usize = 255 * SHA384_OUTPUT_SIZE;
const SHA384_BLOCK_SIZE: usize = 128;

const SHA512_OUTPUT_SIZE: usize = 64;
const SHA512_MAX_OUTPUT_SIZE: usize = 255 * SHA512_OUTPUT_SIZE;
const SHA512_BLOCK_SIZE: usize = 128;

#[inline(always)]
fn write_u32x8_be(dst: &mut [u8; SHA256_OUTPUT_SIZE], words: &[u32; 8]) {
  dst[0..4].copy_from_slice(&words[0].to_be_bytes());
  dst[4..8].copy_from_slice(&words[1].to_be_bytes());
  dst[8..12].copy_from_slice(&words[2].to_be_bytes());
  dst[12..16].copy_from_slice(&words[3].to_be_bytes());
  dst[16..20].copy_from_slice(&words[4].to_be_bytes());
  dst[20..24].copy_from_slice(&words[5].to_be_bytes());
  dst[24..28].copy_from_slice(&words[6].to_be_bytes());
  dst[28..32].copy_from_slice(&words[7].to_be_bytes());
}

#[inline(always)]
fn write_u64x6_be(dst: &mut [u8], words: &[u64; 8]) {
  debug_assert!(dst.len() >= SHA384_OUTPUT_SIZE);
  dst[0..8].copy_from_slice(&words[0].to_be_bytes());
  dst[8..16].copy_from_slice(&words[1].to_be_bytes());
  dst[16..24].copy_from_slice(&words[2].to_be_bytes());
  dst[24..32].copy_from_slice(&words[3].to_be_bytes());
  dst[32..40].copy_from_slice(&words[4].to_be_bytes());
  dst[40..48].copy_from_slice(&words[5].to_be_bytes());
}

#[inline(always)]
fn write_u64x8_be(dst: &mut [u8], words: &[u64; 8]) {
  debug_assert!(dst.len() >= SHA512_OUTPUT_SIZE);
  dst[0..8].copy_from_slice(&words[0].to_be_bytes());
  dst[8..16].copy_from_slice(&words[1].to_be_bytes());
  dst[16..24].copy_from_slice(&words[2].to_be_bytes());
  dst[24..32].copy_from_slice(&words[3].to_be_bytes());
  dst[32..40].copy_from_slice(&words[4].to_be_bytes());
  dst[40..48].copy_from_slice(&words[5].to_be_bytes());
  dst[48..56].copy_from_slice(&words[6].to_be_bytes());
  dst[56..64].copy_from_slice(&words[7].to_be_bytes());
}

#[cfg(all(
  target_arch = "aarch64",
  target_feature = "sha2",
  not(feature = "portable-only"),
  not(miri)
))]
#[inline(always)]
fn compress_hkdf_sha256_block(
  _compress: Sha256CompressBlocksFn,
  state: &mut [u32; 8],
  block: &[u8; SHA256_BLOCK_SIZE],
) {
  // SAFETY: Direct Apple SHA2 single-block compression because:
  // 1. This build enables `target_feature = "sha2"` at compile time.
  // 2. HKDF passes exactly one initialized 64-byte SHA-256 block at each call site.
  // 3. `state` is the initialized HKDF-owned SHA-256 chaining state.
  unsafe {
    crate::hashes::crypto::sha256::aarch64::compress_single_block_aarch64_sha2(state, block);
  }
}

#[cfg(not(all(
  target_arch = "aarch64",
  target_feature = "sha2",
  not(feature = "portable-only"),
  not(miri)
)))]
#[inline(always)]
fn compress_hkdf_sha256_block(compress: Sha256CompressBlocksFn, state: &mut [u32; 8], block: &[u8; SHA256_BLOCK_SIZE]) {
  compress(state, block);
}

const HMAC_IPAD: u8 = 0x36;
const HMAC_OPAD: u8 = 0x5C;

/// Build an owner from the HMAC inner and outer prefix states of a hash-length `key`.
///
/// One block holds the inner pad and then becomes the outer pad in place. Each
/// state is the only memory that `compress` writes, and `build` copies both into
/// the owner before the block and states are cleared. No prefix value has to
/// live anywhere else, so none is left in uncleared spill slots.
#[inline(always)]
fn with_hmac_prefix_states<W: ct::WordZero, const KEY_SIZE: usize, const BLOCK_SIZE: usize, T>(
  key: &[u8; KEY_SIZE],
  h0: [W; 8],
  compress: fn(&mut [W; 8], &[u8]),
  build: impl FnOnce(&[W; 8], &[W; 8]) -> T,
) -> T {
  const { assert!(KEY_SIZE <= BLOCK_SIZE, "an HKDF PRK fits in one HMAC block") };

  let mut block = [HMAC_IPAD; BLOCK_SIZE];
  for (pad, &byte) in block.iter_mut().zip(key) {
    *pad ^= byte;
  }
  let mut inner = h0;
  compress(&mut inner, &block);

  for pad in &mut block {
    *pad ^= HMAC_IPAD ^ HMAC_OPAD;
  }
  let mut outer = h0;
  compress(&mut outer, &block);

  let owner = build(&inner, &outer);
  ct::zeroize_no_fence(&mut block);
  ct::zeroize_words_no_fence(&mut inner);
  ct::zeroize_words_no_fence(&mut outer);
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  owner
}

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
    Self::extract_with(
      salt,
      input_key_material,
      sha256_dispatch::compress_dispatch(),
      HmacSha256::mac_into,
    )
  }

  /// Build the expansion state from an existing pseudorandom key.
  ///
  /// `prk` must already be pseudorandom: an HKDF-Extract output or an earlier
  /// HKDF-Expand block, such as a TLS 1.3 secret. Pass raw input key material
  /// to [`new`](Self::new) instead. The state copies `prk`; the caller keeps
  /// responsibility for clearing its own copy.
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::HkdfSha256;
  ///
  /// let extracted = HkdfSha256::new(b"salt", b"input key material");
  /// let imported = HkdfSha256::from_prk(extracted.prk());
  ///
  /// assert_eq!(
  ///   imported.expand_array::<32>(b"context")?,
  ///   extracted.expand_array::<32>(b"context")?
  /// );
  /// # Ok::<(), rscrypto::auth::HkdfOutputLengthError>(())
  /// ```
  #[must_use]
  pub fn from_prk(prk: &[u8; SHA256_OUTPUT_SIZE]) -> Self {
    Self::from_prk_with(prk, sha256_dispatch::compress_dispatch())
  }

  /// Run HKDF-Extract with `mac`, then precompute the expand prefixes with `compress`.
  #[inline(always)]
  fn extract_with(
    salt: &[u8],
    input_key_material: &[u8],
    compress: Sha256CompressBlocksFn,
    mac: impl FnOnce(&[u8], &[u8], &mut [u8; SHA256_OUTPUT_SIZE]),
  ) -> Self {
    let zero_salt = [0u8; SHA256_OUTPUT_SIZE];
    let salt = if salt.is_empty() { &zero_salt[..] } else { salt };
    let mut prk = [0u8; SHA256_OUTPUT_SIZE];
    mac(salt, input_key_material, &mut prk);
    let hkdf = Self::from_prk_with(&prk, compress);
    ct::zeroize(&mut prk);
    hkdf
  }

  /// Store `prk` and precompute its HMAC prefixes with `compress`.
  #[inline(always)]
  fn from_prk_with(prk: &[u8; SHA256_OUTPUT_SIZE], compress: Sha256CompressBlocksFn) -> Self {
    with_hmac_prefix_states::<_, _, SHA256_BLOCK_SIZE, _>(prk, SHA256_H0, compress, |inner_init, outer_init| Self {
      prk: *prk,
      inner_init: *inner_init,
      outer_init: *outer_init,
      compress,
    })
  }

  /// Expand the stored pseudorandom key into `okm`.
  ///
  /// Uses raw SHA-256 state arrays and a single cached compress function,
  /// bypassing all `Sha256` struct creation, `Drop` zeroization, and dispatch
  /// overhead in the inner loop.
  #[inline]
  pub fn expand(&self, info: &[u8], okm: &mut [u8]) -> Result<(), HkdfOutputLengthError> {
    self.expand_vectored(core::slice::from_ref(&info), okm)
  }

  /// Expand the stored pseudorandom key into `okm`, with `info` given in parts.
  ///
  /// The output equals [`expand`](Self::expand) over the concatenation of the
  /// parts, without building that concatenation. A TLS 1.3 `HkdfLabel` is one
  /// such `info`.
  ///
  /// # Errors
  ///
  /// Returns [`HkdfOutputLengthError`] without writing `okm` when `okm` is
  /// longer than [`MAX_OUTPUT_SIZE`](Self::MAX_OUTPUT_SIZE).
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::HkdfSha256;
  ///
  /// let hkdf = HkdfSha256::new(b"salt", b"input key material");
  ///
  /// let mut okm = [0u8; 32];
  /// hkdf.expand_vectored(&[b"con", b"", b"text"], &mut okm)?;
  /// assert_eq!(okm, hkdf.expand_array::<32>(b"context")?);
  /// # Ok::<(), rscrypto::auth::HkdfOutputLengthError>(())
  /// ```
  #[inline]
  pub fn expand_vectored(&self, info: &[&[u8]], okm: &mut [u8]) -> Result<(), HkdfOutputLengthError> {
    if okm.len() > SHA256_MAX_OUTPUT_SIZE {
      return Err(HkdfOutputLengthError::new());
    }

    if okm.is_empty() {
      return Ok(());
    }

    let compress = self.compress;

    let mut inner_block = [0u8; SHA256_BLOCK_SIZE];
    let mut outer_block = [0u8; SHA256_BLOCK_SIZE];
    outer_block[SHA256_OUTPUT_SIZE] = 0x80;
    outer_block[56..SHA256_BLOCK_SIZE].copy_from_slice(&768u64.to_be_bytes());

    let mut inner_hash = [0u8; SHA256_OUTPUT_SIZE];
    let mut state = self.inner_init;
    let mut counter: u8 = 1;

    let mut prev_tag = [0u8; SHA256_OUTPUT_SIZE];
    let mut chunks = okm.chunks_mut(SHA256_OUTPUT_SIZE);
    let Some(first) = chunks.next() else {
      return Ok(());
    };

    expand_hmac_sha256_inner(
      compress,
      None,
      info,
      counter,
      &mut state,
      &mut inner_hash,
      &mut inner_block,
    );
    expand_hmac_sha256_outer(
      compress,
      &self.outer_init,
      &inner_hash,
      &mut state,
      &mut outer_block,
      &mut prev_tag,
    );
    first.copy_from_slice(&prev_tag[..first.len()]);
    counter = counter.wrapping_add(1);

    for chunk in chunks {
      state = self.inner_init;
      expand_hmac_sha256_inner(
        compress,
        Some(&prev_tag),
        info,
        counter,
        &mut state,
        &mut inner_hash,
        &mut inner_block,
      );
      expand_hmac_sha256_outer(
        compress,
        &self.outer_init,
        &inner_hash,
        &mut state,
        &mut outer_block,
        &mut prev_tag,
      );
      chunk.copy_from_slice(&prev_tag[..chunk.len()]);
      counter = counter.wrapping_add(1);
    }

    ct::zeroize_no_fence(&mut prev_tag);
    ct::zeroize_no_fence(&mut inner_hash);
    ct::zeroize_no_fence(&mut inner_block);
    ct::zeroize_no_fence(&mut outer_block);
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

  #[cfg(any(test, all(rscrypto_internal, feature = "diag")))]
  pub(crate) fn extract_with_compress_for_test(
    salt: &[u8],
    input_key_material: &[u8],
    compress: Sha256CompressBlocksFn,
  ) -> Self {
    Self::extract_with(salt, input_key_material, compress, |salt, input_key_material, prk| {
      *prk = HmacSha256::mac_with_compress_for_test(salt, input_key_material, compress);
    })
  }
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
#[unsafe(no_mangle)]
#[inline(never)]
/// Derive a portable HKDF-SHA256 diagnostic output using fixed `b"salt"` and `b"info"` inputs.
pub fn diag_hkdf_sha256_derive_portable(input_key_material: &[u8; SHA256_OUTPUT_SIZE]) -> [u8; SHA256_OUTPUT_SIZE] {
  let compress = crate::hashes::crypto::sha256::kernels::compress_blocks_fn(
    crate::hashes::crypto::sha256::kernels::Sha256KernelId::Portable,
  );
  HkdfSha256::extract_with_compress_for_test(b"salt", input_key_material, compress)
    .expand_array::<SHA256_OUTPUT_SIZE>(b"info")
    .unwrap_or([0u8; SHA256_OUTPUT_SIZE])
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
pub struct HkdfSha384 {
  prk: [u8; SHA384_OUTPUT_SIZE],
  inner_init: [u64; 8],
  outer_init: [u64; 8],
  compress: Sha384CompressBlocksFn,
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
    Self::extract_with(
      salt,
      input_key_material,
      sha384_dispatch::compress_dispatch(),
      HmacSha384::mac_into,
    )
  }

  /// Build the expansion state from an existing pseudorandom key.
  ///
  /// `prk` must already be pseudorandom: an HKDF-Extract output or an earlier
  /// HKDF-Expand block, such as a TLS 1.3 secret. Pass raw input key material
  /// to [`new`](Self::new) instead. The state copies `prk`; the caller keeps
  /// responsibility for clearing its own copy.
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::HkdfSha384;
  ///
  /// let extracted = HkdfSha384::new(b"salt", b"input key material");
  /// let imported = HkdfSha384::from_prk(extracted.prk());
  ///
  /// assert_eq!(
  ///   imported.expand_array::<48>(b"context")?,
  ///   extracted.expand_array::<48>(b"context")?
  /// );
  /// # Ok::<(), rscrypto::auth::HkdfOutputLengthError>(())
  /// ```
  #[must_use]
  pub fn from_prk(prk: &[u8; SHA384_OUTPUT_SIZE]) -> Self {
    Self::from_prk_with(prk, sha384_dispatch::compress_dispatch())
  }

  /// Run HKDF-Extract with `mac`, then precompute the expand prefixes with `compress`.
  #[inline(always)]
  fn extract_with(
    salt: &[u8],
    input_key_material: &[u8],
    compress: Sha384CompressBlocksFn,
    mac: impl FnOnce(&[u8], &[u8], &mut [u8; SHA384_OUTPUT_SIZE]),
  ) -> Self {
    let zero_salt = [0u8; SHA384_OUTPUT_SIZE];
    let salt = if salt.is_empty() { &zero_salt[..] } else { salt };
    let mut prk = [0u8; SHA384_OUTPUT_SIZE];
    mac(salt, input_key_material, &mut prk);
    let hkdf = Self::from_prk_with(&prk, compress);
    ct::zeroize(&mut prk);
    hkdf
  }

  /// Store `prk` and precompute its HMAC prefixes with `compress`.
  #[inline(always)]
  fn from_prk_with(prk: &[u8; SHA384_OUTPUT_SIZE], compress: Sha384CompressBlocksFn) -> Self {
    with_hmac_prefix_states::<_, _, SHA384_BLOCK_SIZE, _>(prk, SHA384_H0, compress, |inner_init, outer_init| Self {
      prk: *prk,
      inner_init: *inner_init,
      outer_init: *outer_init,
      compress,
    })
  }

  /// Expand the stored pseudorandom key into `okm`.
  ///
  /// Uses raw SHA-384 state arrays and a single cached compress function,
  /// bypassing all `Sha384` struct creation, `Drop` zeroization, and dispatch
  /// overhead in the inner loop.
  #[inline]
  pub fn expand(&self, info: &[u8], okm: &mut [u8]) -> Result<(), HkdfOutputLengthError> {
    self.expand_vectored(core::slice::from_ref(&info), okm)
  }

  /// Expand the stored pseudorandom key into `okm`, with `info` given in parts.
  ///
  /// The output equals [`expand`](Self::expand) over the concatenation of the
  /// parts, without building that concatenation. A TLS 1.3 `HkdfLabel` is one
  /// such `info`.
  ///
  /// # Errors
  ///
  /// Returns [`HkdfOutputLengthError`] without writing `okm` when `okm` is
  /// longer than [`MAX_OUTPUT_SIZE`](Self::MAX_OUTPUT_SIZE).
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::HkdfSha384;
  ///
  /// let hkdf = HkdfSha384::new(b"salt", b"input key material");
  ///
  /// let mut okm = [0u8; 48];
  /// hkdf.expand_vectored(&[b"con", b"", b"text"], &mut okm)?;
  /// assert_eq!(okm, hkdf.expand_array::<48>(b"context")?);
  /// # Ok::<(), rscrypto::auth::HkdfOutputLengthError>(())
  /// ```
  #[inline]
  pub fn expand_vectored(&self, info: &[&[u8]], okm: &mut [u8]) -> Result<(), HkdfOutputLengthError> {
    if okm.len() > SHA384_MAX_OUTPUT_SIZE {
      return Err(HkdfOutputLengthError::new());
    }

    if okm.is_empty() {
      return Ok(());
    }

    let compress = self.compress;

    let mut inner_block = [0u8; SHA384_BLOCK_SIZE];
    let mut outer_block = [0u8; SHA384_BLOCK_SIZE];
    outer_block[SHA384_OUTPUT_SIZE] = 0x80;
    outer_block[112..SHA384_BLOCK_SIZE].copy_from_slice(&1408u128.to_be_bytes());

    let mut state = self.inner_init;
    let mut counter: u8 = 1;

    let mut prev_tag = [0u8; SHA384_OUTPUT_SIZE];
    let mut chunks = okm.chunks_mut(SHA384_OUTPUT_SIZE);
    let Some(first) = chunks.next() else {
      return Ok(());
    };

    expand_hmac_sha384_inner(
      compress,
      None,
      info,
      counter,
      &mut state,
      &mut outer_block,
      &mut inner_block,
    );
    expand_hmac_sha384_outer(compress, &self.outer_init, &mut state, &mut outer_block, &mut prev_tag);
    first.copy_from_slice(&prev_tag[..first.len()]);
    counter = counter.wrapping_add(1);

    for chunk in chunks {
      state = self.inner_init;
      expand_hmac_sha384_inner(
        compress,
        Some(&prev_tag),
        info,
        counter,
        &mut state,
        &mut outer_block,
        &mut inner_block,
      );
      expand_hmac_sha384_outer(compress, &self.outer_init, &mut state, &mut outer_block, &mut prev_tag);
      chunk.copy_from_slice(&prev_tag[..chunk.len()]);
      counter = counter.wrapping_add(1);
    }

    ct::zeroize_no_fence(&mut prev_tag);
    ct::zeroize_no_fence(&mut inner_block);
    ct::zeroize_no_fence(&mut outer_block);
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

  #[cfg(any(test, all(rscrypto_internal, feature = "diag")))]
  pub(crate) fn extract_with_compress_for_test(
    salt: &[u8],
    input_key_material: &[u8],
    compress: Sha384CompressBlocksFn,
  ) -> Self {
    Self::extract_with(salt, input_key_material, compress, |salt, input_key_material, prk| {
      *prk = HmacSha384::mac_with_compress_for_test(salt, input_key_material, compress);
    })
  }
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
#[unsafe(no_mangle)]
#[inline(never)]
/// Derive a portable HKDF-SHA384 diagnostic output using fixed `b"salt"` and `b"info"` inputs.
pub fn diag_hkdf_sha384_derive_portable(input_key_material: &[u8; SHA384_OUTPUT_SIZE]) -> [u8; SHA384_OUTPUT_SIZE] {
  let compress = crate::hashes::crypto::sha384::kernels::compress_blocks_fn(
    crate::hashes::crypto::sha384::kernels::Sha384KernelId::Portable,
  );
  HkdfSha384::extract_with_compress_for_test(b"salt", input_key_material, compress)
    .expand_array::<SHA384_OUTPUT_SIZE>(b"info")
    .unwrap_or([0u8; SHA384_OUTPUT_SIZE])
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

/// HKDF-SHA512 pseudorandom key state.
///
/// `new()` and `extract()` perform HKDF-Extract and store the pseudorandom key
/// for later `expand()` calls.
pub struct HkdfSha512 {
  prk: [u8; SHA512_OUTPUT_SIZE],
  inner_init: [u64; 8],
  outer_init: [u64; 8],
  compress: Sha512CompressBlocksFn,
}

impl fmt::Debug for HkdfSha512 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("HkdfSha512").finish_non_exhaustive()
  }
}

impl HkdfSha512 {
  /// HKDF-SHA512 pseudorandom key size in bytes.
  pub const OUTPUT_SIZE: usize = SHA512_OUTPUT_SIZE;

  /// Maximum RFC 5869 expand output size in bytes.
  pub const MAX_OUTPUT_SIZE: usize = SHA512_MAX_OUTPUT_SIZE;

  /// Perform HKDF-Extract with `salt` and `input_key_material`.
  #[inline]
  #[must_use]
  pub fn new(salt: &[u8], input_key_material: &[u8]) -> Self {
    Self::extract(salt, input_key_material)
  }

  /// Perform HKDF-Extract with `salt` and `input_key_material`.
  #[must_use]
  pub fn extract(salt: &[u8], input_key_material: &[u8]) -> Self {
    Self::extract_with(
      salt,
      input_key_material,
      sha512_dispatch::compress_dispatch(),
      HmacSha512::mac_into,
    )
  }

  /// Run HKDF-Extract with `mac`, then precompute the expand prefixes with `compress`.
  #[inline(always)]
  fn extract_with(
    salt: &[u8],
    input_key_material: &[u8],
    compress: Sha512CompressBlocksFn,
    mac: impl FnOnce(&[u8], &[u8], &mut [u8; SHA512_OUTPUT_SIZE]),
  ) -> Self {
    let zero_salt = [0u8; SHA512_OUTPUT_SIZE];
    let salt = if salt.is_empty() { &zero_salt[..] } else { salt };
    let mut prk = [0u8; SHA512_OUTPUT_SIZE];
    mac(salt, input_key_material, &mut prk);
    let hkdf = Self::from_prk_with(&prk, compress);
    ct::zeroize(&mut prk);
    hkdf
  }

  /// Store `prk` and precompute its HMAC prefixes with `compress`.
  #[inline(always)]
  fn from_prk_with(prk: &[u8; SHA512_OUTPUT_SIZE], compress: Sha512CompressBlocksFn) -> Self {
    with_hmac_prefix_states::<_, _, SHA512_BLOCK_SIZE, _>(prk, SHA512_H0, compress, |inner_init, outer_init| Self {
      prk: *prk,
      inner_init: *inner_init,
      outer_init: *outer_init,
      compress,
    })
  }

  /// Expand the stored pseudorandom key into `okm`.
  #[inline]
  pub fn expand(&self, info: &[u8], okm: &mut [u8]) -> Result<(), HkdfOutputLengthError> {
    if okm.len() > SHA512_MAX_OUTPUT_SIZE {
      return Err(HkdfOutputLengthError::new());
    }

    if okm.is_empty() {
      return Ok(());
    }

    let compress = self.compress;

    let mut inner_block = [0u8; SHA512_BLOCK_SIZE];
    let mut outer_block = [0u8; SHA512_BLOCK_SIZE];
    outer_block[SHA512_OUTPUT_SIZE] = 0x80;
    outer_block[112..SHA512_BLOCK_SIZE].copy_from_slice(&1536u128.to_be_bytes());

    let mut state = self.inner_init;
    let mut counter: u8 = 1;

    let mut prev_tag = [0u8; SHA512_OUTPUT_SIZE];
    let mut chunks = okm.chunks_mut(SHA512_OUTPUT_SIZE);
    let Some(first) = chunks.next() else {
      return Ok(());
    };

    expand_hmac_sha512_inner(
      compress,
      None,
      info,
      counter,
      &mut state,
      &mut outer_block,
      &mut inner_block,
    );
    expand_hmac_sha512_outer(compress, &self.outer_init, &mut state, &mut outer_block, &mut prev_tag);
    first.copy_from_slice(&prev_tag[..first.len()]);
    counter = counter.wrapping_add(1);

    for chunk in chunks {
      state = self.inner_init;
      expand_hmac_sha512_inner(
        compress,
        Some(&prev_tag),
        info,
        counter,
        &mut state,
        &mut outer_block,
        &mut inner_block,
      );
      expand_hmac_sha512_outer(compress, &self.outer_init, &mut state, &mut outer_block, &mut prev_tag);
      chunk.copy_from_slice(&prev_tag[..chunk.len()]);
      counter = counter.wrapping_add(1);
    }

    ct::zeroize_no_fence(&mut prev_tag);
    ct::zeroize_no_fence(&mut inner_block);
    ct::zeroize_no_fence(&mut outer_block);
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
  pub fn prk(&self) -> &[u8; SHA512_OUTPUT_SIZE] {
    &self.prk
  }

  #[cfg(any(test, all(rscrypto_internal, feature = "diag")))]
  pub(crate) fn extract_with_compress_for_test(
    salt: &[u8],
    input_key_material: &[u8],
    compress: Sha512CompressBlocksFn,
  ) -> Self {
    Self::extract_with(salt, input_key_material, compress, |salt, input_key_material, prk| {
      *prk = HmacSha512::mac_with_compress_for_test(salt, input_key_material, compress);
    })
  }
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
#[unsafe(no_mangle)]
#[inline(never)]
/// Derive a portable HKDF-SHA512 diagnostic output using fixed `b"salt"` and `b"info"` inputs.
pub fn diag_hkdf_sha512_derive_portable(input_key_material: &[u8; SHA512_OUTPUT_SIZE]) -> [u8; SHA512_OUTPUT_SIZE] {
  let compress = crate::hashes::crypto::sha512::kernels::compress_blocks_fn(
    crate::hashes::crypto::sha512::kernels::Sha512KernelId::Portable,
  );
  HkdfSha512::extract_with_compress_for_test(b"salt", input_key_material, compress)
    .expand_array::<SHA512_OUTPUT_SIZE>(b"info")
    .unwrap_or([0u8; SHA512_OUTPUT_SIZE])
}

impl Drop for HkdfSha512 {
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
fn expand_hmac_sha256_inner(
  compress: Sha256CompressBlocksFn,
  prev: Option<&[u8; SHA256_OUTPUT_SIZE]>,
  info: &[&[u8]],
  counter: u8,
  state: &mut [u32; 8],
  out: &mut [u8; SHA256_OUTPUT_SIZE],
  block: &mut [u8; SHA256_BLOCK_SIZE],
) {
  let prev_len = if prev.is_some() { SHA256_OUTPUT_SIZE } else { 0 };
  // Sum in u64, not usize: repeated parts can exceed usize on 32-bit targets. A total
  // beyond SHA-256's message limit fails strict arithmetic, as in `Sha256::update`.
  let info_len = info.iter().fold(0u64, |len, part| len.strict_add(part.len() as u64));
  let total_bytes = (SHA256_BLOCK_SIZE as u64)
    .strict_add(prev_len as u64)
    .strict_add(info_len)
    .strict_add(1);

  block.fill(0);
  let mut pos = 0usize;

  if let Some(prev) = prev {
    block[..SHA256_OUTPUT_SIZE].copy_from_slice(prev);
    pos = SHA256_OUTPUT_SIZE;
  }

  for part in info {
    let mut part_off = 0usize;
    while part_off < part.len() {
      let space = SHA256_BLOCK_SIZE.strict_sub(pos);
      let remaining = part.len().strict_sub(part_off);
      let take = if space < remaining { space } else { remaining };
      block[pos..pos.strict_add(take)].copy_from_slice(&part[part_off..part_off.strict_add(take)]);
      pos = pos.strict_add(take);
      part_off = part_off.strict_add(take);
      if pos == SHA256_BLOCK_SIZE {
        compress_hkdf_sha256_block(compress, state, block);
        block.fill(0);
        pos = 0;
      }
    }
  }

  block[pos] = counter;
  pos = pos.strict_add(1);
  if pos == SHA256_BLOCK_SIZE {
    compress_hkdf_sha256_block(compress, state, block);
    block.fill(0);
    pos = 0;
  }

  block[pos] = 0x80;
  if pos.strict_add(1) > 56 {
    compress_hkdf_sha256_block(compress, state, block);
    block.fill(0);
  }
  block[56..SHA256_BLOCK_SIZE].copy_from_slice(&total_bytes.strict_mul(8).to_be_bytes());
  compress_hkdf_sha256_block(compress, state, block);

  write_u32x8_be(out, state);
}

#[inline(always)]
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
  compress_hkdf_sha256_block(compress, state, outer_block);

  write_u32x8_be(out, state);
}

#[inline(always)]
fn expand_hmac_sha384_inner(
  compress: Sha384CompressBlocksFn,
  prev: Option<&[u8; SHA384_OUTPUT_SIZE]>,
  info: &[&[u8]],
  counter: u8,
  state: &mut [u64; 8],
  outer_block: &mut [u8; SHA384_BLOCK_SIZE],
  block: &mut [u8; SHA384_BLOCK_SIZE],
) {
  let prev_len = if prev.is_some() { SHA384_OUTPUT_SIZE } else { 0 };
  let info_len = info.iter().fold(0u128, |len, part| len.strict_add(part.len() as u128));
  let total_bytes = (SHA384_BLOCK_SIZE as u128)
    .strict_add(prev_len as u128)
    .strict_add(info_len)
    .strict_add(1);

  block.fill(0);
  let mut pos = 0usize;

  if let Some(prev) = prev {
    block[..SHA384_OUTPUT_SIZE].copy_from_slice(prev);
    pos = SHA384_OUTPUT_SIZE;
  }

  for part in info {
    let mut part_off = 0usize;
    while part_off < part.len() {
      let space = SHA384_BLOCK_SIZE.strict_sub(pos);
      let remaining = part.len().strict_sub(part_off);
      let take = if space < remaining { space } else { remaining };
      block[pos..pos.strict_add(take)].copy_from_slice(&part[part_off..part_off.strict_add(take)]);
      pos = pos.strict_add(take);
      part_off = part_off.strict_add(take);
      if pos == SHA384_BLOCK_SIZE {
        compress(state, block);
        block.fill(0);
        pos = 0;
      }
    }
  }

  block[pos] = counter;
  pos = pos.strict_add(1);
  if pos == SHA384_BLOCK_SIZE {
    compress(state, block);
    block.fill(0);
    pos = 0;
  }

  block[pos] = 0x80;
  if pos.strict_add(1) > 112 {
    compress(state, block);
    block.fill(0);
  }
  block[112..SHA384_BLOCK_SIZE].copy_from_slice(&total_bytes.strict_mul(8).to_be_bytes());
  compress(state, block);

  write_u64x6_be(&mut outer_block[..SHA384_OUTPUT_SIZE], state);
}

#[inline(always)]
fn expand_hmac_sha384_outer(
  compress: Sha384CompressBlocksFn,
  outer_init: &[u64; 8],
  state: &mut [u64; 8],
  outer_block: &mut [u8; SHA384_BLOCK_SIZE],
  out: &mut [u8; SHA384_OUTPUT_SIZE],
) {
  *state = *outer_init;
  compress(state, outer_block);

  write_u64x6_be(out, state);
}

#[inline(always)]
fn expand_hmac_sha512_inner(
  compress: Sha512CompressBlocksFn,
  prev: Option<&[u8; SHA512_OUTPUT_SIZE]>,
  info: &[u8],
  counter: u8,
  state: &mut [u64; 8],
  outer_block: &mut [u8; SHA512_BLOCK_SIZE],
  block: &mut [u8; SHA512_BLOCK_SIZE],
) {
  let prev_len = if prev.is_some() { SHA512_OUTPUT_SIZE } else { 0 };
  let msg_len = prev_len.strict_add(info.len()).strict_add(1);
  let total_bytes = (SHA512_BLOCK_SIZE as u128).strict_add(msg_len as u128);

  block.fill(0);
  let mut pos = 0usize;

  if let Some(prev) = prev {
    block[..SHA512_OUTPUT_SIZE].copy_from_slice(prev);
    pos = SHA512_OUTPUT_SIZE;
  }

  let mut info_off = 0usize;
  while info_off < info.len() {
    let space = SHA512_BLOCK_SIZE.strict_sub(pos);
    let remaining = info.len().strict_sub(info_off);
    let take = if space < remaining { space } else { remaining };
    block[pos..pos.strict_add(take)].copy_from_slice(&info[info_off..info_off.strict_add(take)]);
    pos = pos.strict_add(take);
    info_off = info_off.strict_add(take);
    if pos == SHA512_BLOCK_SIZE {
      compress(state, block);
      block.fill(0);
      pos = 0;
    }
  }

  block[pos] = counter;
  pos = pos.strict_add(1);
  if pos == SHA512_BLOCK_SIZE {
    compress(state, block);
    block.fill(0);
    pos = 0;
  }

  block[pos] = 0x80;
  if pos.strict_add(1) > 112 {
    compress(state, block);
    block.fill(0);
  }
  block[112..SHA512_BLOCK_SIZE].copy_from_slice(&total_bytes.strict_mul(8).to_be_bytes());
  compress(state, block);

  write_u64x8_be(&mut outer_block[..SHA512_OUTPUT_SIZE], state);
}

#[inline(always)]
fn expand_hmac_sha512_outer(
  compress: Sha512CompressBlocksFn,
  outer_init: &[u64; 8],
  state: &mut [u64; 8],
  outer_block: &mut [u8; SHA512_BLOCK_SIZE],
  out: &mut [u8; SHA512_OUTPUT_SIZE],
) {
  *state = *outer_init;
  compress(state, outer_block);

  write_u64x8_be(out, state);
}

#[cfg(test)]
mod tests {
  use alloc::{vec, vec::Vec};

  use hkdf::Hkdf as RustCryptoHkdf;

  use super::*;
  use crate::hashes::crypto::{
    Sha256,
    sha256::kernels::{
      ALL as SHA256_KERNELS, Sha256KernelId, compress_blocks_fn as sha256_compress_blocks_fn,
      required_caps as sha256_required_caps,
    },
    sha384::kernels::{
      ALL as SHA384_KERNELS, Sha384KernelId, compress_blocks_fn as sha384_compress_blocks_fn,
      required_caps as sha384_required_caps,
    },
    sha512::kernels::{
      ALL as SHA512_KERNELS, Sha512KernelId, compress_blocks_fn as sha512_compress_blocks_fn,
      required_caps as sha512_required_caps,
    },
  };

  type RustCryptoHkdfSha256 = RustCryptoHkdf<sha2::Sha256>;
  type RustCryptoHkdfSha384 = RustCryptoHkdf<sha2::Sha384>;
  type RustCryptoHkdfSha512 = RustCryptoHkdf<sha2::Sha512>;

  fn pattern(len: usize, mul: u8, add: u8) -> Vec<u8> {
    (0..len)
      .map(|i| {
        i.to_le_bytes()[0]
          .wrapping_mul(mul)
          .wrapping_add((i >> 2).to_le_bytes()[0].wrapping_add(add))
      })
      .collect()
  }

  fn assert_hkdf_sha256_kernel(id: Sha256KernelId) {
    let compress = sha256_compress_blocks_fn(id);
    let cases = [
      (0usize, 0usize, 0usize, 0usize),
      (13, 22, 10, 1),
      (16, 32, 31, 32),
      (32, 48, 32, 33),
      (64, 56, 55, 64),
      (96, 80, 56, 65),
      (128, 96, 57, 256),
      (160, 128, 128, 1024),
    ];

    for &(salt_len, ikm_len, info_len, out_len) in &cases {
      let salt = pattern(salt_len, 11, 3);
      let ikm = pattern(ikm_len, 17, 7);
      let info = pattern(info_len, 29, 13);
      let mut expected = vec![0u8; out_len];
      RustCryptoHkdfSha256::new(Some(&salt), &ikm)
        .expand(&info, &mut expected)
        .expect("test output length should be within the HKDF-SHA256 limit");

      let public = HkdfSha256::new(&salt, &ikm);
      let forced = HkdfSha256::extract_with_compress_for_test(&salt, &ikm, compress);

      let mut public_out = vec![0u8; out_len];
      public
        .expand(&info, &mut public_out)
        .expect("test output length should be within the HKDF-SHA256 limit");
      assert_eq!(
        public_out,
        expected,
        "hkdf-sha256 public mismatch kernel={} salt_len={} ikm_len={} info_len={} out_len={}",
        id.as_str(),
        salt_len,
        ikm_len,
        info_len,
        out_len
      );

      let mut forced_out = vec![0u8; out_len];
      forced
        .expand(&info, &mut forced_out)
        .expect("test output length should be within the HKDF-SHA256 limit");
      assert_eq!(
        forced_out,
        expected,
        "hkdf-sha256 forced mismatch kernel={} salt_len={} ikm_len={} info_len={} out_len={}",
        id.as_str(),
        salt_len,
        ikm_len,
        info_len,
        out_len
      );
      assert_eq!(
        forced.prk(),
        public.prk(),
        "hkdf-sha256 prk mismatch kernel={} salt_len={} ikm_len={}",
        id.as_str(),
        salt_len,
        ikm_len
      );
    }
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
        .expect("test output length should be within the HKDF-SHA384 limit");

      let public = HkdfSha384::new(&salt, &ikm);
      let forced = HkdfSha384::extract_with_compress_for_test(&salt, &ikm, compress);

      let mut public_out = vec![0u8; out_len];
      public
        .expand(&info, &mut public_out)
        .expect("test output length should be within the HKDF-SHA384 limit");
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
      forced
        .expand(&info, &mut forced_out)
        .expect("test output length should be within the HKDF-SHA384 limit");
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

  fn assert_hkdf_sha512_kernel(id: Sha512KernelId) {
    let compress = sha512_compress_blocks_fn(id);
    let cases = [
      (0usize, 0usize, 0usize, 0usize),
      (16, 32, 8, 64),
      (64, 64, 64, 128),
      (128, 96, 96, 256),
      (192, 128, 160, 1024),
    ];

    for &(salt_len, ikm_len, info_len, out_len) in &cases {
      let salt = pattern(salt_len, 17, 7);
      let ikm = pattern(ikm_len, 23, 11);
      let info = pattern(info_len, 37, 19);
      let mut expected = vec![0u8; out_len];
      RustCryptoHkdfSha512::new(Some(&salt), &ikm)
        .expand(&info, &mut expected)
        .expect("test output length should be within the HKDF-SHA512 limit");

      let public = HkdfSha512::new(&salt, &ikm);
      let forced = HkdfSha512::extract_with_compress_for_test(&salt, &ikm, compress);

      let mut public_out = vec![0u8; out_len];
      public
        .expand(&info, &mut public_out)
        .expect("test output length should be within the HKDF-SHA512 limit");
      assert_eq!(
        public_out,
        expected,
        "hkdf-sha512 public mismatch kernel={} salt_len={} ikm_len={} info_len={} out_len={}",
        id.as_str(),
        salt_len,
        ikm_len,
        info_len,
        out_len
      );

      let mut forced_out = vec![0u8; out_len];
      forced
        .expand(&info, &mut forced_out)
        .expect("test output length should be within the HKDF-SHA512 limit");
      assert_eq!(
        forced_out,
        expected,
        "hkdf-sha512 forced mismatch kernel={} salt_len={} ikm_len={} info_len={} out_len={}",
        id.as_str(),
        salt_len,
        ikm_len,
        info_len,
        out_len
      );
      assert_eq!(
        forced.prk(),
        public.prk(),
        "hkdf-sha512 prk mismatch kernel={} salt_len={} ikm_len={}",
        id.as_str(),
        salt_len,
        ikm_len
      );
    }
  }

  #[test]
  fn hkdf_sha256_block_compression_matches_portable_compression() {
    let blocks = [[0u8; SHA256_BLOCK_SIZE], [0xffu8; SHA256_BLOCK_SIZE], {
      let mut block = [0u8; SHA256_BLOCK_SIZE];
      block.copy_from_slice(&pattern(SHA256_BLOCK_SIZE, 37, 5));
      block
    }];
    let states = [SHA256_H0, [0u32; 8], [u32::MAX; 8]];

    for block in &blocks {
      for initial in states {
        let mut expected = initial;
        Sha256::compress_blocks_portable(&mut expected, block);
        let mut actual = initial;
        compress_hkdf_sha256_block(sha256_dispatch::compress_dispatch(), &mut actual, block);
        assert_eq!(actual, expected);
      }
    }
  }

  #[test]
  fn hkdf_sha256_forced_kernels_match_oracle() {
    let caps = crate::platform::caps();
    for &id in SHA256_KERNELS {
      if caps.has(sha256_required_caps(id)) {
        assert_hkdf_sha256_kernel(id);
      }
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

  /// Splitting `info` into a head, an empty part, and a tail at every offset
  /// must expand like the concatenation, on every available kernel. The info
  /// spans two blocks and the output three HMAC iterations.
  #[test]
  fn hkdf_sha256_vectored_info_matches_oracle_at_every_split() {
    const INFO_LEN: usize = 2 * SHA256_BLOCK_SIZE + 3;
    const OKM_LEN: usize = 2 * SHA256_OUTPUT_SIZE + 1;
    let salt = pattern(13, 11, 3);
    let ikm = pattern(22, 17, 7);
    let info = pattern(INFO_LEN, 29, 13);
    let mut expected = [0u8; OKM_LEN];
    RustCryptoHkdfSha256::new(Some(&salt), &ikm)
      .expand(&info, &mut expected)
      .expect("test output length should be within the HKDF-SHA256 limit");

    let caps = crate::platform::caps();
    for &id in SHA256_KERNELS {
      if !caps.has(sha256_required_caps(id)) {
        continue;
      }
      let hkdf = HkdfSha256::extract_with_compress_for_test(&salt, &ikm, sha256_compress_blocks_fn(id));
      for split in 0..=INFO_LEN {
        let (head, tail) = info.split_at(split);
        let mut okm = [0u8; OKM_LEN];
        hkdf
          .expand_vectored(&[head, &[], tail], &mut okm)
          .expect("test output length should be within the HKDF-SHA256 limit");
        assert_eq!(
          okm,
          expected,
          "hkdf-sha256 vectored mismatch kernel={} split={split}",
          id.as_str()
        );
      }
    }
  }

  /// The SHA-384 counterpart of the every-split test above.
  #[test]
  fn hkdf_sha384_vectored_info_matches_oracle_at_every_split() {
    const INFO_LEN: usize = 2 * SHA384_BLOCK_SIZE + 3;
    const OKM_LEN: usize = 2 * SHA384_OUTPUT_SIZE + 1;
    let salt = pattern(13, 13, 5);
    let ikm = pattern(22, 19, 9);
    let info = pattern(INFO_LEN, 31, 17);
    let mut expected = [0u8; OKM_LEN];
    RustCryptoHkdfSha384::new(Some(&salt), &ikm)
      .expand(&info, &mut expected)
      .expect("test output length should be within the HKDF-SHA384 limit");

    let caps = crate::platform::caps();
    for &id in SHA384_KERNELS {
      if !caps.has(sha384_required_caps(id)) {
        continue;
      }
      let hkdf = HkdfSha384::extract_with_compress_for_test(&salt, &ikm, sha384_compress_blocks_fn(id));
      for split in 0..=INFO_LEN {
        let (head, tail) = info.split_at(split);
        let mut okm = [0u8; OKM_LEN];
        hkdf
          .expand_vectored(&[head, &[], tail], &mut okm)
          .expect("test output length should be within the HKDF-SHA384 limit");
        assert_eq!(
          okm,
          expected,
          "hkdf-sha384 vectored mismatch kernel={} split={split}",
          id.as_str()
        );
      }
    }
  }

  #[test]
  fn hkdf_sha512_forced_kernels_match_oracle() {
    let caps = crate::platform::caps();
    for &id in SHA512_KERNELS {
      if caps.has(sha512_required_caps(id)) {
        assert_hkdf_sha512_kernel(id);
      }
    }
  }
}
