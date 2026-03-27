//! HKDF-SHA256 (RFC 5869).

use core::fmt;

use super::hmac::HmacSha256;
use crate::traits::{Mac, ct};

const OUTPUT_SIZE: usize = 32;
const MAX_OUTPUT_SIZE: usize = 255 * OUTPUT_SIZE;

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
  /// Cached HMAC keyed with PRK. Avoids re-compressing ipad/opad (2 SHA-256
  /// compressions) on every expand iteration.
  prk_mac: HmacSha256,
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
    let prk_mac = HmacSha256::new(&prk);
    Self { prk, prk_mac }
  }

  /// Expand the stored pseudorandom key into `okm`.
  pub fn expand(&self, info: &[u8], okm: &mut [u8]) -> Result<(), HkdfOutputLengthError> {
    if okm.len() > MAX_OUTPUT_SIZE {
      return Err(HkdfOutputLengthError::new());
    }

    if okm.is_empty() {
      return Ok(());
    }

    // Clone the cached HMAC once. Each iteration resets the inner state
    // (one memcpy) instead of re-creating from scratch (2 SHA-256 compressions).
    let mut mac = self.prk_mac.clone();
    let mut block = [0u8; OUTPUT_SIZE];
    let mut have_block = false;
    let mut counter = 1u8;

    for chunk in okm.chunks_mut(OUTPUT_SIZE) {
      if have_block {
        mac.reset();
        mac.update(&block);
      }
      mac.update(info);
      mac.update(&[counter]);

      block = mac.finalize();
      for (dst, src) in chunk.iter_mut().zip(block.iter()) {
        *dst = *src;
      }
      have_block = true;
      counter = counter.wrapping_add(1);
    }

    ct::zeroize(&mut block);
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
  }
}
