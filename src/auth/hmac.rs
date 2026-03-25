//! HMAC-SHA256 (RFC 2104, FIPS 198-1).

use crate::{
  hashes::crypto::Sha256,
  traits::{Digest, Mac, VerificationError, ct},
};

const BLOCK_SIZE: usize = 64;
const TAG_SIZE: usize = 32;

/// HMAC-SHA256 authentication state.
///
/// # Examples
///
/// ```rust
/// use rscrypto::{HmacSha256, Mac};
///
/// let key = b"shared-secret";
/// let data = b"auth message";
///
/// let tag = HmacSha256::mac(key, data);
///
/// let mut mac = HmacSha256::new(key);
/// mac.update(b"auth ");
/// mac.update(b"message");
/// assert_eq!(mac.finalize(), tag);
/// assert!(mac.verify(&tag).is_ok());
/// ```
#[derive(Clone)]
pub struct HmacSha256 {
  inner: Sha256,
  inner_init: Sha256,
  outer_init: Sha256,
}

impl core::fmt::Debug for HmacSha256 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("HmacSha256").finish_non_exhaustive()
  }
}

impl HmacSha256 {
  /// HMAC-SHA256 block size in bytes.
  pub const BLOCK_SIZE: usize = BLOCK_SIZE;

  /// HMAC-SHA256 tag size in bytes.
  pub const TAG_SIZE: usize = TAG_SIZE;

  /// Compute the HMAC tag of `data` in one shot.
  #[inline]
  #[must_use]
  pub fn mac(key: &[u8], data: &[u8]) -> [u8; TAG_SIZE] {
    <Self as Mac>::mac(key, data)
  }

  /// Verify `expected` against the HMAC tag of `data` in constant time.
  #[inline]
  pub fn verify_tag(key: &[u8], data: &[u8], expected: &[u8; TAG_SIZE]) -> Result<(), VerificationError> {
    <Self as Mac>::verify_tag(key, data, expected)
  }
}

impl Mac for HmacSha256 {
  const TAG_SIZE: usize = TAG_SIZE;
  type Tag = [u8; TAG_SIZE];

  fn new(key: &[u8]) -> Self {
    let mut key_block = [0u8; BLOCK_SIZE];
    if key.len() > BLOCK_SIZE {
      let digest = Sha256::digest(key);
      for (dst, src) in key_block.iter_mut().zip(digest.iter().copied()) {
        *dst = src;
      }
    } else {
      for (dst, src) in key_block.iter_mut().zip(key.iter().copied()) {
        *dst = src;
      }
    }

    let mut ipad = [0x36u8; BLOCK_SIZE];
    let mut opad = [0x5Cu8; BLOCK_SIZE];
    for ((ipad_byte, opad_byte), key_byte) in ipad.iter_mut().zip(opad.iter_mut()).zip(key_block.iter().copied()) {
      *ipad_byte ^= key_byte;
      *opad_byte ^= key_byte;
    }

    let mut inner_init = Sha256::new();
    inner_init.update(&ipad);

    let mut outer_init = Sha256::new();
    outer_init.update(&opad);

    ct::zeroize(&mut key_block);
    ct::zeroize(&mut ipad);
    ct::zeroize(&mut opad);

    Self {
      inner: inner_init.clone(),
      inner_init,
      outer_init,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.inner.update(data);
  }

  #[inline]
  fn finalize(&self) -> Self::Tag {
    let inner_hash = self.inner.finalize();
    let mut outer = self.outer_init.clone();
    outer.update(&inner_hash);
    outer.finalize()
  }

  #[inline]
  fn reset(&mut self) {
    self.inner = self.inner_init.clone();
  }

  #[inline]
  fn verify(&self, expected: &Self::Tag) -> Result<(), VerificationError> {
    if ct::constant_time_eq(&self.finalize(), expected) {
      Ok(())
    } else {
      Err(VerificationError::new())
    }
  }
}
