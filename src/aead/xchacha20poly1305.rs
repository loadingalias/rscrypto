#![allow(clippy::indexing_slicing)]

//! XChaCha20-Poly1305 public AEAD surface.

use core::fmt;

use super::{AeadBufferError, Nonce192, OpenError, SealError, chacha20, poly1305, targets::AeadPrimitive};
use crate::traits::{Aead, ct};

const KEY_SIZE: usize = chacha20::KEY_SIZE;
const TAG_SIZE: usize = 16;
const NONCE_SIZE: usize = Nonce192::LENGTH;
const MAX_PLAINTEXT_LEN: u64 = (u32::MAX as u64) * (chacha20::BLOCK_SIZE as u64);

define_aead_key_type!(XChaCha20Poly1305Key, KEY_SIZE, "XChaCha20-Poly1305 secret key bytes.");

define_aead_tag_type!(
  XChaCha20Poly1305Tag,
  TAG_SIZE,
  "XChaCha20-Poly1305 authentication tag bytes."
);

/// Portable XChaCha20-Poly1305 AEAD.
///
/// # Examples
///
/// ```
/// use rscrypto::{Aead, XChaCha20Poly1305, XChaCha20Poly1305Key, aead::Nonce192};
///
/// let key = XChaCha20Poly1305Key::from_bytes([0x42; 32]);
/// let nonce = Nonce192::from_bytes([0x24; 24]);
/// let cipher = XChaCha20Poly1305::new(&key);
///
/// let mut buf = *b"hello";
/// let tag = cipher.encrypt_in_place(&nonce, b"", &mut buf)?;
/// cipher.decrypt_in_place(&nonce, b"", &mut buf, &tag)?;
/// assert_eq!(&buf, b"hello");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Tampering is reported as an opaque verification failure.
///
/// ```
/// use rscrypto::{
///   Aead, XChaCha20Poly1305, XChaCha20Poly1305Key,
///   aead::{Nonce192, OpenError},
/// };
///
/// let key = XChaCha20Poly1305Key::from_bytes([0x42; 32]);
/// let nonce = Nonce192::from_bytes([0x24; 24]);
/// let cipher = XChaCha20Poly1305::new(&key);
///
/// let mut sealed = [0u8; 5 + XChaCha20Poly1305::TAG_SIZE];
/// cipher.encrypt(&nonce, b"", b"hello", &mut sealed)?;
/// sealed[0] ^= 1;
///
/// let mut opened = [0u8; 5];
/// assert_eq!(
///   cipher.decrypt(&nonce, b"", &sealed, &mut opened),
///   Err(OpenError::verification())
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct XChaCha20Poly1305 {
  key: XChaCha20Poly1305Key,
}

impl fmt::Debug for XChaCha20Poly1305 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("XChaCha20Poly1305").finish_non_exhaustive()
  }
}

impl XChaCha20Poly1305 {
  /// Key length in bytes.
  pub const KEY_SIZE: usize = KEY_SIZE;

  /// Nonce length in bytes.
  pub const NONCE_SIZE: usize = NONCE_SIZE;

  /// Tag length in bytes.
  pub const TAG_SIZE: usize = TAG_SIZE;

  /// Construct a new XChaCha20-Poly1305 instance from `key`.
  #[inline]
  #[must_use]
  pub fn new(key: &XChaCha20Poly1305Key) -> Self {
    <Self as Aead>::new(key)
  }

  /// Rebuild a typed tag from raw tag bytes.
  #[inline]
  pub fn tag_from_slice(bytes: &[u8]) -> Result<XChaCha20Poly1305Tag, AeadBufferError> {
    <Self as Aead>::tag_from_slice(bytes)
  }

  /// Encrypt `buffer` in place and return the detached authentication tag.
  #[inline]
  pub fn encrypt_in_place(
    &self,
    nonce: &Nonce192,
    aad: &[u8],
    buffer: &mut [u8],
  ) -> Result<XChaCha20Poly1305Tag, SealError> {
    <Self as Aead>::encrypt_in_place(self, nonce, aad, buffer)
  }

  /// Decrypt `buffer` in place and verify the detached authentication tag.
  #[inline]
  pub fn decrypt_in_place(
    &self,
    nonce: &Nonce192,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &XChaCha20Poly1305Tag,
  ) -> Result<(), OpenError> {
    <Self as Aead>::decrypt_in_place(self, nonce, aad, buffer, tag)
  }

  /// Encrypt `plaintext` into `out` as `ciphertext || tag`.
  #[inline]
  pub fn encrypt(&self, nonce: &Nonce192, aad: &[u8], plaintext: &[u8], out: &mut [u8]) -> Result<(), SealError> {
    <Self as Aead>::encrypt(self, nonce, aad, plaintext, out)
  }

  /// Decrypt a combined `ciphertext || tag` into `out`.
  #[inline]
  pub fn decrypt(
    &self,
    nonce: &Nonce192,
    aad: &[u8],
    ciphertext_and_tag: &[u8],
    out: &mut [u8],
  ) -> Result<(), OpenError> {
    <Self as Aead>::decrypt(self, nonce, aad, ciphertext_and_tag, out)
  }

  fn derive_subkey_and_nonce(&self, nonce: &Nonce192) -> ([u8; KEY_SIZE], [u8; chacha20::NONCE_SIZE]) {
    let mut h_nonce = [0u8; chacha20::HCHACHA_NONCE_SIZE];
    h_nonce.copy_from_slice(&nonce.as_bytes()[..chacha20::HCHACHA_NONCE_SIZE]);

    let subkey = chacha20::hchacha20(self.key.as_bytes(), &h_nonce);
    let mut ietf_nonce = [0u8; chacha20::NONCE_SIZE];
    ietf_nonce[4..].copy_from_slice(&nonce.as_bytes()[chacha20::HCHACHA_NONCE_SIZE..]);
    (subkey, ietf_nonce)
  }
}

impl Aead for XChaCha20Poly1305 {
  const KEY_SIZE: usize = KEY_SIZE;
  const NONCE_SIZE: usize = NONCE_SIZE;
  const TAG_SIZE: usize = TAG_SIZE;

  type Key = XChaCha20Poly1305Key;
  type Nonce = Nonce192;
  type Tag = XChaCha20Poly1305Tag;

  fn new(key: &Self::Key) -> Self {
    Self { key: key.clone() }
  }

  fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
    if bytes.len() != TAG_SIZE {
      return Err(AeadBufferError::new());
    }

    let mut tag = [0u8; TAG_SIZE];
    tag.copy_from_slice(bytes);
    Ok(XChaCha20Poly1305Tag::from_bytes(tag))
  }

  fn encrypt_in_place(&self, nonce: &Self::Nonce, aad: &[u8], buffer: &mut [u8]) -> Result<Self::Tag, SealError> {
    super::seal_bounded_length_as_u64(buffer.len(), MAX_PLAINTEXT_LEN)?;

    let (mut subkey, ietf_nonce) = self.derive_subkey_and_nonce(nonce);
    chacha20::xor_keystream(AeadPrimitive::XChaCha20Poly1305, &subkey, 1, &ietf_nonce, buffer)
      .map_err(|_| SealError::too_large())?;

    let mut poly_key = chacha20::poly1305_key_gen(&subkey, &ietf_nonce);
    let tag = XChaCha20Poly1305Tag::from_bytes(
      poly1305::authenticate_aead(AeadPrimitive::XChaCha20Poly1305, aad, buffer, &poly_key)
        .map_err(|_| SealError::too_large())?,
    );

    ct::zeroize(&mut poly_key);
    ct::zeroize(&mut subkey);
    Ok(tag)
  }

  fn decrypt_in_place(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), OpenError> {
    super::open_bounded_length_as_u64(buffer.len(), MAX_PLAINTEXT_LEN)?;

    // Derive subkey once and reuse for both tag verification and decryption.
    let (mut subkey, ietf_nonce) = self.derive_subkey_and_nonce(nonce);
    let mut poly_key = chacha20::poly1305_key_gen(&subkey, &ietf_nonce);
    let expected = poly1305::authenticate_aead(AeadPrimitive::XChaCha20Poly1305, aad, buffer, &poly_key)
      .map_err(|_| OpenError::too_large())?;
    ct::zeroize(&mut poly_key);

    if !ct::constant_time_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      ct::zeroize(&mut subkey);
      return Err(OpenError::verification());
    }

    chacha20::xor_keystream(AeadPrimitive::XChaCha20Poly1305, &subkey, 1, &ietf_nonce, buffer)
      .map_err(|_| OpenError::too_large())?;
    ct::zeroize(&mut subkey);
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn round_trip() {
    let key = XChaCha20Poly1305Key::from_bytes([0x42u8; 32]);
    let nonce = Nonce192::from_bytes([0x07u8; 24]);
    let cipher = XChaCha20Poly1305::new(&key);

    let mut buf = *b"hello xchacha";
    let tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buf).unwrap();
    cipher.decrypt_in_place(&nonce, b"aad", &mut buf, &tag).unwrap();
    assert_eq!(&buf, b"hello xchacha");
  }

  #[test]
  fn wrong_nonce_rejected() {
    let key = XChaCha20Poly1305Key::from_bytes([0x42u8; 32]);
    let nonce = Nonce192::from_bytes([0x07u8; 24]);
    let cipher = XChaCha20Poly1305::new(&key);

    let mut buf = *b"nonce test";
    let tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buf).unwrap();

    let wrong_nonce = Nonce192::from_bytes([0x08u8; 24]);
    let result = cipher.decrypt_in_place(&wrong_nonce, b"aad", &mut buf, &tag);
    assert!(result.is_err());
  }

  #[test]
  fn buffer_zeroed_on_auth_failure() {
    let key = XChaCha20Poly1305Key::from_bytes([0x42u8; 32]);
    let nonce = Nonce192::from_bytes([0x07u8; 24]);
    let cipher = XChaCha20Poly1305::new(&key);

    let mut buf = *b"zero me on failure";
    let tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buf).unwrap();

    let mut bad_tag = tag.to_bytes();
    bad_tag[0] ^= 0xFF;
    let bad_tag = XChaCha20Poly1305Tag::from_bytes(bad_tag);

    let result = cipher.decrypt_in_place(&nonce, b"aad", &mut buf, &bad_tag);
    assert!(result.is_err());
    assert!(buf.iter().all(|&b| b == 0), "buffer not zeroed on auth failure");
  }

  #[test]
  fn wrong_aad_rejected() {
    let key = XChaCha20Poly1305Key::from_bytes([0x42u8; 32]);
    let nonce = Nonce192::from_bytes([0x07u8; 24]);
    let cipher = XChaCha20Poly1305::new(&key);

    let mut buf = *b"aad test";
    let tag = cipher.encrypt_in_place(&nonce, b"correct", &mut buf).unwrap();

    let result = cipher.decrypt_in_place(&nonce, b"wrong", &mut buf, &tag);
    assert!(result.is_err());
  }
}
