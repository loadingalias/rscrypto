#![allow(clippy::indexing_slicing)]

//! XChaCha20-Poly1305 public AEAD surface.

use core::fmt;

use super::{AeadBufferError, Nonce192, OpenError, chacha20, poly1305, targets::AeadPrimitive};
use crate::traits::{Aead, VerificationError, ct};

const KEY_SIZE: usize = chacha20::KEY_SIZE;
const TAG_SIZE: usize = 16;
const NONCE_SIZE: usize = Nonce192::LENGTH;
const MAX_PLAINTEXT_LEN: u64 = (u32::MAX as u64) * (chacha20::BLOCK_SIZE as u64);

/// XChaCha20-Poly1305 secret key bytes.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct XChaCha20Poly1305Key([u8; Self::LENGTH]);

impl XChaCha20Poly1305Key {
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

impl Default for XChaCha20Poly1305Key {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl AsRef<[u8]> for XChaCha20Poly1305Key {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

impl fmt::Debug for XChaCha20Poly1305Key {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("XChaCha20Poly1305Key(****)")
  }
}

impl XChaCha20Poly1305Key {
  /// Construct a key by filling bytes from the provided closure.
  ///
  /// ```ignore
  /// let key = XChaCha20Poly1305Key::generate(|buf| getrandom::fill(buf).unwrap());
  /// ```
  #[inline]
  #[must_use]
  pub fn generate(fill: impl FnOnce(&mut [u8; Self::LENGTH])) -> Self {
    let mut bytes = [0u8; Self::LENGTH];
    fill(&mut bytes);
    Self(bytes)
  }
}

impl_hex_fmt_secret!(XChaCha20Poly1305Key);

impl Drop for XChaCha20Poly1305Key {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

/// XChaCha20-Poly1305 authentication tag bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct XChaCha20Poly1305Tag([u8; Self::LENGTH]);

impl XChaCha20Poly1305Tag {
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

impl Default for XChaCha20Poly1305Tag {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl AsRef<[u8]> for XChaCha20Poly1305Tag {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

impl fmt::Debug for XChaCha20Poly1305Tag {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "XChaCha20Poly1305Tag(")?;
    crate::hex::fmt_hex_lower(&self.0, f)?;
    write!(f, ")")
  }
}

impl_hex_fmt!(XChaCha20Poly1305Tag);

/// Portable XChaCha20-Poly1305 AEAD.
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
  #[must_use]
  pub fn encrypt_in_place(&self, nonce: &Nonce192, aad: &[u8], buffer: &mut [u8]) -> XChaCha20Poly1305Tag {
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
  ) -> Result<(), VerificationError> {
    <Self as Aead>::decrypt_in_place(self, nonce, aad, buffer, tag)
  }

  /// Encrypt `plaintext` into `out` as `ciphertext || tag`.
  #[inline]
  pub fn encrypt(&self, nonce: &Nonce192, aad: &[u8], plaintext: &[u8], out: &mut [u8]) -> Result<(), AeadBufferError> {
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

  fn ensure_message_len(len: usize) {
    let len = match u64::try_from(len) {
      Ok(value) => value,
      Err(_) => panic!("message length exceeds u64"),
    };
    assert!(len <= MAX_PLAINTEXT_LEN, "XChaCha20-Poly1305 message too large");
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

  fn encrypt_in_place(&self, nonce: &Self::Nonce, aad: &[u8], buffer: &mut [u8]) -> Self::Tag {
    Self::ensure_message_len(buffer.len());

    let (mut subkey, ietf_nonce) = self.derive_subkey_and_nonce(nonce);
    chacha20::xor_keystream(AeadPrimitive::XChaCha20Poly1305, &subkey, 1, &ietf_nonce, buffer);

    let mut poly_key = chacha20::poly1305_key_gen(&subkey, &ietf_nonce);
    let tag = XChaCha20Poly1305Tag::from_bytes(poly1305::authenticate_aead(
      AeadPrimitive::XChaCha20Poly1305,
      aad,
      buffer,
      &poly_key,
    ));

    ct::zeroize(&mut poly_key);
    ct::zeroize(&mut subkey);
    tag
  }

  fn decrypt_in_place(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), VerificationError> {
    Self::ensure_message_len(buffer.len());

    // Derive subkey once and reuse for both tag verification and decryption.
    let (mut subkey, ietf_nonce) = self.derive_subkey_and_nonce(nonce);
    let mut poly_key = chacha20::poly1305_key_gen(&subkey, &ietf_nonce);
    let expected = poly1305::authenticate_aead(AeadPrimitive::XChaCha20Poly1305, aad, buffer, &poly_key);
    ct::zeroize(&mut poly_key);

    if !ct::constant_time_eq(&expected, tag.as_bytes()) {
      ct::zeroize(&mut subkey);
      return Err(VerificationError::new());
    }

    chacha20::xor_keystream(AeadPrimitive::XChaCha20Poly1305, &subkey, 1, &ietf_nonce, buffer);
    ct::zeroize(&mut subkey);
    Ok(())
  }
}
