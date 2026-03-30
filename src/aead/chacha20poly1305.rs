#![allow(clippy::indexing_slicing)]

//! ChaCha20-Poly1305 public AEAD surface.

use core::fmt;

use super::{AeadBufferError, Nonce96, OpenError, chacha20, poly1305, targets::AeadPrimitive};
use crate::traits::{Aead, VerificationError, ct};

const KEY_SIZE: usize = chacha20::KEY_SIZE;
const TAG_SIZE: usize = 16;
const NONCE_SIZE: usize = Nonce96::LENGTH;
const MAX_PLAINTEXT_LEN: u64 = (u32::MAX as u64) * (chacha20::BLOCK_SIZE as u64);

/// ChaCha20-Poly1305 secret key bytes.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ChaCha20Poly1305Key([u8; Self::LENGTH]);

impl ChaCha20Poly1305Key {
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

impl Default for ChaCha20Poly1305Key {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl AsRef<[u8]> for ChaCha20Poly1305Key {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

impl fmt::Debug for ChaCha20Poly1305Key {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("ChaCha20Poly1305Key").finish_non_exhaustive()
  }
}

impl Drop for ChaCha20Poly1305Key {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

/// ChaCha20-Poly1305 authentication tag bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChaCha20Poly1305Tag([u8; Self::LENGTH]);

impl ChaCha20Poly1305Tag {
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

impl Default for ChaCha20Poly1305Tag {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl AsRef<[u8]> for ChaCha20Poly1305Tag {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

impl fmt::Debug for ChaCha20Poly1305Tag {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_tuple("ChaCha20Poly1305Tag").field(&self.0).finish()
  }
}

/// Portable ChaCha20-Poly1305 AEAD.
#[derive(Clone)]
pub struct ChaCha20Poly1305 {
  key: ChaCha20Poly1305Key,
}

impl fmt::Debug for ChaCha20Poly1305 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("ChaCha20Poly1305").finish_non_exhaustive()
  }
}

impl ChaCha20Poly1305 {
  /// Key length in bytes.
  pub const KEY_SIZE: usize = KEY_SIZE;

  /// Nonce length in bytes.
  pub const NONCE_SIZE: usize = NONCE_SIZE;

  /// Tag length in bytes.
  pub const TAG_SIZE: usize = TAG_SIZE;

  /// Construct a new ChaCha20-Poly1305 instance from `key`.
  #[inline]
  #[must_use]
  pub fn new(key: &ChaCha20Poly1305Key) -> Self {
    <Self as Aead>::new(key)
  }

  /// Rebuild a typed tag from raw tag bytes.
  #[inline]
  pub fn tag_from_slice(bytes: &[u8]) -> Result<ChaCha20Poly1305Tag, AeadBufferError> {
    <Self as Aead>::tag_from_slice(bytes)
  }

  /// Encrypt `buffer` in place and return the detached authentication tag.
  #[inline]
  #[must_use]
  pub fn encrypt_in_place(&self, nonce: &Nonce96, aad: &[u8], buffer: &mut [u8]) -> ChaCha20Poly1305Tag {
    <Self as Aead>::encrypt_in_place(self, nonce, aad, buffer)
  }

  /// Decrypt `buffer` in place and verify the detached authentication tag.
  #[inline]
  pub fn decrypt_in_place(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &ChaCha20Poly1305Tag,
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

  fn ensure_message_len(len: usize) {
    let len = match u64::try_from(len) {
      Ok(value) => value,
      Err(_) => panic!("message length exceeds u64"),
    };
    assert!(len <= MAX_PLAINTEXT_LEN, "ChaCha20-Poly1305 message too large");
  }

  fn compute_tag(&self, nonce: &Nonce96, aad: &[u8], ciphertext: &[u8]) -> [u8; TAG_SIZE] {
    let mut poly_key = chacha20::poly1305_key_gen(self.key.as_bytes(), nonce.as_bytes());
    let tag = poly1305::authenticate_aead(AeadPrimitive::ChaCha20Poly1305, aad, ciphertext, &poly_key);
    ct::zeroize(&mut poly_key);
    tag
  }
}

impl Aead for ChaCha20Poly1305 {
  const KEY_SIZE: usize = KEY_SIZE;
  const NONCE_SIZE: usize = NONCE_SIZE;
  const TAG_SIZE: usize = TAG_SIZE;

  type Key = ChaCha20Poly1305Key;
  type Nonce = Nonce96;
  type Tag = ChaCha20Poly1305Tag;

  fn new(key: &Self::Key) -> Self {
    Self { key: key.clone() }
  }

  fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
    if bytes.len() != TAG_SIZE {
      return Err(AeadBufferError::new());
    }

    let mut tag = [0u8; TAG_SIZE];
    tag.copy_from_slice(bytes);
    Ok(ChaCha20Poly1305Tag::from_bytes(tag))
  }

  fn encrypt_in_place(&self, nonce: &Self::Nonce, aad: &[u8], buffer: &mut [u8]) -> Self::Tag {
    Self::ensure_message_len(buffer.len());

    chacha20::xor_keystream(
      AeadPrimitive::ChaCha20Poly1305,
      self.key.as_bytes(),
      1,
      nonce.as_bytes(),
      buffer,
    );

    let mut poly_key = chacha20::poly1305_key_gen(self.key.as_bytes(), nonce.as_bytes());
    let tag = ChaCha20Poly1305Tag::from_bytes(poly1305::authenticate_aead(
      AeadPrimitive::ChaCha20Poly1305,
      aad,
      buffer,
      &poly_key,
    ));
    ct::zeroize(&mut poly_key);
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

    let expected = self.compute_tag(nonce, aad, buffer);
    if !ct::constant_time_eq(&expected, tag.as_bytes()) {
      return Err(VerificationError::new());
    }

    chacha20::xor_keystream(
      AeadPrimitive::ChaCha20Poly1305,
      self.key.as_bytes(),
      1,
      nonce.as_bytes(),
      buffer,
    );
    Ok(())
  }
}
