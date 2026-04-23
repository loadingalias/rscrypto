//! Authenticated encryption with associated data traits.
//!
//! The core contract mirrors the rest of rscrypto:
//! explicit typed construction, byte-slice inputs, caller-provided output
//! buffers, and opaque verification failures.

use core::fmt::Debug;

use crate::aead::{AeadBufferError, OpenError, SealError};

/// Authenticated encryption with associated data.
///
/// The in-place methods mutate the payload buffer and return or consume the tag
/// separately. Combined ciphertext-plus-tag helpers are layered on top through
/// caller-provided output buffers so the trait remains `no_std` and `no_alloc`
/// friendly.
///
/// # Examples
///
/// ```
/// use rscrypto::{Aead, ChaCha20Poly1305, ChaCha20Poly1305Key, aead::Nonce96};
///
/// let key = ChaCha20Poly1305Key::from_bytes([0x42; 32]);
/// let nonce = Nonce96::from_bytes([0x24; 12]);
/// let cipher = ChaCha20Poly1305::new(&key);
///
/// // In-place encrypt with detached tag.
/// let mut buf = *b"hello";
/// let tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buf)?;
///
/// // In-place decrypt.
/// cipher.decrypt_in_place(&nonce, b"aad", &mut buf, &tag)?;
/// assert_eq!(&buf, b"hello");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait Aead: Clone {
  /// Key size in bytes.
  const KEY_SIZE: usize;

  /// Nonce size in bytes.
  const NONCE_SIZE: usize;

  /// Authentication tag size in bytes.
  const TAG_SIZE: usize;

  /// Algorithm-specific key wrapper.
  type Key: Clone + Eq;

  /// Algorithm-specific nonce wrapper.
  type Nonce: Copy + Eq + Debug + AsRef<[u8]>;

  /// Algorithm-specific tag wrapper.
  type Tag: Copy + Eq + Debug + AsRef<[u8]>;

  /// Construct a new AEAD instance from `key`.
  #[must_use]
  fn new(key: &Self::Key) -> Self;

  /// Rebuild a typed tag from raw tag bytes.
  ///
  /// Implementations must reject slices whose length is not exactly
  /// [`TAG_SIZE`](Self::TAG_SIZE).
  fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError>;

  /// Encrypt `buffer` in place and return the detached authentication tag.
  fn encrypt_in_place(&self, nonce: &Self::Nonce, aad: &[u8], buffer: &mut [u8]) -> Result<Self::Tag, SealError>;

  /// Decrypt `buffer` in place and verify the detached authentication tag.
  ///
  /// # Errors
  ///
  /// Returns [`OpenError`] when authentication fails or the input exceeds the
  /// algorithm's supported length bound.
  fn decrypt_in_place(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), OpenError>;

  /// Alias for [`encrypt_in_place`](Self::encrypt_in_place).
  #[inline]
  fn encrypt_in_place_detached(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
  ) -> Result<Self::Tag, SealError> {
    self.encrypt_in_place(nonce, aad, buffer)
  }

  /// Alias for [`decrypt_in_place`](Self::decrypt_in_place).
  #[inline]
  fn decrypt_in_place_detached(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), OpenError> {
    self.decrypt_in_place(nonce, aad, buffer, tag)
  }

  /// Return the required combined ciphertext-plus-tag length.
  ///
  /// # Errors
  ///
  /// Returns [`AeadBufferError`] if the requested length overflows `usize`.
  #[inline]
  fn ciphertext_len(plaintext_len: usize) -> Result<usize, AeadBufferError> {
    plaintext_len
      .checked_add(Self::TAG_SIZE)
      .ok_or_else(AeadBufferError::new)
  }

  /// Return the plaintext length contained in a combined ciphertext-plus-tag.
  ///
  /// # Errors
  ///
  /// Returns [`AeadBufferError`] if `ciphertext_and_tag_len` is smaller than
  /// [`TAG_SIZE`](Self::TAG_SIZE).
  #[inline]
  fn plaintext_len(ciphertext_and_tag_len: usize) -> Result<usize, AeadBufferError> {
    if ciphertext_and_tag_len < Self::TAG_SIZE {
      return Err(AeadBufferError::new());
    }

    Ok(ciphertext_and_tag_len.strict_sub(Self::TAG_SIZE))
  }

  /// Encrypt `plaintext` into `out` as `ciphertext || tag`.
  ///
  /// # Errors
  ///
  /// Returns [`SealError`] if `out.len()` does not match `plaintext.len() +
  /// TAG_SIZE`, that addition overflows, or the input exceeds the algorithm's
  /// supported length bound.
  #[inline]
  fn encrypt(&self, nonce: &Self::Nonce, aad: &[u8], plaintext: &[u8], out: &mut [u8]) -> Result<(), SealError> {
    let expected = Self::ciphertext_len(plaintext.len()).map_err(SealError::from)?;
    if out.len() != expected {
      return Err(SealError::buffer());
    }

    let (ciphertext, tag_out) = out.split_at_mut(plaintext.len());
    ciphertext.copy_from_slice(plaintext);
    let tag = match self.encrypt_in_place(nonce, aad, ciphertext) {
      Ok(tag) => tag,
      Err(err) => {
        super::ct::zeroize(out);
        return Err(err);
      }
    };
    tag_out.copy_from_slice(tag.as_ref());
    Ok(())
  }

  /// Decrypt a combined `ciphertext || tag` into `out`.
  ///
  /// # Errors
  ///
  /// Returns [`OpenError`] if the input is malformed, `out` has the wrong
  /// length, or authentication fails.
  #[inline]
  fn decrypt(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    ciphertext_and_tag: &[u8],
    out: &mut [u8],
  ) -> Result<(), OpenError> {
    let plaintext_len = Self::plaintext_len(ciphertext_and_tag.len())?;
    if out.len() != plaintext_len {
      return Err(OpenError::buffer());
    }

    let (ciphertext, tag_bytes) = ciphertext_and_tag.split_at(plaintext_len);
    out.copy_from_slice(ciphertext);
    let tag = Self::tag_from_slice(tag_bytes)?;
    if let Err(e) = self.decrypt_in_place(nonce, aad, out, &tag) {
      super::ct::zeroize(out);
      return Err(e);
    }
    Ok(())
  }
}
