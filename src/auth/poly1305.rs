//! Standalone Poly1305 one-time authenticator (RFC 8439).

use core::fmt;

use crate::{
  SecretBytes,
  backend::poly1305::State,
  secret::ZeroizingBytes,
  traits::{VerificationError, ct},
};

const KEY_SIZE: usize = 32;
const TAG_SIZE: usize = 16;
/// Poly1305 one-time key.
///
/// This type is intentionally not `Clone` or `Copy`. Poly1305 keys must be
/// single-use at the protocol layer.
pub struct Poly1305OneTimeKey([u8; Self::LENGTH]);

impl Poly1305OneTimeKey {
  /// Poly1305 one-time key length in bytes.
  pub const LENGTH: usize = KEY_SIZE;

  /// Construct a one-time key from raw bytes.
  #[inline]
  #[must_use]
  pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
    Self(bytes)
  }

  /// Borrow the one-time key bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }

  /// Explicitly extract the key bytes into a zeroizing wrapper.
  #[inline]
  #[must_use]
  pub fn expose_secret(&self) -> SecretBytes<{ Self::LENGTH }> {
    SecretBytes::new(self.0)
  }

  /// Generate a one-time key with caller-supplied entropy.
  #[inline]
  pub fn try_generate_with<E>(mut fill: impl FnMut(&mut [u8]) -> Result<(), E>) -> Result<Self, E> {
    let mut bytes = ZeroizingBytes::zeroed();
    fill(bytes.as_mut_array())?;
    Ok(Self::from_bytes(*bytes.as_array()))
  }

  /// Generate a one-time key from the platform entropy source.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[inline]
  pub fn try_generate() -> Result<Self, getrandom::Error> {
    Self::try_generate_with(getrandom::fill)
  }
}

impl Drop for Poly1305OneTimeKey {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

impl fmt::Debug for Poly1305OneTimeKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("Poly1305OneTimeKey(****)")
  }
}

/// Poly1305 authentication tag.
#[derive(Clone, Copy)]
pub struct Poly1305Tag([u8; Self::LENGTH]);

impl core::hash::Hash for Poly1305Tag {
  #[inline]
  fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
    core::hash::Hash::hash(&self.0, state);
  }
}

impl Poly1305Tag {
  /// Poly1305 tag length in bytes.
  pub const LENGTH: usize = TAG_SIZE;

  /// Compare two tags without exposing a branchable boolean.
  #[inline]
  pub fn ct_eq(&self, other: &Self) -> ct::CtDecision {
    ct::fixed_eq(&self.0, &other.0)
  }

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

  /// Return the tag bytes.
  #[inline]
  #[must_use]
  pub const fn into_bytes(self) -> [u8; Self::LENGTH] {
    self.0
  }

  /// Borrow the tag bytes as a fixed-size array.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }

  /// Borrow the tag bytes as a slice.
  #[inline]
  #[must_use]
  pub fn as_slice(&self) -> &[u8] {
    &self.0
  }
}

impl Default for Poly1305Tag {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl From<[u8; TAG_SIZE]> for Poly1305Tag {
  #[inline]
  fn from(bytes: [u8; TAG_SIZE]) -> Self {
    Self::from_bytes(bytes)
  }
}

impl From<Poly1305Tag> for [u8; TAG_SIZE] {
  #[inline]
  fn from(tag: Poly1305Tag) -> Self {
    tag.to_bytes()
  }
}

impl TryFrom<&[u8]> for Poly1305Tag {
  type Error = core::array::TryFromSliceError;

  #[inline]
  fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
    Ok(Self::from_bytes(bytes.try_into()?))
  }
}

impl AsRef<[u8]> for Poly1305Tag {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

impl AsRef<[u8; TAG_SIZE]> for Poly1305Tag {
  #[inline]
  fn as_ref(&self) -> &[u8; TAG_SIZE] {
    &self.0
  }
}

impl fmt::Debug for Poly1305Tag {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Poly1305Tag(")?;
    for byte in self.0 {
      write!(f, "{byte:02x}")?;
    }
    write!(f, ")")
  }
}

/// Streaming Poly1305 authenticator.
///
/// Construction consumes a [`Poly1305OneTimeKey`]. Finalization consumes the
/// authenticator so the keyed state cannot be reset and reused.
pub struct Poly1305 {
  state: State,
  buffer: [u8; TAG_SIZE],
  buffer_len: usize,
}

impl fmt::Debug for Poly1305 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Poly1305").finish_non_exhaustive()
  }
}

impl Poly1305 {
  /// Poly1305 key size in bytes.
  pub const KEY_SIZE: usize = KEY_SIZE;

  /// Poly1305 tag size in bytes.
  pub const TAG_SIZE: usize = TAG_SIZE;

  /// Construct a streaming Poly1305 authenticator, consuming the one-time key.
  #[inline]
  #[must_use]
  pub fn new(key: Poly1305OneTimeKey) -> Self {
    let state = State::new(key.as_bytes());
    drop(key);
    Self {
      state,
      buffer: [0u8; TAG_SIZE],
      buffer_len: 0,
    }
  }

  /// Absorb more message bytes.
  #[inline]
  pub fn update(&mut self, mut data: &[u8]) {
    if self.buffer_len != 0 {
      let take = core::cmp::min(TAG_SIZE.strict_sub(self.buffer_len), data.len());
      self.buffer[self.buffer_len..self.buffer_len.strict_add(take)].copy_from_slice(&data[..take]);
      self.buffer_len = self.buffer_len.strict_add(take);
      data = &data[take..];

      if self.buffer_len == TAG_SIZE {
        self.state.compute_block_portable(&self.buffer, false);
        ct::zeroize_no_fence(&mut self.buffer);
        self.buffer_len = 0;
      }
    }

    let (blocks, rem) = data.as_chunks::<TAG_SIZE>();
    for block in blocks {
      self.state.compute_block_portable(block, false);
    }

    if !rem.is_empty() {
      self.buffer[..rem.len()].copy_from_slice(rem);
      self.buffer_len = rem.len();
    }
  }

  /// Finalize and return the Poly1305 tag.
  #[inline]
  #[must_use]
  pub fn finalize(mut self) -> Poly1305Tag {
    if self.buffer_len != 0 {
      self.buffer[self.buffer_len] = 1;
      self.state.compute_block_portable(&self.buffer, true);
    }
    ct::zeroize_no_fence(&mut self.buffer);
    self.buffer_len = 0;
    Poly1305Tag::from_bytes(core::mem::take(&mut self.state).finalize())
  }

  /// Verify `expected` through the tag owner's sealed comparison decision.
  ///
  /// Generated-code timing claims are configuration- and release-evidence-bound;
  /// see `ct.toml`.
  #[inline]
  #[must_use = "Poly1305 verification must be checked; a dropped Result silently accepts a forged tag"]
  pub fn verify(self, expected: &Poly1305Tag) -> Result<(), VerificationError> {
    if self.finalize().ct_eq(expected).declassify() {
      Ok(())
    } else {
      Err(VerificationError::new())
    }
  }

  /// Compute a one-shot Poly1305 tag, consuming the one-time key.
  #[inline]
  #[must_use]
  pub fn authenticate_once(key: Poly1305OneTimeKey, data: &[u8]) -> Poly1305Tag {
    let mut authenticator = Self::new(key);
    authenticator.update(data);
    authenticator.finalize()
  }

  /// Verify a one-shot Poly1305 tag, consuming the one-time key.
  #[inline]
  #[must_use = "Poly1305 verification must be checked; a dropped Result silently accepts a forged tag"]
  pub fn verify_once(key: Poly1305OneTimeKey, data: &[u8], expected: &Poly1305Tag) -> Result<(), VerificationError> {
    Self::authenticate_once(key, data)
      .ct_eq(expected)
      .declassify()
      .then_some(())
      .ok_or_else(VerificationError::new)
  }
}

impl Drop for Poly1305 {
  fn drop(&mut self) {
    ct::zeroize(&mut self.buffer);
    // SAFETY: field is a valid, aligned, dereferenceable pointer to initialized memory.
    unsafe { core::ptr::write_volatile(&raw mut self.buffer_len, 0) };
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}
