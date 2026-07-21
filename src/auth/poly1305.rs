//! Standalone Poly1305 one-time authenticator (RFC 8439).

#![allow(clippy::indexing_slicing)] // Poly1305 uses fixed 16-byte block and limb offsets.

use core::fmt;

use crate::{
  SecretBytes,
  secret::ZeroizingBytes,
  traits::{VerificationError, ct},
};

const KEY_SIZE: usize = 32;
const TAG_SIZE: usize = 16;
const LIMB_MASK: u32 = 0x03ff_ffff;
const FULL_BLOCK_HIBIT: u32 = 1 << 24;

#[inline]
fn load_u32_le(input: &[u8]) -> u32 {
  let mut bytes = [0u8; 4];
  bytes.copy_from_slice(input);
  u32::from_le_bytes(bytes)
}

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

#[derive(Default)]
struct State {
  r: [u32; 5],
  h: [u32; 5],
  pad: [u32; 4],
}

impl State {
  #[inline]
  fn new(key: &[u8; KEY_SIZE]) -> Self {
    Self {
      r: [
        load_u32_le(&key[0..4]) & LIMB_MASK,
        (load_u32_le(&key[3..7]) >> 2) & 0x03ff_ff03,
        (load_u32_le(&key[6..10]) >> 4) & 0x03ff_c0ff,
        (load_u32_le(&key[9..13]) >> 6) & 0x03f0_3fff,
        (load_u32_le(&key[12..16]) >> 8) & 0x000f_ffff,
      ],
      h: [0u32; 5],
      pad: [
        load_u32_le(&key[16..20]),
        load_u32_le(&key[20..24]),
        load_u32_le(&key[24..28]),
        load_u32_le(&key[28..32]),
      ],
    }
  }

  #[inline(always)]
  fn compute_block(&mut self, block: &[u8; 16], partial: bool) {
    let hibit = if partial { 0 } else { FULL_BLOCK_HIBIT };

    let r0 = self.r[0];
    let r1 = self.r[1];
    let r2 = self.r[2];
    let r3 = self.r[3];
    let r4 = self.r[4];

    let s1 = r1 * 5;
    let s2 = r2 * 5;
    let s3 = r3 * 5;
    let s4 = r4 * 5;

    let mut h0 = self.h[0];
    let mut h1 = self.h[1];
    let mut h2 = self.h[2];
    let mut h3 = self.h[3];
    let mut h4 = self.h[4];

    h0 = h0.wrapping_add(load_u32_le(&block[0..4]) & LIMB_MASK);
    h1 = h1.wrapping_add((load_u32_le(&block[3..7]) >> 2) & LIMB_MASK);
    h2 = h2.wrapping_add((load_u32_le(&block[6..10]) >> 4) & LIMB_MASK);
    h3 = h3.wrapping_add((load_u32_le(&block[9..13]) >> 6) & LIMB_MASK);
    h4 = h4.wrapping_add((load_u32_le(&block[12..16]) >> 8) | hibit);

    let d0 = (u64::from(h0) * u64::from(r0))
      + (u64::from(h1) * u64::from(s4))
      + (u64::from(h2) * u64::from(s3))
      + (u64::from(h3) * u64::from(s2))
      + (u64::from(h4) * u64::from(s1));
    let mut d1 = (u64::from(h0) * u64::from(r1))
      + (u64::from(h1) * u64::from(r0))
      + (u64::from(h2) * u64::from(s4))
      + (u64::from(h3) * u64::from(s3))
      + (u64::from(h4) * u64::from(s2));
    let mut d2 = (u64::from(h0) * u64::from(r2))
      + (u64::from(h1) * u64::from(r1))
      + (u64::from(h2) * u64::from(r0))
      + (u64::from(h3) * u64::from(s4))
      + (u64::from(h4) * u64::from(s3));
    let mut d3 = (u64::from(h0) * u64::from(r3))
      + (u64::from(h1) * u64::from(r2))
      + (u64::from(h2) * u64::from(r1))
      + (u64::from(h3) * u64::from(r0))
      + (u64::from(h4) * u64::from(s4));
    let mut d4 = (u64::from(h0) * u64::from(r4))
      + (u64::from(h1) * u64::from(r3))
      + (u64::from(h2) * u64::from(r2))
      + (u64::from(h3) * u64::from(r1))
      + (u64::from(h4) * u64::from(r0));

    let mut c = (d0 >> 26) as u32;
    h0 = (d0 as u32) & LIMB_MASK;
    d1 += u64::from(c);

    c = (d1 >> 26) as u32;
    h1 = (d1 as u32) & LIMB_MASK;
    d2 += u64::from(c);

    c = (d2 >> 26) as u32;
    h2 = (d2 as u32) & LIMB_MASK;
    d3 += u64::from(c);

    c = (d3 >> 26) as u32;
    h3 = (d3 as u32) & LIMB_MASK;
    d4 += u64::from(c);

    c = (d4 >> 26) as u32;
    h4 = (d4 as u32) & LIMB_MASK;
    h0 = h0.wrapping_add(c * 5);

    c = h0 >> 26;
    h0 &= LIMB_MASK;
    h1 = h1.wrapping_add(c);

    self.h = [h0, h1, h2, h3, h4];
  }

  #[inline(always)]
  fn finalize(self) -> [u8; TAG_SIZE] {
    let mut h0 = self.h[0];
    let mut h1 = self.h[1];
    let mut h2 = self.h[2];
    let mut h3 = self.h[3];
    let mut h4 = self.h[4];

    let mut c = h1 >> 26;
    h1 &= LIMB_MASK;
    h2 = h2.wrapping_add(c);

    c = h2 >> 26;
    h2 &= LIMB_MASK;
    h3 = h3.wrapping_add(c);

    c = h3 >> 26;
    h3 &= LIMB_MASK;
    h4 = h4.wrapping_add(c);

    c = h4 >> 26;
    h4 &= LIMB_MASK;
    h0 = h0.wrapping_add(c * 5);

    c = h0 >> 26;
    h0 &= LIMB_MASK;
    h1 = h1.wrapping_add(c);

    let mut g0 = h0.wrapping_add(5);
    c = g0 >> 26;
    g0 &= LIMB_MASK;

    let mut g1 = h1.wrapping_add(c);
    c = g1 >> 26;
    g1 &= LIMB_MASK;

    let mut g2 = h2.wrapping_add(c);
    c = g2 >> 26;
    g2 &= LIMB_MASK;

    let mut g3 = h3.wrapping_add(c);
    c = g3 >> 26;
    g3 &= LIMB_MASK;

    let mut g4 = h4.wrapping_add(c).wrapping_sub(1 << 26);

    let mut mask = (g4 >> 31).wrapping_sub(1);
    g0 &= mask;
    g1 &= mask;
    g2 &= mask;
    g3 &= mask;
    g4 &= mask;
    mask = !mask;

    h0 = (h0 & mask) | g0;
    h1 = (h1 & mask) | g1;
    h2 = (h2 & mask) | g2;
    h3 = (h3 & mask) | g3;
    h4 = (h4 & mask) | g4;

    h0 |= h1 << 26;
    h1 = (h1 >> 6) | (h2 << 20);
    h2 = (h2 >> 12) | (h3 << 14);
    h3 = (h3 >> 18) | (h4 << 8);

    let mut f = u64::from(h0) + u64::from(self.pad[0]);
    h0 = f as u32;
    f = u64::from(h1) + u64::from(self.pad[1]) + (f >> 32);
    h1 = f as u32;
    f = u64::from(h2) + u64::from(self.pad[2]) + (f >> 32);
    h2 = f as u32;
    f = u64::from(h3) + u64::from(self.pad[3]) + (f >> 32);
    h3 = f as u32;

    let mut tag = [0u8; TAG_SIZE];
    tag[0..4].copy_from_slice(&h0.to_le_bytes());
    tag[4..8].copy_from_slice(&h1.to_le_bytes());
    tag[8..12].copy_from_slice(&h2.to_le_bytes());
    tag[12..16].copy_from_slice(&h3.to_le_bytes());
    tag
  }
}

impl Drop for State {
  fn drop(&mut self) {
    ct::zeroize_words_no_fence(&mut self.r);
    ct::zeroize_words_no_fence(&mut self.h);
    ct::zeroize_words_no_fence(&mut self.pad);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
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
      let take = core::cmp::min(TAG_SIZE - self.buffer_len, data.len());
      self.buffer[self.buffer_len..self.buffer_len.strict_add(take)].copy_from_slice(&data[..take]);
      self.buffer_len = self.buffer_len.strict_add(take);
      data = &data[take..];

      if self.buffer_len == TAG_SIZE {
        self.state.compute_block(&self.buffer, false);
        ct::zeroize_no_fence(&mut self.buffer);
        self.buffer_len = 0;
      }
    }

    let mut blocks = data.chunks_exact(TAG_SIZE);
    for chunk in &mut blocks {
      let mut block = [0u8; TAG_SIZE];
      block.copy_from_slice(chunk);
      self.state.compute_block(&block, false);
      ct::zeroize_no_fence(&mut block);
    }

    let rem = blocks.remainder();
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
      self.state.compute_block(&self.buffer, true);
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
    unsafe { core::ptr::write_volatile(&mut self.buffer_len, 0) };
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}
