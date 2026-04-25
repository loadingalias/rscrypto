//! KMAC256 (SP 800-185).

#![allow(clippy::indexing_slicing)] // Fixed-size scratch buffers and encoded suffix slices.

use core::fmt;

use crate::{
  hashes::crypto::{
    Cshake256,
    sp800185::{left_encode, right_encode},
  },
  traits::{VerificationError, Xof, ct},
};

/// KMAC256 keyed state.
///
/// KMAC256 is a variable-output MAC/PRF. It intentionally does not implement
/// [`crate::traits::Mac`], which assumes a fixed-size tag.
#[derive(Clone)]
pub struct Kmac256 {
  state: Cshake256,
  initial_state: Cshake256,
}

impl fmt::Debug for Kmac256 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Kmac256").finish_non_exhaustive()
  }
}

impl Kmac256 {
  /// Construct a new KMAC256 state keyed by `key` and domain-separated by
  /// `customization`.
  #[must_use]
  pub fn new(key: &[u8], customization: &[u8]) -> Self {
    let mut state = Cshake256::new(b"KMAC", customization);
    Self::absorb_key(&mut state, key);
    let initial_state = state.clone();
    Self { state, initial_state }
  }

  #[inline]
  fn absorb_key(state: &mut Cshake256, key: &[u8]) {
    let (key_prefix, key_prefix_len) = left_encode(crate::bytes_to_bits_saturating(key.len()));
    let payload_len = key_prefix_len.strict_add(key.len());
    state.absorb_bytepad_segments(&[&key_prefix[..key_prefix_len], key], payload_len);
  }

  #[inline]
  fn finalize_reader(&self, output_len: usize) -> impl Xof {
    let mut state = self.state.clone();
    let (suffix, suffix_len) = right_encode(crate::bytes_to_bits_saturating(output_len));
    state.update(&suffix[..suffix_len]);
    state.finalize_xof()
  }

  /// Absorb more message bytes.
  #[inline]
  pub fn update(&mut self, data: &[u8]) {
    self.state.update(data);
  }

  /// Finalize into `out`.
  #[inline]
  pub fn finalize_into(&self, out: &mut [u8]) {
    let mut reader = self.finalize_reader(out.len());
    reader.squeeze(out);
  }

  /// Reset back to the keyed initial state.
  #[inline]
  pub fn reset(&mut self) {
    self.state = self.initial_state.clone();
  }

  /// Compute a one-shot KMAC256 output into `out`.
  #[inline]
  pub fn mac_into(key: &[u8], customization: &[u8], data: &[u8], out: &mut [u8]) {
    let mut state = Self::new(key, customization);
    state.update(data);
    state.finalize_into(out);
  }

  /// Compute a one-shot KMAC256 output into a fixed-size array.
  pub fn mac_array<const N: usize>(key: &[u8], customization: &[u8], data: &[u8]) -> [u8; N] {
    let mut out = [0u8; N];
    Self::mac_into(key, customization, data, &mut out);
    out
  }

  /// Verify `expected` against the MAC of `data` in constant time.
  ///
  /// This is the one-shot helper. For pre-release snapshots that inverted the
  /// naming, use `verify_tag` for `(key, customization, data, expected)` and
  /// [`Self::verify`] for an already-accumulated state.
  #[must_use = "MAC verification must be checked; a dropped Result silently accepts a forged tag"]
  pub fn verify_tag(key: &[u8], customization: &[u8], data: &[u8], expected: &[u8]) -> Result<(), VerificationError> {
    let mut state = Self::new(key, customization);
    state.update(data);
    state.verify(expected)
  }

  /// Verify `expected` against the current KMAC256 output in constant time.
  ///
  /// This checks the MAC for the bytes already absorbed into `self`; it does
  /// not recompute from `(key, customization, data)` like [`Self::verify_tag`].
  #[must_use = "MAC verification must be checked; a dropped Result silently accepts a forged tag"]
  pub fn verify(&self, expected: &[u8]) -> Result<(), VerificationError> {
    let mut reader = self.finalize_reader(expected.len());
    let mut diff = 0u8;
    let mut block = [0u8; 64];

    for chunk in expected.chunks(block.len()) {
      reader.squeeze(&mut block[..chunk.len()]);
      diff |= u8::from(!ct::constant_time_eq(&block[..chunk.len()], chunk));
    }

    ct::zeroize(&mut block);
    if core::hint::black_box(diff) == 0 {
      Ok(())
    } else {
      Err(VerificationError::new())
    }
  }
}

#[cfg(test)]
mod tests {
  use super::Kmac256;

  #[test]
  fn reset_restores_keyed_state() {
    let mut kmac = Kmac256::new(b"key", b"custom");
    kmac.update(b"abc");
    let expected = Kmac256::mac_array::<32>(b"key", b"custom", b"abc");

    let mut actual = [0u8; 32];
    kmac.finalize_into(&mut actual);
    assert_eq!(actual, expected);

    kmac.reset();
    kmac.update(b"abc");
    kmac.finalize_into(&mut actual);
    assert_eq!(actual, expected);
  }
}
