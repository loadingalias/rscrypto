//! cSHAKE256 (SP 800-185).

#![allow(clippy::indexing_slicing)] // Fixed-width prefix encodings and rate-sized zero padding.

#[cfg(feature = "auth")]
use super::sp800185::absorb_bytepad;
use super::{
  keccak::{KeccakCore, KeccakXof},
  sp800185::{RATE_256, absorb_encoded_string, encoded_string_len, left_encode},
};
use crate::traits::Xof;

const SHAKE_PAD: u8 = 0x1F;
const CSHAKE_PAD: u8 = 0x04;

/// cSHAKE256 hasher with explicit function-name and customization strings.
#[derive(Clone)]
pub struct Cshake256 {
  core: KeccakCore<RATE_256>,
  initial_state: KeccakCore<RATE_256>,
  pad: u8,
}

impl core::fmt::Debug for Cshake256 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Cshake256").finish_non_exhaustive()
  }
}

impl Cshake256 {
  /// Construct a cSHAKE256 state from `function_name` and `customization`.
  #[inline]
  #[must_use]
  pub fn new(function_name: &[u8], customization: &[u8]) -> Self {
    if function_name.is_empty() && customization.is_empty() {
      let core = KeccakCore::<RATE_256>::default();
      return Self {
        core: core.clone(),
        initial_state: core,
        pad: SHAKE_PAD,
      };
    }

    let mut core = KeccakCore::<RATE_256>::default();
    let (rate_prefix, rate_prefix_len) = left_encode(RATE_256 as u64);
    let fn_len = encoded_string_len(function_name);
    let custom_len = encoded_string_len(customization);
    core.update(&rate_prefix[..rate_prefix_len]);
    absorb_encoded_string::<RATE_256>(&mut core, function_name);
    absorb_encoded_string::<RATE_256>(&mut core, customization);
    let total_len = rate_prefix_len.strict_add(fn_len).strict_add(custom_len);
    let pad_len = (RATE_256.strict_sub(total_len % RATE_256)) % RATE_256;
    if pad_len != 0 {
      core.update(&[0u8; RATE_256][..pad_len]);
    }

    let initial_state = core.clone();
    Self {
      core,
      initial_state,
      pad: CSHAKE_PAD,
    }
  }

  /// Compute a one-shot cSHAKE256 XOF reader.
  #[inline]
  #[must_use]
  pub fn xof(function_name: &[u8], customization: &[u8], data: &[u8]) -> Cshake256Xof {
    let mut hasher = Self::new(function_name, customization);
    hasher.update(data);
    hasher.finalize_xof()
  }

  /// Absorb additional message bytes.
  #[inline]
  pub fn update(&mut self, data: &[u8]) {
    self.core.update(data);
  }

  #[cfg(feature = "auth")]
  #[inline]
  pub(crate) fn absorb_bytepad_segments(&mut self, segments: &[&[u8]], payload_len: usize) {
    absorb_bytepad(&mut self.core, segments, payload_len);
  }

  /// Finalize into an extendable-output reader.
  #[inline]
  #[must_use]
  pub fn finalize_xof(&self) -> Cshake256Xof {
    Cshake256Xof {
      inner: self.core.finalize_xof(self.pad),
    }
  }

  /// Reset back to the initial cSHAKE256 prefix state.
  #[inline]
  pub fn reset(&mut self) {
    self.core = self.initial_state.clone();
  }

  /// Fill `out` in one shot.
  #[inline]
  pub fn hash_into(function_name: &[u8], customization: &[u8], data: &[u8], out: &mut [u8]) {
    Self::xof(function_name, customization, data).squeeze(out);
  }
}

/// cSHAKE256 output reader.
#[derive(Clone)]
pub struct Cshake256Xof {
  inner: KeccakXof<RATE_256>,
}

impl core::fmt::Debug for Cshake256Xof {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Cshake256Xof").finish_non_exhaustive()
  }
}

impl Xof for Cshake256Xof {
  #[inline]
  fn squeeze(&mut self, out: &mut [u8]) {
    self.inner.squeeze_into(out);
  }
}

#[cfg(test)]
mod tests {
  use alloc::vec;

  use super::Cshake256;
  use crate::traits::Xof;

  fn squeeze_hex(mut reader: impl Xof, len: usize) -> alloc::string::String {
    use alloc::string::String;
    use core::fmt::Write;

    let mut out = vec![0u8; len];
    reader.squeeze(&mut out);

    let mut hex = String::new();
    for byte in out {
      write!(&mut hex, "{byte:02x}").unwrap();
    }
    hex
  }

  #[test]
  fn empty_function_and_customization_matches_shake256() {
    let ours = squeeze_hex(Cshake256::xof(b"", b"", b"abc"), 64);
    let expected = squeeze_hex(crate::hashes::crypto::Shake256::xof(b"abc"), 64);
    assert_eq!(ours, expected);
  }

  #[test]
  fn reset_restores_prefix_state() {
    let mut hasher = Cshake256::new(b"KMAC", b"custom");
    hasher.update(b"abc");
    let expected = squeeze_hex(hasher.finalize_xof(), 64);
    hasher.reset();
    hasher.update(b"abc");
    let actual = squeeze_hex(hasher.finalize_xof(), 64);
    assert_eq!(actual, expected);
  }
}
