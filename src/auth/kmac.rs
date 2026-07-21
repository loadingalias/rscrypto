//! KMAC128 and KMAC256 (SP 800-185).

#![allow(clippy::indexing_slicing)] // Fixed-size scratch buffers and encoded suffix slices.

use core::fmt;

use crate::{
  hashes::crypto::{
    Cshake128, Cshake256,
    sp800185::{left_encode, right_encode},
  },
  traits::{VerificationError, Xof, ct},
};

macro_rules! define_kmac {
  ($name:ident, $cshake:ident, $bits:literal) => {
    #[doc = concat!("KMAC", $bits, " keyed state.")]
    /// KMAC is a variable-output MAC/PRF. It intentionally does not implement
    /// [`crate::traits::Mac`], which assumes a fixed-size tag.
    #[derive(Clone)]
    pub struct $name {
      state: $cshake,
      initial_state: $cshake,
    }

    impl fmt::Debug for $name {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(stringify!($name)).finish_non_exhaustive()
      }
    }

    impl $name {
      /// Construct a new KMAC state keyed by `key` and domain-separated by
      /// `customization`.
      #[must_use]
      pub fn new(key: &[u8], customization: &[u8]) -> Self {
        let mut state = $cshake::new(b"KMAC", customization);
        Self::absorb_key(&mut state, key);
        let initial_state = state.clone();
        Self { state, initial_state }
      }

      #[inline]
      fn absorb_key(state: &mut $cshake, key: &[u8]) {
        let (key_prefix, key_prefix_len) = left_encode(crate::bytes_to_bits(key.len()));
        let payload_len = key_prefix_len.strict_add(key.len());
        state.absorb_bytepad_segments(&[&key_prefix[..key_prefix_len], key], payload_len);
      }

      #[inline]
      fn finalize_reader(&self, output_len: usize) -> impl Xof {
        let mut state = self.state.clone();
        let (suffix, suffix_len) = right_encode(crate::bytes_to_bits(output_len));
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

      #[doc = concat!("Compute a one-shot KMAC", $bits, " output into `out`.")]
      #[inline]
      pub fn mac_into(key: &[u8], customization: &[u8], data: &[u8], out: &mut [u8]) {
        let mut state = Self::new(key, customization);
        state.update(data);
        state.finalize_into(out);
      }

      #[doc = concat!("Compute a one-shot KMAC", $bits, " output into a fixed-size array.")]
      pub fn mac_array<const N: usize>(key: &[u8], customization: &[u8], data: &[u8]) -> [u8; N] {
        let mut out = [0u8; N];
        Self::mac_into(key, customization, data, &mut out);
        out
      }

      /// Verify `expected` after traversing its public-length contents.
      ///
      /// This is the one-shot helper. Use [`Self::verify`] for an already-accumulated state.
      /// Generated-code timing claims are configuration- and release-evidence-bound;
      /// see `ct.toml`.
      #[must_use = "MAC verification must be checked; a dropped Result silently accepts a forged tag"]
      pub fn verify_tag(
        key: &[u8],
        customization: &[u8],
        data: &[u8],
        expected: &[u8],
      ) -> Result<(), VerificationError> {
        let mut state = Self::new(key, customization);
        state.update(data);
        state.verify(expected)
      }

      #[doc = concat!("Verify `expected` against the current KMAC", $bits, " output after a full public-length comparison.")]
      /// This checks the MAC for the bytes already absorbed into `self`; it does
      /// not recompute from `(key, customization, data)` like [`Self::verify_tag`].
      /// Generated-code timing claims are configuration- and release-evidence-bound;
      /// see `ct.toml`.
      #[must_use = "MAC verification must be checked; a dropped Result silently accepts a forged tag"]
      pub fn verify(&self, expected: &[u8]) -> Result<(), VerificationError> {
        if expected.is_empty() {
          return Err(VerificationError::new());
        }

        let mut reader = self.finalize_reader(expected.len());
        let mut diff = 0u8;
        let mut block = [0u8; 64];

        for chunk in expected.chunks(block.len()) {
          reader.squeeze(&mut block[..chunk.len()]);
          diff |= (!ct::public_len_eq(&block[..chunk.len()], chunk)).into_u8();
        }

        ct::zeroize(&mut block);
        if core::hint::black_box(diff) == 0 {
          Ok(())
        } else {
          Err(VerificationError::new())
        }
      }
    }
  };
}

define_kmac!(Kmac128, Cshake128, "128");
define_kmac!(Kmac256, Cshake256, "256");

#[cfg(test)]
mod tests {
  use super::{Kmac128, Kmac256};

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

    let mut kmac128 = Kmac128::new(b"key", b"custom");
    kmac128.update(b"abc");
    let expected128 = Kmac128::mac_array::<32>(b"key", b"custom", b"abc");
    kmac128.finalize_into(&mut actual);
    assert_eq!(actual, expected128);
  }
}
