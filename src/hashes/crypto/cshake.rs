//! cSHAKE128 and cSHAKE256 (SP 800-185).

use super::{
  keccak::{KeccakCore, KeccakXof},
  sp800185::{RATE_128, RATE_256, cshake_core},
};
use crate::traits::Xof;

macro_rules! define_cshake {
  ($name:ident, $reader:ident, $rate:expr, $bits:literal) => {
    #[doc = concat!("cSHAKE", $bits, " state with explicit function-name and customization strings.")]
    /// Standardized in NIST SP 800-185.
    #[derive(Clone)]
    pub struct $name {
      core: KeccakCore<$rate>,
      initial_state: KeccakCore<$rate>,
      pad: u8,
    }

    impl core::fmt::Debug for $name {
      fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct(stringify!($name)).finish_non_exhaustive()
      }
    }

    impl $name {
      /// Construct a cSHAKE state from `function_name` and `customization`.
      #[inline]
      #[must_use]
      pub fn new(function_name: &[u8], customization: &[u8]) -> Self {
        let (core, pad) = cshake_core::<$rate>(function_name, customization);
        let initial_state = core.clone();
        Self {
          core,
          initial_state,
          pad,
        }
      }

      /// Compute a one-shot cSHAKE XOF reader.
      #[inline]
      #[must_use]
      pub fn xof(function_name: &[u8], customization: &[u8], data: &[u8]) -> $reader {
        let (mut core, pad) = cshake_core::<$rate>(function_name, customization);
        core.update(data);
        $reader {
          inner: core.finalize_xof(pad),
        }
      }

      /// Absorb additional message bytes.
      #[inline]
      pub fn update(&mut self, data: &[u8]) {
        self.core.update(data);
      }

      /// Finalize into an extendable-output reader.
      #[inline]
      #[must_use]
      pub fn finalize_xof(&self) -> $reader {
        $reader {
          inner: self.core.finalize_xof(self.pad),
        }
      }

      /// Reset back to the initial cSHAKE prefix state.
      #[inline]
      pub fn reset(&mut self) {
        self.core = self.initial_state.clone();
      }

      /// Fill `out` in one shot.
      #[inline]
      pub fn hash_into(function_name: &[u8], customization: &[u8], data: &[u8], out: &mut [u8]) {
        let (mut core, pad) = cshake_core::<$rate>(function_name, customization);
        core.update(data);
        core.finalize_xof_into(pad, out);
      }
    }

    #[doc = concat!("cSHAKE", $bits, " output reader.")]
    #[derive(Clone)]
    pub struct $reader {
      inner: KeccakXof<$rate>,
    }

    impl core::fmt::Debug for $reader {
      fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct(stringify!($reader)).finish_non_exhaustive()
      }
    }

    impl Xof for $reader {
      #[inline]
      fn squeeze(&mut self, out: &mut [u8]) {
        self.inner.squeeze_into(out);
      }
    }

    impl_xof_read!($reader);
  };
}

define_cshake!(Cshake128, Cshake128XofReader, RATE_128, "128");
define_cshake!(Cshake256, Cshake256XofReader, RATE_256, "256");

#[cfg(test)]
mod tests {
  use alloc::vec;

  use super::{Cshake128, Cshake256};
  use crate::traits::Xof;

  fn squeeze_hex(mut reader: impl Xof, len: usize) -> alloc::string::String {
    use alloc::string::String;
    use core::fmt::Write;

    let mut out = vec![0u8; len];
    reader.squeeze(&mut out);

    let mut hex = String::new();
    for byte in out {
      write!(&mut hex, "{byte:02x}").expect("writing hexadecimal to String must succeed");
    }
    hex
  }

  #[test]
  fn empty_function_and_customization_matches_shake128() {
    let ours = squeeze_hex(Cshake128::xof(b"", b"", b"abc"), 64);
    let expected = squeeze_hex(crate::hashes::crypto::Shake128::xof(b"abc"), 64);
    assert_eq!(ours, expected);
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
