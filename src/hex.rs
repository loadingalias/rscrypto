#![allow(clippy::indexing_slicing)]

//! Internal hex encoding, decoding, and formatting utilities.
//!
//! Provides zero-allocation, `no_std`-compatible hex formatting through
//! `core::fmt::Write`. Secret key types use [`DisplaySecret`] for explicit
//! opt-in hex display.

use core::fmt;

/// Hex decoding error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum InvalidHexError {
  /// Input length is not exactly twice the expected byte count.
  InvalidLength,
  /// A non-hex character was encountered.
  InvalidChar {
    /// The offending byte value.
    byte: u8,
    /// Zero-based position in the hex string.
    index: usize,
  },
}

impl fmt::Display for InvalidHexError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::InvalidLength => f.write_str("invalid hex length"),
      Self::InvalidChar { byte, index } => {
        write!(f, "invalid hex character 0x{byte:02x} at index {index}")
      }
    }
  }
}

impl core::error::Error for InvalidHexError {}

/// Decode a single hex character to its 4-bit value.
#[inline]
const fn decode_nibble(byte: u8) -> Option<u8> {
  match byte {
    b'0'..=b'9' => Some(byte - b'0'),
    b'a'..=b'f' => Some(byte - b'a' + 10),
    b'A'..=b'F' => Some(byte - b'A' + 10),
    _ => None,
  }
}

/// Decode a hex string into `out`. Accepts mixed case, no `0x` prefix.
///
/// Returns `InvalidHexError::InvalidLength` when `hex.len() != out.len() * 2`.
pub fn from_hex(hex: &str, out: &mut [u8]) -> Result<(), InvalidHexError> {
  let hex = hex.as_bytes();
  if hex.len() != out.len().strict_mul(2) {
    return Err(InvalidHexError::InvalidLength);
  }
  let mut i = 0;
  while i < out.len() {
    let hi_idx = i.strict_mul(2);
    let lo_idx = hi_idx.strict_add(1);
    let hi = match decode_nibble(hex[hi_idx]) {
      Some(v) => v,
      None => {
        return Err(InvalidHexError::InvalidChar {
          byte: hex[hi_idx],
          index: hi_idx,
        });
      }
    };
    let lo = match decode_nibble(hex[lo_idx]) {
      Some(v) => v,
      None => {
        return Err(InvalidHexError::InvalidChar {
          byte: hex[lo_idx],
          index: lo_idx,
        });
      }
    };
    out[i] = (hi << 4) | lo;
    i = i.strict_add(1);
  }
  Ok(())
}

/// Write each byte as two lowercase hex characters.
pub fn fmt_hex_lower(bytes: &[u8], f: &mut fmt::Formatter<'_>) -> fmt::Result {
  for &b in bytes {
    write!(f, "{b:02x}")?;
  }
  Ok(())
}

/// Write each byte as two uppercase hex characters.
pub fn fmt_hex_upper(bytes: &[u8], f: &mut fmt::Formatter<'_>) -> fmt::Result {
  for &b in bytes {
    write!(f, "{b:02X}")?;
  }
  Ok(())
}

/// Explicit opt-in wrapper for displaying secret key bytes as hex.
///
/// Returned by the `display_secret()` method on secret key types. Implements
/// [`Display`](fmt::Display) so you can `format!("{}", key.display_secret())`.
pub struct DisplaySecret<'a>(pub(crate) &'a [u8]);

impl fmt::Display for DisplaySecret<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    fmt_hex_lower(self.0, f)
  }
}

impl fmt::Debug for DisplaySecret<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "DisplaySecret(\"")?;
    fmt_hex_lower(self.0, f)?;
    write!(f, "\")")
  }
}

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------

/// Implement `LowerHex`, `UpperHex`, `Display`, `Debug`, and `FromStr` for
/// a public byte-array newtype that has `as_bytes()`, `from_bytes()`, and
/// `LENGTH`.
macro_rules! impl_hex_fmt {
  ($type:ty) => {
    impl core::fmt::LowerHex for $type {
      fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::hex::fmt_hex_lower(self.as_bytes(), f)
      }
    }

    impl core::fmt::UpperHex for $type {
      fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::hex::fmt_hex_upper(self.as_bytes(), f)
      }
    }

    impl core::fmt::Display for $type {
      fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::LowerHex::fmt(self, f)
      }
    }

    impl core::str::FromStr for $type {
      type Err = crate::hex::InvalidHexError;

      fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut buf = [0u8; Self::LENGTH];
        crate::hex::from_hex(s, &mut buf)?;
        Ok(Self::from_bytes(buf))
      }
    }
  };
}

/// Implement masked `Debug`, `FromStr`, and `display_secret()` for a secret
/// key newtype. Does **not** implement `Display`, `LowerHex`, or `UpperHex`
/// to prevent accidental logging of key material.
macro_rules! impl_hex_fmt_secret {
  ($type:ty) => {
    impl core::str::FromStr for $type {
      type Err = crate::hex::InvalidHexError;

      fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut buf = [0u8; Self::LENGTH];
        crate::hex::from_hex(s, &mut buf)?;
        Ok(Self::from_bytes(buf))
      }
    }

    impl $type {
      /// Returns a wrapper that displays the key bytes as lowercase hex.
      ///
      /// This is an explicit opt-in for showing secret key material.
      /// The wrapper implements [`Display`](core::fmt::Display).
      #[must_use]
      pub fn display_secret(&self) -> crate::hex::DisplaySecret<'_> {
        crate::hex::DisplaySecret(self.as_bytes())
      }
    }
  };
}

/// Implement `serde::Serialize` and `serde::Deserialize` for a byte-array
/// newtype with `as_bytes() -> &[u8; N]`, `from_bytes([u8; N]) -> Self`,
/// and `LENGTH`.
#[cfg(feature = "serde")]
macro_rules! impl_serde_bytes {
  ($type:ty) => {
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    impl serde::Serialize for $type {
      fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(self.as_bytes())
      }
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    impl<'de> serde::Deserialize<'de> for $type {
      fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ByteVisitor;

        impl<'de> serde::de::Visitor<'de> for ByteVisitor {
          type Value = $type;

          fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "{} bytes", <$type>::LENGTH)
          }

          fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
            let arr: [u8; <$type>::LENGTH] = v.try_into().map_err(|_| E::invalid_length(v.len(), &self))?;
            Ok(<$type>::from_bytes(arr))
          }

          fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut arr = [0u8; <$type>::LENGTH];
            for (i, byte) in arr.iter_mut().enumerate() {
              *byte = seq
                .next_element()?
                .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
            }
            Ok(<$type>::from_bytes(arr))
          }
        }

        deserializer.deserialize_bytes(ByteVisitor)
      }
    }
  };
}

// No-op when serde feature is disabled.
#[cfg(not(feature = "serde"))]
macro_rules! impl_serde_bytes {
  ($type:ty) => {};
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn hex_lower_round_trip() {
    let bytes = [0xde, 0xad, 0xbe, 0xef];
    let hex = alloc::format!("{}", DisplaySecret(&bytes));
    assert_eq!(hex, "deadbeef");

    let mut out = [0u8; 4];
    from_hex(&hex, &mut out).unwrap();
    assert_eq!(out, bytes);
  }

  #[test]
  fn hex_upper() {
    struct W([u8; 2]);
    impl fmt::Display for W {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_hex_upper(&self.0, f)
      }
    }
    assert_eq!(alloc::format!("{}", W([0xab, 0xcd])), "ABCD");
  }

  #[test]
  fn from_hex_mixed_case() {
    let mut out = [0u8; 3];
    from_hex("aAbBcC", &mut out).unwrap();
    assert_eq!(out, [0xaa, 0xbb, 0xcc]);
  }

  #[test]
  fn from_hex_invalid_length() {
    let mut out = [0u8; 2];
    assert_eq!(from_hex("abc", &mut out), Err(InvalidHexError::InvalidLength));
  }

  #[test]
  fn from_hex_invalid_char() {
    let mut out = [0u8; 2];
    let err = from_hex("abzz", &mut out).unwrap_err();
    assert_eq!(err, InvalidHexError::InvalidChar { byte: b'z', index: 2 });
  }

  #[test]
  fn display_secret_debug() {
    let bytes = [0x42; 4];
    let d = DisplaySecret(&bytes);
    assert_eq!(alloc::format!("{d:?}"), "DisplaySecret(\"42424242\")");
  }

  #[test]
  fn error_display() {
    assert_eq!(
      alloc::format!("{}", InvalidHexError::InvalidLength),
      "invalid hex length"
    );
    assert_eq!(
      alloc::format!("{}", InvalidHexError::InvalidChar { byte: b'z', index: 5 }),
      "invalid hex character 0x7a at index 5"
    );
  }
}
