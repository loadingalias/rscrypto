//! PHC string format codec (shared by Argon2 and scrypt).
//!
//! Implements the [PHC string format][phc] `$alg$version$params$salt$hash`
//! with RFC 4648 base64 encoding (standard alphabet, no padding, no line
//! wrapping). The parser is strict: malformed separators, empty segments,
//! out-of-range parameter values, and trailing bytes in base64 are all
//! rejected.
//!
//! This module is used internally by `crate::auth::argon2` and
//! `crate::auth::scrypt` for their `hash_string` / `verify_string`
//! helpers. The only public surface is [`PhcError`], which surfaces parse
//! failures when callers use the typed `decode_*` helpers on the hashers.
//!
//! [phc]: https://github.com/P-H-C/phc-string-format/blob/master/phc-sf-spec.md

#![allow(clippy::indexing_slicing)]
// `decode_base64_to_vec` and `push_u32_decimal` are only reachable when a
// PHC-aware hasher (argon2 or scrypt) is enabled. Without either, the
// helpers are dead code — silence the warning rather than cfg-gate every
// symbol individually.
#![cfg_attr(
  all(feature = "phc-strings", not(any(feature = "argon2", feature = "scrypt"))),
  allow(dead_code)
)]

use alloc::{string::String, vec, vec::Vec};
use core::fmt;

// ─── Base64 (standard alphabet, no padding) ─────────────────────────────────

/// RFC 4648 standard alphabet (`A-Za-z0-9+/`), no padding, no line wrap.
const B64_ENCODE_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Reverse table: byte → 6-bit value (0..=63), or 0xFF for invalid.
const B64_DECODE_TABLE: [u8; 256] = {
  let mut table = [0xFFu8; 256];
  let mut i = 0u8;
  while i < 64 {
    table[B64_ENCODE_TABLE[i as usize] as usize] = i;
    i = i.wrapping_add(1);
  }
  table
};

/// Encode `bytes` as PHC-variant base64 (standard alphabet, no padding).
///
/// Appends to `out` — callers managing multi-segment PHC strings reuse the
/// same `String` buffer without intermediate allocation.
pub(crate) fn base64_encode_into(bytes: &[u8], out: &mut String) {
  let full_triples = bytes.len() / 3;
  let tail = bytes.len() % 3;

  for i in 0..full_triples {
    let off = i.strict_mul(3);
    // SAFETY: off + 3 <= bytes.len() by construction of full_triples.
    let b0 = bytes[off] as u32;
    let b1 = bytes[off.strict_add(1)] as u32;
    let b2 = bytes[off.strict_add(2)] as u32;
    let word = (b0 << 16) | (b1 << 8) | b2;

    out.push(B64_ENCODE_TABLE[((word >> 18) & 0x3F) as usize] as char);
    out.push(B64_ENCODE_TABLE[((word >> 12) & 0x3F) as usize] as char);
    out.push(B64_ENCODE_TABLE[((word >> 6) & 0x3F) as usize] as char);
    out.push(B64_ENCODE_TABLE[(word & 0x3F) as usize] as char);
  }

  let off = full_triples.strict_mul(3);
  match tail {
    1 => {
      let b0 = bytes[off] as u32;
      out.push(B64_ENCODE_TABLE[((b0 >> 2) & 0x3F) as usize] as char);
      out.push(B64_ENCODE_TABLE[((b0 << 4) & 0x3F) as usize] as char);
    }
    2 => {
      let b0 = bytes[off] as u32;
      let b1 = bytes[off.strict_add(1)] as u32;
      let word = (b0 << 8) | b1;
      out.push(B64_ENCODE_TABLE[((word >> 10) & 0x3F) as usize] as char);
      out.push(B64_ENCODE_TABLE[((word >> 4) & 0x3F) as usize] as char);
      out.push(B64_ENCODE_TABLE[((word << 2) & 0x3F) as usize] as char);
    }
    _ => {}
  }
}

/// Maximum number of bytes a base64-encoded string of `len` characters can
/// decode to. Used to size destination buffers.
pub(crate) const fn base64_decoded_len(encoded_len: usize) -> usize {
  // Each 4 chars → 3 bytes; each remaining 2 chars → 1 byte, 3 chars → 2 bytes.
  let full = encoded_len / 4;
  let tail = encoded_len % 4;
  let bytes = full.wrapping_mul(3);
  match tail {
    0 => bytes,
    2 => bytes.wrapping_add(1),
    3 => bytes.wrapping_add(2),
    _ => bytes, // tail == 1 is invalid but we don't fail from const
  }
}

/// Decode `s` (PHC-variant base64) into `out`. Returns the number of bytes
/// actually written.
///
/// Rejects:
/// - any non-alphabet byte,
/// - a `tail == 1` group (impossible output),
/// - trailing bits that are not zero (strict mode — prevents canonicalisation mismatches on
///   round-trip).
pub(crate) fn base64_decode_into(s: &str, out: &mut [u8]) -> Result<usize, PhcError> {
  let bytes = s.as_bytes();
  let full = bytes.len() / 4;
  let tail = bytes.len() % 4;

  if tail == 1 {
    return Err(PhcError::InvalidBase64);
  }

  let expected_out = base64_decoded_len(bytes.len());
  if out.len() < expected_out {
    return Err(PhcError::OutputBufferTooSmall);
  }

  let mut written = 0usize;
  for i in 0..full {
    let off = i.strict_mul(4);
    let d0 = B64_DECODE_TABLE[bytes[off] as usize];
    let d1 = B64_DECODE_TABLE[bytes[off.strict_add(1)] as usize];
    let d2 = B64_DECODE_TABLE[bytes[off.strict_add(2)] as usize];
    let d3 = B64_DECODE_TABLE[bytes[off.strict_add(3)] as usize];
    if (d0 | d1 | d2 | d3) == 0xFF {
      return Err(PhcError::InvalidBase64);
    }
    let word = ((d0 as u32) << 18) | ((d1 as u32) << 12) | ((d2 as u32) << 6) | (d3 as u32);
    out[written] = (word >> 16) as u8;
    out[written.strict_add(1)] = (word >> 8) as u8;
    out[written.strict_add(2)] = word as u8;
    written = written.strict_add(3);
  }

  let off = full.strict_mul(4);
  match tail {
    0 => {}
    2 => {
      let d0 = B64_DECODE_TABLE[bytes[off] as usize];
      let d1 = B64_DECODE_TABLE[bytes[off.strict_add(1)] as usize];
      if (d0 | d1) == 0xFF {
        return Err(PhcError::InvalidBase64);
      }
      // Trailing 4 bits of d1 must be zero (strict mode).
      if (d1 & 0x0F) != 0 {
        return Err(PhcError::InvalidBase64);
      }
      out[written] = (d0 << 2) | (d1 >> 4);
      written = written.strict_add(1);
    }
    3 => {
      let d0 = B64_DECODE_TABLE[bytes[off] as usize];
      let d1 = B64_DECODE_TABLE[bytes[off.strict_add(1)] as usize];
      let d2 = B64_DECODE_TABLE[bytes[off.strict_add(2)] as usize];
      if (d0 | d1 | d2) == 0xFF {
        return Err(PhcError::InvalidBase64);
      }
      // Trailing 2 bits of d2 must be zero (strict mode).
      if (d2 & 0x03) != 0 {
        return Err(PhcError::InvalidBase64);
      }
      let word = ((d0 as u32) << 10) | ((d1 as u32) << 4) | ((d2 as u32) >> 2);
      out[written] = (word >> 8) as u8;
      out[written.strict_add(1)] = word as u8;
      written = written.strict_add(2);
    }
    _ => return Err(PhcError::InvalidBase64),
  }

  Ok(written)
}

/// Decode a PHC base64 segment into a freshly-allocated `Vec<u8>`.
///
/// Shared between the Argon2 and scrypt PHC integrations — both call this
/// from `decode_string` to materialise the salt and hash bytes.
pub(crate) fn decode_base64_to_vec(encoded: &str) -> Result<Vec<u8>, PhcError> {
  let cap = base64_decoded_len(encoded.len());
  let mut buf = vec![0u8; cap];
  let n = base64_decode_into(encoded, &mut buf)?;
  buf.truncate(n);
  Ok(buf)
}

/// Append `n` as base-10 decimal (no leading zero) to `out`.
///
/// Shared decimal writer used by Argon2 and scrypt PHC encoders for cost
/// parameters. Produces exactly the canonical form `PhcParamIter` +
/// `parse_param_u32` accept on round-trip.
pub(crate) fn push_u32_decimal(out: &mut String, n: u32) {
  if n == 0 {
    out.push('0');
    return;
  }
  // Reverse-decimal into a small stack buffer, then flip on emit. `u32::MAX`
  // fits in 10 decimal digits.
  let mut digits = [0u8; 10];
  let mut len = 0usize;
  let mut v = n;
  while v > 0 {
    digits[len] = b'0' + (v % 10) as u8;
    v /= 10;
    len = len.strict_add(1);
  }
  for i in (0..len).rev() {
    out.push(digits[i] as char);
  }
}

// ─── Parameter scanner (k=v,k=v,...) ────────────────────────────────────────

/// Iterator over comma-separated `key=value` pairs.
///
/// Both key and value are borrowed substrings of the original input. Empty
/// keys, empty values, missing `=`, and empty pair segments are reported as
/// `PhcError::MalformedParams`.
pub(crate) struct PhcParamIter<'a> {
  rest: &'a str,
  done: bool,
}

impl<'a> PhcParamIter<'a> {
  pub(crate) fn new(params: &'a str) -> Self {
    Self {
      rest: params,
      done: params.is_empty(),
    }
  }
}

impl<'a> Iterator for PhcParamIter<'a> {
  type Item = Result<(&'a str, &'a str), PhcError>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.done {
      return None;
    }
    let (pair, advance) = match self.rest.find(',') {
      Some(idx) => {
        // SAFETY: idx is a valid char boundary (',' is ASCII).
        let pair = &self.rest[..idx];
        self.rest = &self.rest[idx.strict_add(1)..];
        (pair, false)
      }
      None => {
        let pair = self.rest;
        self.rest = "";
        (pair, true)
      }
    };
    if advance {
      self.done = true;
    }

    if pair.is_empty() {
      return Some(Err(PhcError::MalformedParams));
    }
    let eq = match pair.find('=') {
      Some(i) => i,
      None => return Some(Err(PhcError::MalformedParams)),
    };
    let key = &pair[..eq];
    let value = &pair[eq.strict_add(1)..];
    if key.is_empty() || value.is_empty() {
      return Some(Err(PhcError::MalformedParams));
    }
    Some(Ok((key, value)))
  }
}

/// Parse a numeric parameter value as `u32` (decimal, no leading zeros
/// except for the literal `0`).
pub(crate) fn parse_param_u32(value: &str) -> Result<u32, PhcError> {
  if value.is_empty() {
    return Err(PhcError::MalformedParams);
  }
  // Reject leading zeros (e.g. "01") and leading sign (e.g. "-1", "+1").
  let bytes = value.as_bytes();
  if bytes.len() > 1 && bytes[0] == b'0' {
    return Err(PhcError::MalformedParams);
  }
  let mut acc: u64 = 0;
  for &b in bytes {
    if !b.is_ascii_digit() {
      return Err(PhcError::MalformedParams);
    }
    acc = acc.strict_mul(10).strict_add((b - b'0') as u64);
    if acc > u32::MAX as u64 {
      return Err(PhcError::ParamOutOfRange);
    }
  }
  Ok(u32::try_from(acc).unwrap_or_else(|_| unreachable!("acc <= u32::MAX, enforced inside the loop")))
}

// ─── Segmented PHC parser ───────────────────────────────────────────────────

/// Parsed `$alg$[v=...$]params$salt$hash` components.
///
/// The `version` slot is optional: PHC encoders may omit the version segment
/// entirely (common for scrypt) or include it (mandatory for Argon2 per
/// RFC 9106 §3.1 recommendations).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PhcParts<'a> {
  pub algorithm: &'a str,
  pub version: Option<&'a str>,
  pub parameters: &'a str,
  pub salt_b64: &'a str,
  pub hash_b64: &'a str,
}

/// Maximum accepted PHC string length, chosen to leave ample room for
/// reasonable Argon2/scrypt encodings while bounding parse work on
/// adversarial inputs.
const MAX_PHC_LEN: usize = 1024;

/// Parse a PHC string into its five components.
///
/// Requires a leading `$`, an algorithm segment, an optional `v=<number>`
/// segment, a parameters segment, a salt segment, and a hash segment.
/// Rejects:
/// - total length > [`MAX_PHC_LEN`],
/// - missing leading `$`,
/// - empty segments,
/// - trailing `$` or trailing bytes.
pub(crate) fn parse(encoded: &str) -> Result<PhcParts<'_>, PhcError> {
  if encoded.len() > MAX_PHC_LEN {
    return Err(PhcError::InputTooLong);
  }
  let rest = encoded.strip_prefix('$').ok_or(PhcError::MalformedInput)?;

  let mut segments = rest.split('$');
  let algorithm = segments.next().ok_or(PhcError::MalformedInput)?;
  if algorithm.is_empty() {
    return Err(PhcError::EmptySegment);
  }

  let second = segments.next().ok_or(PhcError::MalformedInput)?;
  if second.is_empty() {
    return Err(PhcError::EmptySegment);
  }

  let (version, parameters) = if let Some(v) = second.strip_prefix("v=") {
    if v.is_empty() {
      return Err(PhcError::InvalidVersion);
    }
    let params = segments.next().ok_or(PhcError::MalformedInput)?;
    if params.is_empty() {
      return Err(PhcError::EmptySegment);
    }
    (Some(v), params)
  } else {
    (None, second)
  };

  let salt_b64 = segments.next().ok_or(PhcError::MalformedInput)?;
  if salt_b64.is_empty() {
    return Err(PhcError::EmptySegment);
  }

  let hash_b64 = segments.next().ok_or(PhcError::MalformedInput)?;
  if hash_b64.is_empty() {
    return Err(PhcError::EmptySegment);
  }

  if segments.next().is_some() {
    return Err(PhcError::MalformedInput);
  }

  Ok(PhcParts {
    algorithm,
    version,
    parameters,
    salt_b64,
    hash_b64,
  })
}

// ─── Error type ─────────────────────────────────────────────────────────────

/// Parse or decode error for PHC-format strings.
///
/// Surfaced by the explicit `decode_*` helpers on Argon2/scrypt hashers.
/// The `verify_string` flow collapses these into
/// [`crate::VerificationError`] to avoid leaking whether a failure was a
/// parse error vs. a wrong password.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PhcError {
  /// Input is longer than the implementation accepts.
  InputTooLong,
  /// Missing leading `$`, missing segment, or extra segments.
  MalformedInput,
  /// A mandatory segment was empty.
  EmptySegment,
  /// The algorithm identifier did not match the expected value.
  AlgorithmMismatch,
  /// The `v=<number>` segment was empty or malformed.
  InvalidVersion,
  /// The encoded PHC is for a version the decoder does not support.
  UnsupportedVersion,
  /// A parameter pair was empty, missing `=`, or had an empty key/value.
  MalformedParams,
  /// A parameter appeared more than once.
  DuplicateParam,
  /// A required parameter was missing.
  MissingParam,
  /// An unrecognised parameter key was present.
  UnknownParam,
  /// A parameter value did not fit the target type or violated algorithm
  /// constraints (e.g. Argon2 `m < 8·p`).
  ParamOutOfRange,
  /// Base64 payload contained an invalid character, had a tail of length 1,
  /// or had non-zero trailing bits (strict canonicalisation).
  InvalidBase64,
  /// The decoded salt or hash did not satisfy the algorithm's length
  /// requirements (e.g. Argon2 salt < 8 bytes).
  InvalidLength,
  /// Supplied scratch buffer was too small for the decoded payload.
  OutputBufferTooSmall,
}

impl fmt::Display for PhcError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let msg = match self {
      Self::InputTooLong => "PHC string exceeds maximum length",
      Self::MalformedInput => "PHC string is malformed",
      Self::EmptySegment => "PHC string contains an empty segment",
      Self::AlgorithmMismatch => "PHC algorithm does not match expected value",
      Self::InvalidVersion => "PHC version segment is malformed",
      Self::UnsupportedVersion => "PHC version is not supported",
      Self::MalformedParams => "PHC parameters segment is malformed",
      Self::DuplicateParam => "PHC parameter appears more than once",
      Self::MissingParam => "PHC parameter segment is missing a required key",
      Self::UnknownParam => "PHC parameter segment contains an unknown key",
      Self::ParamOutOfRange => "PHC parameter value is out of range",
      Self::InvalidBase64 => "PHC base64 payload is invalid",
      Self::InvalidLength => "PHC decoded payload has invalid length",
      Self::OutputBufferTooSmall => "PHC decode buffer is too small",
    };
    f.write_str(msg)
  }
}

impl core::error::Error for PhcError {}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use alloc::{string::String, vec, vec::Vec};

  use super::*;

  // ── Base64 ────────────────────────────────────────────────────────────

  #[test]
  fn base64_roundtrip_all_lengths_0_to_64() {
    for len in 0..=64 {
      let input: Vec<u8> = (0..len).map(|i| ((i * 31 + 7) & 0xff) as u8).collect();
      let mut encoded = String::new();
      base64_encode_into(&input, &mut encoded);

      let mut decoded = vec![0u8; base64_decoded_len(encoded.len())];
      let n = base64_decode_into(&encoded, &mut decoded).unwrap();
      decoded.truncate(n);
      assert_eq!(decoded, input, "roundtrip failed at len={len}");
    }
  }

  #[test]
  fn base64_no_padding_emitted() {
    let mut s = String::new();
    base64_encode_into(b"A", &mut s);
    assert_eq!(s, "QQ"); // 1 byte → 2 chars, no padding
    s.clear();
    base64_encode_into(b"AB", &mut s);
    assert_eq!(s, "QUI"); // 2 bytes → 3 chars
    s.clear();
    base64_encode_into(b"ABC", &mut s);
    assert_eq!(s, "QUJD"); // 3 bytes → 4 chars
  }

  #[test]
  fn base64_rejects_invalid_char() {
    let mut out = [0u8; 32];
    assert_eq!(base64_decode_into("AAA!", &mut out), Err(PhcError::InvalidBase64));
    assert_eq!(base64_decode_into("AA=A", &mut out), Err(PhcError::InvalidBase64)); // '=' is not in the no-pad alphabet
    assert_eq!(base64_decode_into("A A A", &mut out), Err(PhcError::InvalidBase64));
  }

  #[test]
  fn base64_rejects_tail_of_one() {
    let mut out = [0u8; 32];
    assert_eq!(base64_decode_into("A", &mut out), Err(PhcError::InvalidBase64));
    assert_eq!(base64_decode_into("AAAAA", &mut out), Err(PhcError::InvalidBase64));
  }

  #[test]
  fn base64_rejects_non_canonical_trailing_bits() {
    // `AB` decodes to 1 byte; last base64 char's low 4 bits must be zero.
    // 'A'=0, 'B'=1. 'B' has low nibble = 0001 → non-canonical.
    let mut out = [0u8; 4];
    assert_eq!(base64_decode_into("AB", &mut out), Err(PhcError::InvalidBase64));
    // Similar for 3-char tail: last char's low 2 bits must be zero.
    // 'AAB' → last char 'B' has low 2 bits = 01 → non-canonical.
    assert_eq!(base64_decode_into("AAB", &mut out), Err(PhcError::InvalidBase64));
  }

  #[test]
  fn base64_rejects_output_too_small() {
    let mut out = [0u8; 1];
    // "QUJD" decodes to 3 bytes; buffer is only 1 byte.
    assert_eq!(
      base64_decode_into("QUJD", &mut out),
      Err(PhcError::OutputBufferTooSmall)
    );
  }

  // ── Param scanner ─────────────────────────────────────────────────────

  #[test]
  fn param_iter_single_pair() {
    let mut it = PhcParamIter::new("m=65536");
    assert_eq!(it.next().unwrap().unwrap(), ("m", "65536"));
    assert!(it.next().is_none());
  }

  #[test]
  fn param_iter_multiple_pairs() {
    let mut it = PhcParamIter::new("m=65536,t=3,p=4");
    assert_eq!(it.next().unwrap().unwrap(), ("m", "65536"));
    assert_eq!(it.next().unwrap().unwrap(), ("t", "3"));
    assert_eq!(it.next().unwrap().unwrap(), ("p", "4"));
    assert!(it.next().is_none());
  }

  #[test]
  fn param_iter_empty_input() {
    let mut it = PhcParamIter::new("");
    assert!(it.next().is_none());
  }

  #[test]
  fn param_iter_rejects_missing_equals() {
    let mut it = PhcParamIter::new("mX65536");
    assert_eq!(it.next().unwrap(), Err(PhcError::MalformedParams));
  }

  #[test]
  fn param_iter_rejects_empty_pair_segment() {
    let mut it = PhcParamIter::new("m=1,,p=2");
    assert_eq!(it.next().unwrap().unwrap(), ("m", "1"));
    assert_eq!(it.next().unwrap(), Err(PhcError::MalformedParams));
  }

  #[test]
  fn param_iter_rejects_empty_key() {
    let mut it = PhcParamIter::new("=65536");
    assert_eq!(it.next().unwrap(), Err(PhcError::MalformedParams));
  }

  #[test]
  fn param_iter_rejects_empty_value() {
    let mut it = PhcParamIter::new("m=");
    assert_eq!(it.next().unwrap(), Err(PhcError::MalformedParams));
  }

  #[test]
  fn parse_param_u32_accepts_valid() {
    assert_eq!(parse_param_u32("0").unwrap(), 0);
    assert_eq!(parse_param_u32("1").unwrap(), 1);
    assert_eq!(parse_param_u32("65536").unwrap(), 65_536);
    assert_eq!(parse_param_u32("4294967295").unwrap(), u32::MAX);
  }

  #[test]
  fn parse_param_u32_rejects_bad() {
    assert_eq!(parse_param_u32(""), Err(PhcError::MalformedParams));
    assert_eq!(parse_param_u32("01"), Err(PhcError::MalformedParams)); // leading zero
    assert_eq!(parse_param_u32("-1"), Err(PhcError::MalformedParams));
    assert_eq!(parse_param_u32("+1"), Err(PhcError::MalformedParams));
    assert_eq!(parse_param_u32("1 "), Err(PhcError::MalformedParams));
    assert_eq!(parse_param_u32("abc"), Err(PhcError::MalformedParams));
    assert_eq!(parse_param_u32("4294967296"), Err(PhcError::ParamOutOfRange));
  }

  // ── Segmented parser ──────────────────────────────────────────────────

  #[test]
  fn parse_argon2id_canonical() {
    let encoded = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$c29tZWhhc2g";
    let parts = parse(encoded).unwrap();
    assert_eq!(parts.algorithm, "argon2id");
    assert_eq!(parts.version, Some("19"));
    assert_eq!(parts.parameters, "m=65536,t=3,p=4");
    assert_eq!(parts.salt_b64, "c29tZXNhbHQ");
    assert_eq!(parts.hash_b64, "c29tZWhhc2g");
  }

  #[test]
  fn parse_scrypt_no_version() {
    let encoded = "$scrypt$ln=14,r=8,p=1$c29tZXNhbHQ$c29tZWhhc2g";
    let parts = parse(encoded).unwrap();
    assert_eq!(parts.algorithm, "scrypt");
    assert_eq!(parts.version, None);
    assert_eq!(parts.parameters, "ln=14,r=8,p=1");
  }

  #[test]
  fn parse_rejects_missing_leading_dollar() {
    assert_eq!(
      parse("argon2id$v=19$m=1,t=1,p=1$c29tZQ$c29tZQ"),
      Err(PhcError::MalformedInput)
    );
  }

  #[test]
  fn parse_rejects_empty_segment() {
    // Missing algorithm: "$$v=19$..." — two dollars in a row.
    assert_eq!(parse("$$v=19$m=1,t=1,p=1$c29tZQ$c29tZQ"), Err(PhcError::EmptySegment));
    // Missing salt.
    assert_eq!(parse("$argon2id$v=19$m=1,t=1,p=1$$c29tZQ"), Err(PhcError::EmptySegment));
    // Missing hash.
    assert_eq!(parse("$argon2id$v=19$m=1,t=1,p=1$c29tZQ$"), Err(PhcError::EmptySegment));
  }

  #[test]
  fn parse_rejects_trailing_garbage() {
    assert_eq!(
      parse("$argon2id$v=19$m=1,t=1,p=1$c29tZQ$c29tZQ$extra"),
      Err(PhcError::MalformedInput)
    );
  }

  #[test]
  fn parse_rejects_too_long_input() {
    let mut s = String::from("$argon2id$v=19$");
    while s.len() <= MAX_PHC_LEN {
      s.push('A');
    }
    assert_eq!(parse(&s), Err(PhcError::InputTooLong));
  }

  #[test]
  fn parse_rejects_empty_version_value() {
    assert_eq!(
      parse("$argon2id$v=$m=1,t=1,p=1$c29tZQ$c29tZQ"),
      Err(PhcError::InvalidVersion)
    );
  }

  #[test]
  fn parse_without_version_segment_returns_none() {
    let parts = parse("$argon2id$m=1,t=1,p=1$c29tZQ$c29tZQ").unwrap();
    assert_eq!(parts.version, None);
  }

  #[test]
  fn parse_truncated_segments() {
    // only algorithm.
    assert_eq!(parse("$argon2id"), Err(PhcError::MalformedInput));
    // alg + params, missing salt & hash.
    assert_eq!(parse("$argon2id$m=1,t=1,p=1"), Err(PhcError::MalformedInput));
    // alg + params + salt, missing hash.
    assert_eq!(parse("$argon2id$m=1,t=1,p=1$c29tZQ"), Err(PhcError::MalformedInput));
  }

  // ── Error trait plumbing ──────────────────────────────────────────────

  #[test]
  fn error_is_copy_and_implements_error_trait() {
    fn assert_copy<T: Copy>() {}
    fn assert_err<T: core::error::Error>() {}
    assert_copy::<PhcError>();
    assert_err::<PhcError>();
  }

  #[test]
  fn error_display_is_non_empty_for_every_variant() {
    let all = [
      PhcError::InputTooLong,
      PhcError::MalformedInput,
      PhcError::EmptySegment,
      PhcError::AlgorithmMismatch,
      PhcError::InvalidVersion,
      PhcError::UnsupportedVersion,
      PhcError::MalformedParams,
      PhcError::DuplicateParam,
      PhcError::MissingParam,
      PhcError::UnknownParam,
      PhcError::ParamOutOfRange,
      PhcError::InvalidBase64,
      PhcError::InvalidLength,
      PhcError::OutputBufferTooSmall,
    ];
    for e in all {
      let s = alloc::format!("{e}");
      assert!(!s.is_empty());
    }
  }
}
