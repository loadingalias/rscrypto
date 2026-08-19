//! PHC string format codec (shared by Argon2 and scrypt).
//!
//! Implements the [PHC string format][phc] `$alg$version$params$salt$hash`
//! with RFC 4648 base64 encoding (standard alphabet, no padding, no line
//! wrapping). The parser is strict: malformed separators, empty segments,
//! out-of-range parameter values, and trailing bytes in base64 are all
//! rejected.
//!
//! This module is crate-private. Algorithm modules own format-specific
//! validation and expose only bounded password operations.
//!
//! [phc]: https://github.com/P-H-C/phc-string-format/blob/master/phc-sf-spec.md

use alloc::string::String;
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
  let (triples, tail) = bytes.as_chunks::<3>();
  for &[b0, b1, b2] in triples {
    out.push(char::from(B64_ENCODE_TABLE[usize::from(b0 >> 2)]));
    out.push(char::from(
      B64_ENCODE_TABLE[usize::from(((b0 & 0x03) << 4) | (b1 >> 4))],
    ));
    out.push(char::from(
      B64_ENCODE_TABLE[usize::from(((b1 & 0x0F) << 2) | (b2 >> 6))],
    ));
    out.push(char::from(B64_ENCODE_TABLE[usize::from(b2 & 0x3F)]));
  }

  if let &[b0, b1] = tail {
    out.push(char::from(B64_ENCODE_TABLE[usize::from(b0 >> 2)]));
    out.push(char::from(
      B64_ENCODE_TABLE[usize::from(((b0 & 0x03) << 4) | (b1 >> 4))],
    ));
    out.push(char::from(B64_ENCODE_TABLE[usize::from((b1 & 0x0F) << 2)]));
  } else if let &[b0] = tail {
    out.push(char::from(B64_ENCODE_TABLE[usize::from(b0 >> 2)]));
    out.push(char::from(B64_ENCODE_TABLE[usize::from((b0 & 0x03) << 4)]));
  }
}

/// Maximum number of bytes a base64-encoded string of `len` characters can
/// decode to. Used to size destination buffers.
pub(crate) const fn base64_decoded_len(encoded_len: usize) -> usize {
  // Each 4 chars → 3 bytes; each remaining 2 chars → 1 byte, 3 chars → 2 bytes.
  let full = encoded_len / 4;
  let tail = encoded_len % 4;
  let bytes = full.strict_mul(3);
  match tail {
    0 => bytes,
    2 => bytes.strict_add(1),
    3 => bytes.strict_add(2),
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
  let (groups, tail) = bytes.as_chunks::<4>();
  if tail.len() == 1 {
    return Err(PhcError::InvalidBase64);
  }
  let expected_out = base64_decoded_len(bytes.len());
  let destination = out.get_mut(..expected_out).ok_or(PhcError::OutputBufferTooSmall)?;
  let (output_groups, output_tail) = destination.as_chunks_mut::<3>();

  for (&[b0, b1, b2, b3], output) in groups.iter().zip(output_groups) {
    let d0 = B64_DECODE_TABLE[usize::from(b0)];
    let d1 = B64_DECODE_TABLE[usize::from(b1)];
    let d2 = B64_DECODE_TABLE[usize::from(b2)];
    let d3 = B64_DECODE_TABLE[usize::from(b3)];
    if (d0 | d1 | d2 | d3) == 0xFF {
      return Err(PhcError::InvalidBase64);
    }
    *output = [(d0 << 2) | (d1 >> 4), (d1 << 4) | (d2 >> 2), (d2 << 6) | d3];
  }

  match (tail, output_tail) {
    ([], []) => {}
    (&[b0, b1], [output]) => {
      let d0 = B64_DECODE_TABLE[usize::from(b0)];
      let d1 = B64_DECODE_TABLE[usize::from(b1)];
      if (d0 | d1) == 0xFF {
        return Err(PhcError::InvalidBase64);
      }
      // Trailing 4 bits of d1 must be zero (strict mode).
      if (d1 & 0x0F) != 0 {
        return Err(PhcError::InvalidBase64);
      }
      *output = (d0 << 2) | (d1 >> 4);
    }
    (&[b0, b1, b2], [output0, output1]) => {
      let d0 = B64_DECODE_TABLE[usize::from(b0)];
      let d1 = B64_DECODE_TABLE[usize::from(b1)];
      let d2 = B64_DECODE_TABLE[usize::from(b2)];
      if (d0 | d1 | d2) == 0xFF {
        return Err(PhcError::InvalidBase64);
      }
      // Trailing 2 bits of d2 must be zero (strict mode).
      if (d2 & 0x03) != 0 {
        return Err(PhcError::InvalidBase64);
      }
      *output0 = (d0 << 2) | (d1 >> 4);
      *output1 = (d1 << 4) | (d2 >> 2);
    }
    _ => return Err(PhcError::InvalidBase64),
  }

  Ok(expected_out)
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
    let [digit, _, _, _] = (v % 10).to_le_bytes();
    digits[len] = b'0'.wrapping_add(digit);
    v /= 10;
    len = len.strict_add(1);
  }
  for i in (0..len).rev() {
    out.push(char::from(digits[i]));
  }
}

// ─── Parameter scanner (k=v,k=v,...) ────────────────────────────────────────

/// Iterator over comma-separated `key=value` pairs.
///
/// Both key and value are borrowed substrings of the original input. Empty
/// keys, empty values, missing `=`, and empty pair segments are reported as
/// `PhcError::MalformedParams`.
pub(crate) struct PhcParamIter<'a> {
  rest: Option<&'a str>,
}

impl<'a> PhcParamIter<'a> {
  pub(crate) fn new(params: &'a str) -> Self {
    Self {
      rest: (!params.is_empty()).then_some(params),
    }
  }
}

impl<'a> Iterator for PhcParamIter<'a> {
  type Item = Result<(&'a str, &'a str), PhcError>;

  fn next(&mut self) -> Option<Self::Item> {
    let rest = self.rest?;
    let (pair, remaining) = match rest.split_once(',') {
      Some((pair, remaining)) => (pair, Some(remaining)),
      None => (rest, None),
    };
    self.rest = remaining;

    if pair.is_empty() {
      return Some(Err(PhcError::MalformedParams));
    }
    let (key, value) = match pair.split_once('=') {
      Some(fields) => fields,
      None => return Some(Err(PhcError::MalformedParams)),
    };
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
  if matches!(bytes, [b'0', _, ..]) {
    return Err(PhcError::MalformedParams);
  }
  let mut acc = 0u32;
  for &b in bytes {
    if !b.is_ascii_digit() {
      return Err(PhcError::MalformedParams);
    }
    let digit = u32::from(b.wrapping_sub(b'0'));
    acc = acc
      .checked_mul(10)
      .and_then(|prefix| prefix.checked_add(digit))
      .ok_or(PhcError::ParamOutOfRange)?;
  }
  Ok(acc)
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

/// Internal parse or decode error for PHC-format strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PhcError {
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
    for len in 0u8..=64 {
      let input: Vec<u8> = (0..len).map(|i| i.wrapping_mul(31).wrapping_add(7)).collect();
      let mut encoded = String::new();
      base64_encode_into(&input, &mut encoded);

      let mut decoded = vec![0u8; base64_decoded_len(encoded.len())];
      let n = base64_decode_into(&encoded, &mut decoded).expect("encoder output must be canonical base64");
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

    let mut empty = [];
    assert_eq!(
      base64_decode_into("A", &mut empty),
      Err(PhcError::InvalidBase64),
      "an invalid tail takes precedence over destination sizing"
    );
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
    assert_eq!(
      it.next()
        .expect("one parameter must be present")
        .expect("the parameter must be well-formed"),
      ("m", "65536")
    );
    assert!(it.next().is_none());
  }

  #[test]
  fn param_iter_multiple_pairs() {
    let mut it = PhcParamIter::new("m=65536,t=3,p=4");
    for expected in [("m", "65536"), ("t", "3"), ("p", "4")] {
      assert_eq!(
        it.next()
          .expect("the expected parameter must be present")
          .expect("the parameter must be well-formed"),
        expected
      );
    }
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
    assert_eq!(
      it.next().expect("the malformed parameter must be emitted"),
      Err(PhcError::MalformedParams)
    );
  }

  #[test]
  fn param_iter_rejects_empty_pair_segment() {
    let mut it = PhcParamIter::new("m=1,,p=2");
    assert_eq!(
      it.next()
        .expect("the first parameter must be present")
        .expect("the first parameter must be well-formed"),
      ("m", "1")
    );
    assert_eq!(
      it.next().expect("the empty parameter must be emitted"),
      Err(PhcError::MalformedParams)
    );
  }

  #[test]
  fn param_iter_rejects_trailing_empty_pair_segment() {
    let mut it = PhcParamIter::new("m=1,");
    assert_eq!(
      it.next()
        .expect("the first parameter must be present")
        .expect("the first parameter must be well-formed"),
      ("m", "1")
    );
    assert_eq!(
      it.next().expect("the trailing empty parameter must be emitted"),
      Err(PhcError::MalformedParams)
    );
    assert!(it.next().is_none());
  }

  #[test]
  fn param_iter_rejects_empty_key() {
    let mut it = PhcParamIter::new("=65536");
    assert_eq!(
      it.next().expect("the malformed parameter must be emitted"),
      Err(PhcError::MalformedParams)
    );
  }

  #[test]
  fn param_iter_rejects_empty_value() {
    let mut it = PhcParamIter::new("m=");
    assert_eq!(
      it.next().expect("the malformed parameter must be emitted"),
      Err(PhcError::MalformedParams)
    );
  }

  #[test]
  fn parse_param_u32_accepts_valid() {
    for (encoded, expected) in [("0", 0), ("1", 1), ("65536", 65_536), ("4294967295", u32::MAX)] {
      assert_eq!(
        parse_param_u32(encoded).expect("canonical u32 text must parse"),
        expected
      );
    }
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
    let parts = parse(encoded).expect("canonical Argon2id PHC text must parse");
    assert_eq!(parts.algorithm, "argon2id");
    assert_eq!(parts.version, Some("19"));
    assert_eq!(parts.parameters, "m=65536,t=3,p=4");
    assert_eq!(parts.salt_b64, "c29tZXNhbHQ");
    assert_eq!(parts.hash_b64, "c29tZWhhc2g");
  }

  #[test]
  fn parse_scrypt_no_version() {
    let encoded = "$scrypt$ln=14,r=8,p=1$c29tZXNhbHQ$c29tZWhhc2g";
    let parts = parse(encoded).expect("canonical scrypt PHC text must parse");
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
    let parts = parse("$argon2id$m=1,t=1,p=1$c29tZQ$c29tZQ").expect("PHC text without a version segment must parse");
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
