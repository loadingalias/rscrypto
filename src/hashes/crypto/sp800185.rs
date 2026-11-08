//! SP 800-185 framing helpers shared by cSHAKE and KMAC.

use super::keccak::KeccakCore;

/// SHAKE domain and padding byte: cSHAKE with empty function name and customization.
pub(crate) const SHAKE_PAD: u8 = 0x1F;

/// cSHAKE domain and padding byte.
pub(crate) const CSHAKE_PAD: u8 = 0x04;

/// cSHAKE128 / KMAC128 bitrate in bytes.
pub(crate) const RATE_128: usize = 168;

/// cSHAKE256 / KMAC256 bitrate in bytes.
pub(crate) const RATE_256: usize = 136;

/// Encodes `value` as SP 800-185 does: its big-endian bytes without leading
/// zeros, at least one byte, with their count before them (`left_encode`) or
/// after them (`right_encode`). Returns the buffer and the encoded length.
#[inline(always)]
fn encode_u64_be(value: u64, right: bool) -> ([u8; 9], usize) {
  // `value | 1` has the byte width of every nonzero `value`, and zero
  // encodes in one byte.
  let zero_bytes =
    usize::try_from((value | 1).leading_zeros() / 8).expect("a u64 has at most seven leading zero bytes");
  let width = 8usize.strict_sub(zero_bytes);
  // Shifting out the leading zero bytes moves the significant bytes to the
  // front of the big-endian form, so fixed-size stores replace a
  // variable-length copy.
  let digits = (value << zero_bytes.strict_mul(8)).to_be_bytes();
  let count = u8::try_from(width).expect("SP 800-185 integer width is at most eight bytes");

  let mut out = [0u8; 9];
  if right {
    out[..8].copy_from_slice(&digits);
    out[width] = count;
  } else {
    out[0] = count;
    out[1..].copy_from_slice(&digits);
  }
  (out, width.strict_add(1))
}

#[inline]
pub(crate) fn left_encode(value: u64) -> ([u8; 9], usize) {
  encode_u64_be(value, false)
}

#[inline]
#[cfg(feature = "kmac")]
pub(crate) fn right_encode(value: u64) -> ([u8; 9], usize) {
  encode_u64_be(value, true)
}

#[inline]
pub(crate) fn encoded_string_len(data: &[u8]) -> usize {
  left_encode(crate::bytes_to_bits(data.len())).1.strict_add(data.len())
}

pub(crate) fn absorb_bytepad<const RATE: usize>(core: &mut KeccakCore<RATE>, segments: &[&[u8]], payload_len: usize) {
  let rate = u64::try_from(RATE).expect("Keccak rate must fit u64");
  let (prefix, prefix_len) = left_encode(rate);
  core.update_short(&prefix[..prefix_len]);
  for segment in segments {
    core.update_short(segment);
  }

  let total_len = prefix_len.strict_add(payload_len);
  let rem = total_len.strict_rem(RATE);
  if rem != 0 {
    core.absorb_zeros(RATE.strict_sub(rem));
  }
}

/// Absorbs the cSHAKE prefix for `function_name` and `customization` into
/// `core`, a new empty sponge, and returns the padding byte. With both strings
/// empty, cSHAKE is SHAKE. The caller owns the sponge, so returning moves no
/// state.
pub(crate) fn absorb_cshake_prefix<const RATE: usize>(
  core: &mut KeccakCore<RATE>,
  function_name: &[u8],
  customization: &[u8],
) -> u8 {
  if function_name.is_empty() && customization.is_empty() {
    return SHAKE_PAD;
  }

  let (function_prefix, function_prefix_len) = left_encode(crate::bytes_to_bits(function_name.len()));
  let (custom_prefix, custom_prefix_len) = left_encode(crate::bytes_to_bits(customization.len()));
  let payload_len = encoded_string_len(function_name).strict_add(encoded_string_len(customization));
  absorb_bytepad::<RATE>(
    core,
    &[
      &function_prefix[..function_prefix_len],
      function_name,
      &custom_prefix[..custom_prefix_len],
      customization,
    ],
    payload_len,
  );
  CSHAKE_PAD
}

#[cfg(test)]
mod tests {
  /// SP 800-185 section 2.3.1 encodings at each byte-width boundary: the
  /// big-endian bytes without leading zeros, and the byte count before or
  /// after them. Zero encodes in one byte.
  const CASES: &[(u64, &[u8], &[u8])] = &[
    (0, &[1, 0x00], &[0x00, 1]),
    (1, &[1, 0x01], &[0x01, 1]),
    (0xff, &[1, 0xff], &[0xff, 1]),
    (0x100, &[2, 0x01, 0x00], &[0x01, 0x00, 2]),
    (0xffff, &[2, 0xff, 0xff], &[0xff, 0xff, 2]),
    (0x1_0000, &[3, 0x01, 0x00, 0x00], &[0x01, 0x00, 0x00, 3]),
    (0x1234_5678, &[4, 0x12, 0x34, 0x56, 0x78], &[0x12, 0x34, 0x56, 0x78, 4]),
    (
      0x00ff_ffff_ffff_ffff,
      &[7, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
      &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 7],
    ),
    (
      0x0100_0000_0000_0000,
      &[8, 0x01, 0, 0, 0, 0, 0, 0, 0],
      &[0x01, 0, 0, 0, 0, 0, 0, 0, 8],
    ),
    (
      u64::MAX,
      &[8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
      &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 8],
    ),
  ];

  #[test]
  fn left_encode_matches_sp800_185_at_width_boundaries() {
    for &(value, left, _) in CASES {
      let (out, len) = super::left_encode(value);
      assert_eq!(&out[..len], left, "left_encode({value:#x})");
    }
  }

  #[cfg(feature = "kmac")]
  #[test]
  fn right_encode_matches_sp800_185_at_width_boundaries() {
    for &(value, _, right) in CASES {
      let (out, len) = super::right_encode(value);
      assert_eq!(&out[..len], right, "right_encode({value:#x})");
    }
  }
}
