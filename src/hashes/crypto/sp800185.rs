//! SP 800-185 framing helpers shared by cSHAKE and KMAC.

use super::keccak::KeccakCore;

/// cSHAKE128 / KMAC128 bitrate in bytes.
pub(crate) const RATE_128: usize = 168;

/// cSHAKE256 / KMAC256 bitrate in bytes.
pub(crate) const RATE_256: usize = 136;

#[inline]
fn encode_u64_be(value: u64, out: &mut [u8; 9], right: bool) -> usize {
  let bytes = value.to_be_bytes();
  let first = bytes.iter().position(|&byte| byte != 0).unwrap_or(7);
  let width = 8usize.strict_sub(first);

  if right {
    out[..width].copy_from_slice(&bytes[first..]);
    out[width] = u8::try_from(width).expect("SP 800-185 integer width is at most eight bytes");
    width.strict_add(1)
  } else {
    out[0] = u8::try_from(width).expect("SP 800-185 integer width is at most eight bytes");
    out[1..=width].copy_from_slice(&bytes[first..]);
    width.strict_add(1)
  }
}

#[inline]
pub(crate) fn left_encode(value: u64) -> ([u8; 9], usize) {
  let mut out = [0u8; 9];
  let len = encode_u64_be(value, &mut out, false);
  (out, len)
}

#[inline]
#[cfg(feature = "kmac")]
pub(crate) fn right_encode(value: u64) -> ([u8; 9], usize) {
  let mut out = [0u8; 9];
  let len = encode_u64_be(value, &mut out, true);
  (out, len)
}

#[inline]
pub(crate) fn encoded_string_len(data: &[u8]) -> usize {
  left_encode(crate::bytes_to_bits(data.len())).1.strict_add(data.len())
}

pub(crate) fn absorb_bytepad<const RATE: usize>(core: &mut KeccakCore<RATE>, segments: &[&[u8]], payload_len: usize) {
  let rate = u64::try_from(RATE).expect("Keccak rate must fit u64");
  let (prefix, prefix_len) = left_encode(rate);
  core.update(&prefix[..prefix_len]);
  for segment in segments {
    core.update(segment);
  }

  let total_len = prefix_len.strict_add(payload_len);
  let rem = total_len.strict_rem(RATE);
  if rem != 0 {
    core.update(&[0u8; RATE][..RATE.strict_sub(rem)]);
  }
}
