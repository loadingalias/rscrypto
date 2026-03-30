//! SP 800-185 framing helpers shared by cSHAKE and KMAC.

#![allow(clippy::indexing_slicing)] // Fixed-width 9-byte encodings and rate-sized padding buffers.

use super::keccak::KeccakCore;

/// cSHAKE256 / KMAC256 bitrate in bytes.
pub(crate) const RATE_256: usize = 136;

#[inline]
fn encode_u64_be(value: u64, out: &mut [u8; 9], right: bool) -> usize {
  let bytes = value.to_be_bytes();
  let first = bytes.iter().position(|&byte| byte != 0).unwrap_or(7);
  let width = 8usize.strict_sub(first);

  if right {
    out[..width].copy_from_slice(&bytes[first..]);
    out[width] = width as u8;
    width.strict_add(1)
  } else {
    out[0] = width as u8;
    out[1..=width].copy_from_slice(&bytes[first..]);
    width.strict_add(1)
  }
}

#[inline]
fn bits_from_len(len: usize) -> u64 {
  match u64::try_from(len) {
    Ok(value) => value.strict_mul(8),
    Err(_) => panic!("length exceeds u64"),
  }
}

#[inline]
pub(crate) fn left_encode(value: u64) -> ([u8; 9], usize) {
  let mut out = [0u8; 9];
  let len = encode_u64_be(value, &mut out, false);
  (out, len)
}

#[inline]
#[cfg(feature = "auth")]
pub(crate) fn right_encode(value: u64) -> ([u8; 9], usize) {
  let mut out = [0u8; 9];
  let len = encode_u64_be(value, &mut out, true);
  (out, len)
}

#[inline]
pub(crate) fn encoded_string_len(data: &[u8]) -> usize {
  left_encode(bits_from_len(data.len())).1.strict_add(data.len())
}

#[inline]
pub(crate) fn absorb_encoded_string<const RATE: usize>(core: &mut KeccakCore<RATE>, data: &[u8]) {
  let (prefix, prefix_len) = left_encode(bits_from_len(data.len()));
  core.update(&prefix[..prefix_len]);
  core.update(data);
}

#[cfg(feature = "auth")]
pub(crate) fn absorb_bytepad<const RATE: usize>(core: &mut KeccakCore<RATE>, segments: &[&[u8]], payload_len: usize) {
  let (prefix, prefix_len) = left_encode(RATE as u64);
  core.update(&prefix[..prefix_len]);
  for segment in segments {
    core.update(segment);
  }

  let total_len = prefix_len.strict_add(payload_len);
  let pad_len = (RATE.strict_sub(total_len % RATE)) % RATE;
  if pad_len != 0 {
    core.update(&[0u8; RATE][..pad_len]);
  }
}
