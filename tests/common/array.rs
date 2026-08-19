//! Fixed-size hexadecimal test-vector decoding.

/// Decode a hex string into a fixed-size byte array.
///
/// Panics if the decoded length does not equal `N`.
pub(crate) fn decode_hex_array<const N: usize>(hex: &str) -> [u8; N] {
  crate::common::decode_hex_vec(hex)
    .try_into()
    .expect("hex length must match the destination array")
}
