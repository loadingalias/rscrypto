//! Shared test utilities for integration tests.

/// Decode a hex string into a `Vec<u8>`.
///
/// Panics on odd-length input or invalid hex characters.
#[allow(dead_code)]
pub fn decode_hex_vec(hex: &str) -> Vec<u8> {
  assert_eq!(hex.len() % 2, 0, "hex length must be even");
  hex
    .as_bytes()
    .chunks_exact(2)
    .map(|pair| (nibble(pair[0]) << 4) | nibble(pair[1]))
    .collect()
}

/// Decode a hex string into a fixed-size byte array.
///
/// Panics if the decoded length does not equal `N`.
#[allow(dead_code)]
pub fn decode_hex_array<const N: usize>(hex: &str) -> [u8; N] {
  let v = decode_hex_vec(hex);
  v.try_into().expect("hex length does not match array size")
}

fn nibble(b: u8) -> u8 {
  match b {
    b'0'..=b'9' => b - b'0',
    b'a'..=b'f' => b - b'a' + 10,
    b'A'..=b'F' => b - b'A' + 10,
    _ => panic!("invalid hex byte: {b:#04x}"),
  }
}
