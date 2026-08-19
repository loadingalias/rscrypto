//! Shared test utilities for integration tests.

/// Decode a hex string into a `Vec<u8>`.
///
/// Panics on odd-length input or invalid hex characters.
pub(crate) fn decode_hex_vec(hex: &str) -> Vec<u8> {
  assert_eq!(hex.len() % 2, 0, "hex length must be even");
  hex
    .as_bytes()
    .chunks_exact(2)
    .map(|pair| {
      let high = nibble(pair[0]).expect("hex input must contain only hexadecimal digits");
      let low = nibble(pair[1]).expect("hex input must contain only hexadecimal digits");
      (high << 4) | low
    })
    .collect()
}

fn nibble(b: u8) -> Option<u8> {
  match b {
    b'0'..=b'9' => Some(b.strict_sub(b'0')),
    b'a'..=b'f' => Some(b.strict_sub(b'a').strict_add(10)),
    b'A'..=b'F' => Some(b.strict_sub(b'A').strict_add(10)),
    _ => None,
  }
}
