#![cfg(feature = "auth")]

use rscrypto::HmacSha256;

fn decode_hex<const N: usize>(hex: &str) -> [u8; N] {
  assert_eq!(hex.len(), N * 2, "hex length must match output size");
  let mut out = [0u8; N];
  let bytes = hex.as_bytes();
  let mut i = 0usize;
  while i < N {
    let hi = char::from(bytes[i * 2]).to_digit(16).unwrap();
    let lo = char::from(bytes[i * 2 + 1]).to_digit(16).unwrap();
    out[i] = ((hi << 4) | lo) as u8;
    i += 1;
  }
  out
}

#[test]
fn hmac_sha256_rfc4231_vectors() {
  let cases: &[(&[u8], &[u8], &str)] = &[
    (
      &[0x0b; 20],
      b"Hi There",
      "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7",
    ),
    (
      b"Jefe",
      b"what do ya want for nothing?",
      "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
    ),
    (
      &[0xaa; 20],
      &[0xdd; 50],
      "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe",
    ),
    (
      &[0xaa; 131],
      b"Test Using Larger Than Block-Size Key - Hash Key First",
      "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54",
    ),
  ];

  for (i, (key, data, expected_hex)) in cases.iter().enumerate() {
    let expected = decode_hex::<32>(&expected_hex.replace('\n', ""));
    let actual = HmacSha256::mac(key, data);
    assert_eq!(actual, expected, "HMAC-SHA256 RFC 4231 vector {i} mismatch");
    assert!(HmacSha256::verify_tag(key, data, &expected).is_ok());
  }
}

#[test]
fn hmac_sha256_verify_rejects_corrupted_tag() {
  let key = b"shared-secret";
  let data = b"auth message";
  let mut tag = HmacSha256::mac(key, data);
  tag[0] ^= 0x80;
  assert!(HmacSha256::verify_tag(key, data, &tag).is_err());
}
