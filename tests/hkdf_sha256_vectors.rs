#![cfg(feature = "hkdf")]

use rscrypto::{HkdfSha256, auth::HkdfOutputLengthError};

fn decode_hex_vec(hex: &str) -> Vec<u8> {
  assert_eq!(hex.len() % 2, 0, "hex length must be even");
  let mut out = Vec::with_capacity(hex.len() / 2);
  let bytes = hex.as_bytes();
  let mut i = 0usize;
  while i < bytes.len() {
    let hi = char::from(bytes[i]).to_digit(16).unwrap();
    let lo = char::from(bytes[i + 1]).to_digit(16).unwrap();
    out.push(((hi << 4) | lo) as u8);
    i += 2;
  }
  out
}

fn decode_hex_array<const N: usize>(hex: &str) -> [u8; N] {
  decode_hex_vec(&hex.replace('\n', "")).try_into().unwrap()
}

#[test]
fn hkdf_sha256_rfc5869_case_1() {
  let ikm = [0x0b; 22];
  let salt = decode_hex_vec("000102030405060708090a0b0c");
  let info = decode_hex_vec("f0f1f2f3f4f5f6f7f8f9");

  let hkdf = HkdfSha256::new(&salt, &ikm);
  assert_eq!(
    hkdf.prk(),
    &decode_hex_array::<32>("077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5")
  );

  let okm = hkdf.expand_array::<42>(&info).unwrap();
  assert_eq!(
    okm,
    decode_hex_array::<42>(
      "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34
007208d5b887185865",
    )
  );
}

#[test]
fn hkdf_sha256_rfc5869_case_3() {
  let okm = HkdfSha256::derive_array::<42>(b"", &[0x0b; 22], b"").unwrap();
  assert_eq!(
    okm,
    decode_hex_array::<42>(
      "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d
201395faa4b61a96c8",
    )
  );
}

#[test]
fn hkdf_sha256_rejects_oversized_output() {
  let mut out = vec![0u8; HkdfSha256::MAX_OUTPUT_SIZE + 1];
  let err = HkdfSha256::derive(b"salt", b"ikm", b"info", &mut out).unwrap_err();
  assert_eq!(err, HkdfOutputLengthError::new());
}
