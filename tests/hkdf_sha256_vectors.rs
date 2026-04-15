#![cfg(feature = "hkdf")]

use rscrypto::{HkdfSha256, auth::HkdfOutputLengthError};

mod common;
use common::decode_hex_vec;

fn decode_hex_array<const N: usize>(hex: &str) -> [u8; N] {
  common::decode_hex_array(&hex.replace('\n', ""))
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
