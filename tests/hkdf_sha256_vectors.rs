#![cfg(feature = "hkdf")]

use rscrypto::{HkdfSha256, auth::HkdfOutputLengthError};

mod common;
#[path = "common/array.rs"]
mod hex_array;
use common::decode_hex_vec;

fn decode_hex_array<const N: usize>(hex: &str) -> [u8; N] {
  hex_array::decode_hex_array(&hex.replace('\n', ""))
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

  let okm = hkdf
    .expand_array::<42>(&info)
    .expect("RFC 5869 HKDF-SHA-256 case 1 expansion must succeed");
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
  let okm = HkdfSha256::derive_array::<42>(b"", &[0x0b; 22], b"")
    .expect("RFC 5869 HKDF-SHA-256 case 3 derivation must succeed");
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
  let mut out = vec![0u8; HkdfSha256::MAX_OUTPUT_SIZE.strict_add(1)];
  let err =
    HkdfSha256::derive(b"salt", b"ikm", b"info", &mut out).expect_err("HKDF-SHA-256 must reject oversized output");
  assert_eq!(err, HkdfOutputLengthError::new());
}

/// RFC 5869 Appendix A.1-A.3 as (case, PRK, info, OKM).
const RFC5869_PRK_CASES: [(&str, &str, &str, &str); 3] = [
  (
    "A.1",
    "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5",
    "f0f1f2f3f4f5f6f7f8f9",
    "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865",
  ),
  (
    "A.2",
    "06a6b88c5853361a06104c9ceb35b45cef760014904671014a193f40c15fc244",
    concat!(
      "b0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecf",
      "d0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeef",
      "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff",
    ),
    concat!(
      "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c",
      "59045a99cac7827271cb41c65e590e09da3275600c2f09b8367793a9aca3db71",
      "cc30c58179ec3e87c14c01d5c1f3434f1d87",
    ),
  ),
  (
    "A.3",
    "19ef24a32c717b167f33a91d6f648bdf96596776afdb6377ac434c1c293ccb04",
    "",
    "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8",
  ),
];

#[test]
fn hkdf_sha256_rfc5869_from_prk_with_split_info() {
  for (case, prk, info, okm) in RFC5869_PRK_CASES {
    let prk = decode_hex_array::<32>(prk);
    let info = decode_hex_vec(info);
    let expected = decode_hex_vec(okm);

    let hkdf = HkdfSha256::from_prk(&prk);
    assert_eq!(hkdf.prk(), &prk, "RFC 5869 {case}");
    for split in 0..=info.len() {
      let (head, tail) = info.split_at(split);
      let mut okm = vec![0u8; expected.len()];
      hkdf
        .expand_vectored(&[head, tail], &mut okm)
        .expect("RFC 5869 output lengths are within the HKDF-SHA-256 limit");
      assert_eq!(okm, expected, "RFC 5869 {case} split {split}");
    }
  }
}

#[test]
fn hkdf_sha256_expand_vectored_enforces_the_output_bound() {
  let prk = decode_hex_array::<32>(RFC5869_PRK_CASES[1].1);
  let info: [&[u8]; 2] = [b"in", b"fo"];
  let hkdf = HkdfSha256::from_prk(&prk);

  let mut oversized = vec![0xa5u8; HkdfSha256::MAX_OUTPUT_SIZE.strict_add(1)];
  assert_eq!(
    hkdf.expand_vectored(&info, &mut oversized),
    Err(HkdfOutputLengthError::new())
  );
  assert!(
    oversized.iter().all(|&byte| byte == 0xa5),
    "a rejected expansion must not write the output"
  );

  let mut ours = vec![0u8; HkdfSha256::MAX_OUTPUT_SIZE];
  hkdf
    .expand_vectored(&info, &mut ours)
    .expect("HKDF-SHA-256 must accept the maximum output length");
  let mut expected = vec![0u8; HkdfSha256::MAX_OUTPUT_SIZE];
  hkdf::Hkdf::<sha2::Sha256>::from_prk(&prk)
    .expect("a 32-byte PRK is valid for RustCrypto HKDF-SHA-256")
    .expand_multi_info(&info, &mut expected)
    .expect("RustCrypto must accept the maximum output length");
  assert_eq!(ours, expected);
}
