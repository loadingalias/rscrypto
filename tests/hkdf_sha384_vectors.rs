#![cfg(feature = "hkdf")]

use hkdf::Hkdf as RustCryptoHkdf;
use rscrypto::{HkdfSha384, auth::HkdfOutputLengthError};

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
  decode_hex_vec(hex).try_into().unwrap()
}

#[test]
fn hkdf_sha384_case_1() {
  let ikm = [0x0b; 22];
  let salt = decode_hex_vec("000102030405060708090a0b0c");
  let info = decode_hex_vec("f0f1f2f3f4f5f6f7f8f9");

  let hkdf = HkdfSha384::new(&salt, &ikm);
  assert_eq!(
    hkdf.prk(),
    &decode_hex_array::<48>(
      "704b39990779ce1dc548052c7dc39f303570dd13fb39f7acc564680bef80e8dec70ee9a7e1f3e293ef68eceb072a5ade",
    )
  );

  let okm = hkdf.expand_array::<42>(&info).unwrap();
  assert_eq!(
    okm,
    decode_hex_array::<42>("9b5097a86038b805309076a44b3a9f38063e25b516dcbf369f394cfab43685f748b6457763e4f0204fc5",)
  );
}

#[test]
fn hkdf_sha384_case_2() {
  let ikm = decode_hex_vec(
    "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f\
     202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f\
     404142434445464748494a4b4c4d4e4f",
  );
  let salt = decode_hex_vec(
    "606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f\
     808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f\
     a0a1a2a3a4a5a6a7a8a9aaabacadaeaf",
  );
  let info = decode_hex_vec(
    "b0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecf\
     d0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeef\
     f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff",
  );

  let hkdf = HkdfSha384::new(&salt, &ikm);
  assert_eq!(
    hkdf.prk(),
    &decode_hex_array::<48>(
      "b319f6831dff9314efb643baa29263b30e4a8d779fe31e9c901efd7de737c85b62e676d4dc87b0895c6a7dc97b52cebb",
    )
  );

  let okm = hkdf.expand_array::<82>(&info).unwrap();
  assert_eq!(
    okm,
    decode_hex_array::<82>(
      "484ca052b8cc724fd1c4ec64d57b4e818c7e25a8e0f4569ed72a6a05fe0649eebf69f8d5c832856bf4e4fbc17967d549\
       75324a94987f7f41835817d8994fdbd6f4c09c5500dca24a56222fea53d8967a8b2e",
    )
  );
}

#[test]
fn hkdf_sha384_case_3() {
  let okm = HkdfSha384::derive_array::<42>(b"", &[0x0b; 22], b"").unwrap();
  assert_eq!(
    okm,
    decode_hex_array::<42>("c8c96e710f89b0d7990bca68bcdec8cf854062e54c73a7abc743fade9b242daacc1cea5670415b52849c",)
  );
}

#[test]
fn hkdf_sha384_matches_rustcrypto() {
  let salt = [0x11u8; 48];
  let ikm = [0x22u8; 48];
  let info = [0x33u8; 80];
  let hkdf = HkdfSha384::new(&salt, &ikm);
  let rustcrypto = RustCryptoHkdf::<sha2::Sha384>::new(Some(&salt), &ikm);

  for len in [1usize, 32, 48, 49, 96, 256, 1024] {
    let mut ours = vec![0u8; len];
    let mut theirs = vec![0u8; len];

    hkdf.expand(&info, &mut ours).unwrap();
    rustcrypto.expand(&info, &mut theirs).unwrap();

    assert_eq!(ours, theirs, "HKDF-SHA384 mismatch at output len {len}");
  }
}

#[test]
fn hkdf_sha384_derive_matches_extract_then_expand() {
  let salt = b"salt";
  let ikm = b"input key material";
  let info = b"context";

  let extracted = HkdfSha384::new(salt, ikm);
  let derived = HkdfSha384::derive_array::<96>(salt, ikm, info).unwrap();
  assert_eq!(derived, extracted.expand_array::<96>(info).unwrap());
}

#[test]
fn hkdf_sha384_rejects_oversized_output() {
  let mut out = vec![0u8; HkdfSha384::MAX_OUTPUT_SIZE + 1];
  let err = HkdfSha384::derive(b"salt", b"ikm", b"info", &mut out).unwrap_err();
  assert_eq!(err, HkdfOutputLengthError::new());
}
