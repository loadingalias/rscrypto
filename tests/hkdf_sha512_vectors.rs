#![cfg(feature = "hkdf")]

use hkdf::Hkdf as RustCryptoHkdf;
use rscrypto::{HkdfSha512, auth::HkdfOutputLengthError};

mod common;
use common::{decode_hex_array, decode_hex_vec};

#[test]
fn hkdf_sha512_rfc5869_case_1() {
  let ikm = [0x0b; 22];
  let salt = decode_hex_vec("000102030405060708090a0b0c");
  let info = decode_hex_vec("f0f1f2f3f4f5f6f7f8f9");

  let hkdf = HkdfSha512::new(&salt, &ikm);
  assert_eq!(
    hkdf.prk(),
    &decode_hex_array::<64>(
      "665799823737ded04a88e47e54a5890bb2c3d247c7a4254a8e61350723590a26\
       c36238127d8661b88cf80ef802d57e2f7cebcf1e00e083848be19929c61b4237",
    )
  );

  let okm = hkdf.expand_array::<42>(&info).unwrap();
  assert_eq!(
    okm,
    decode_hex_array::<42>("832390086cda71fb47625bb5ceB168e4c8e26a1a16ed34d9fc7fe92c1481579338da362cb8d9f925d7cb",)
  );
}

#[test]
fn hkdf_sha512_matches_rustcrypto() {
  let salt = [0x11u8; 64];
  let ikm = [0x22u8; 80];
  let info = [0x33u8; 96];
  let hkdf = HkdfSha512::new(&salt, &ikm);
  let rustcrypto = RustCryptoHkdf::<sha2::Sha512>::new(Some(&salt), &ikm);

  for len in [1usize, 32, 64, 65, 128, 256, 1024] {
    let mut ours = vec![0u8; len];
    let mut theirs = vec![0u8; len];

    hkdf.expand(&info, &mut ours).unwrap();
    rustcrypto.expand(&info, &mut theirs).unwrap();

    assert_eq!(ours, theirs, "HKDF-SHA512 mismatch at output len {len}");
  }
}

#[test]
fn hkdf_sha512_derive_matches_extract_then_expand() {
  let salt = b"salt";
  let ikm = b"input key material";
  let info = b"context";

  let extracted = HkdfSha512::new(salt, ikm);
  let derived = HkdfSha512::derive_array::<128>(salt, ikm, info).unwrap();
  assert_eq!(derived, extracted.expand_array::<128>(info).unwrap());
}

#[test]
fn hkdf_sha512_rejects_oversized_output() {
  let mut out = vec![0u8; HkdfSha512::MAX_OUTPUT_SIZE + 1];
  let err = HkdfSha512::derive(b"salt", b"ikm", b"info", &mut out).unwrap_err();
  assert_eq!(err, HkdfOutputLengthError::new());
}
