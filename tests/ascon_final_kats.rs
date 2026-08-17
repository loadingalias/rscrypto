#![cfg(any(feature = "ascon-aead", feature = "ascon-hash"))]

mod common;

#[cfg(feature = "ascon-aead")]
#[path = "common/array.rs"]
mod hex_array;
use common::decode_hex_vec;
#[cfg(feature = "ascon-aead")]
use hex_array::decode_hex_array;

#[cfg(feature = "ascon-aead")]
const AEAD_VECTORS: &str = include_str!("../testdata/ascon/asconaead128.txt");
#[cfg(feature = "ascon-hash")]
const CXOF_VECTORS: &str = include_str!("../testdata/ascon/asconcxof128.txt");

fn cases(data: &str) -> impl Iterator<Item = &str> {
  data.split("\n\n").filter(|case| !case.trim().is_empty())
}

fn field<'a>(case: &'a str, name: &str) -> &'a str {
  case
    .lines()
    .find_map(|line| line.split_once(" = ").filter(|(key, _)| *key == name))
    .map(|(_, value)| value)
    .expect("Ascon KAT field must be present")
}

fn case_number(case: &str) -> usize {
  field(case, "Count")
    .parse()
    .expect("Ascon KAT case number must be valid")
}

#[cfg(feature = "ascon-aead")]
#[test]
fn ascon_aead128_matches_final_reference_kats() {
  use rscrypto::{
    AsconAead128, AsconAead128Key, AsconAead128Tag,
    aead::{Nonce128, expert::AeadWithNonce as _},
  };

  let mut rows = 0usize;
  for case in cases(AEAD_VECTORS) {
    let count = case_number(case);
    let key = AsconAead128Key::from_bytes(decode_hex_array(field(case, "Key")));
    let nonce = Nonce128::from_bytes(decode_hex_array(field(case, "Nonce")));
    let plaintext = decode_hex_vec(field(case, "PT"));
    let aad = decode_hex_vec(field(case, "AD"));
    let combined = decode_hex_vec(field(case, "CT"));
    let tag_start = combined
      .len()
      .checked_sub(AsconAead128::TAG_SIZE)
      .expect("Ascon-AEAD128 KAT output is shorter than its tag");
    let (expected_ciphertext, expected_tag) = combined.split_at(tag_start);
    let expected_tag = AsconAead128Tag::from_bytes(
      expected_tag
        .try_into()
        .expect("Ascon-AEAD128 KAT tag has the wrong length"),
    );
    let cipher = AsconAead128::new(&key);

    let mut ciphertext = plaintext.clone();
    let tag = cipher
      .encrypt_in_place(&nonce, &aad, &mut ciphertext)
      .expect("Ascon-AEAD128 KAT encryption must succeed");
    assert_eq!(
      ciphertext, expected_ciphertext,
      "Ascon-AEAD128 KAT {count} ciphertext mismatch"
    );
    assert_eq!(
      tag.as_bytes(),
      expected_tag.as_bytes(),
      "Ascon-AEAD128 KAT {count} tag mismatch"
    );

    let mut decrypted = expected_ciphertext.to_vec();
    cipher
      .decrypt_in_place(&nonce, &aad, &mut decrypted, &expected_tag)
      .expect("Ascon-AEAD128 KAT decryption must succeed");
    assert_eq!(decrypted, plaintext, "Ascon-AEAD128 KAT {count} plaintext mismatch");

    rows = rows.strict_add(1);
  }
  assert_eq!(rows, 1089, "incomplete Ascon-AEAD128 KAT corpus");
}

#[cfg(feature = "ascon-hash")]
#[test]
fn ascon_cxof128_matches_final_reference_kats() {
  use rscrypto::{AsconCxof128, traits::Xof as _};

  let mut rows = 0usize;
  for case in cases(CXOF_VECTORS) {
    let count = case_number(case);
    let message = decode_hex_vec(field(case, "Msg"));
    let customization = decode_hex_vec(field(case, "Z"));
    let expected = decode_hex_vec(field(case, "MD"));
    let mut actual = vec![0u8; expected.len()];

    AsconCxof128::xof(&customization, &message)
      .expect("Ascon-CXOF128 KAT setup must succeed")
      .squeeze(&mut actual);
    assert_eq!(actual, expected, "Ascon-CXOF128 KAT {count} output mismatch");

    rows = rows.strict_add(1);
  }
  assert_eq!(rows, 1089, "incomplete Ascon-CXOF128 KAT corpus");
}
