//! X-Wing against its published vectors and against hostile inputs.
//!
//! The vectors are the 3 of draft-connolly-cfrg-xwing-kem-11 Appendix C and
//! the 10 MLKEM768-X25519 vectors of draft-irtf-cfrg-concrete-hybrid-kems-04
//! Appendix B.2. The small-order X25519 encodings come from X-Wing pull
//! request 39. Provenance is in testdata/xwing/README.md.
#![cfg(feature = "x-wing")]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::alloc::Global;

use rscrypto::{
  Kem, XWing, XWingCiphertext, XWingDecapsulationKey, XWingEncapsulationKey, XWingError, XWingPreparedEncapsulationKey,
};
use serde_json::Value;

mod common;
use common::decode_hex_vec;

struct Vector {
  seed: Vec<u8>,
  decapsulation_key: Vec<u8>,
  encapsulation_key: Vec<u8>,
  randomness: Vec<u8>,
  ciphertext: Vec<u8>,
  shared_secret: Vec<u8>,
}

fn field(case: &Value, name: &str) -> Vec<u8> {
  decode_hex_vec(case[name].as_str().expect("hex vector field"))
}

fn vectors() -> Vec<Vector> {
  let draft: Value =
    serde_json::from_str(include_str!("../testdata/xwing/draft-11-test-vectors.json")).expect("draft vectors");
  let concrete: Value = serde_json::from_str(include_str!(
    "../testdata/xwing/concrete-hybrid-kems-04-mlkem768-x25519.json"
  ))
  .expect("concrete hybrid KEM vectors");
  let draft = draft.as_array().expect("draft vector list").iter().map(|case| Vector {
    seed: field(case, "seed"),
    decapsulation_key: field(case, "sk"),
    encapsulation_key: field(case, "pk"),
    randomness: field(case, "eseed"),
    ciphertext: field(case, "ct"),
    shared_secret: field(case, "ss"),
  });
  let concrete = concrete
    .as_array()
    .expect("concrete vector list")
    .iter()
    .map(|case| Vector {
      seed: field(case, "seed"),
      decapsulation_key: field(case, "decapsulation_key"),
      encapsulation_key: field(case, "encapsulation_key"),
      randomness: field(case, "randomness"),
      ciphertext: field(case, "ciphertext"),
      shared_secret: field(case, "shared_secret"),
    });
  let vectors: Vec<Vector> = draft.chain(concrete).collect();
  assert_eq!(vectors.len(), 13);
  vectors
}

fn low_order_points() -> Vec<[u8; 32]> {
  let points: Vec<[u8; 32]> = include_str!("../testdata/xwing/low-order-points.txt")
    .lines()
    .map(|line| decode_hex_vec(line).try_into().expect("32-byte encoding"))
    .collect();
  assert_eq!(points.len(), 14);
  points
}

/// A random source that must be asked for exactly `bytes`.
fn filled(bytes: &[u8]) -> impl FnMut(&mut [u8]) -> Result<(), XWingError> + '_ {
  |out| {
    out.copy_from_slice(bytes);
    Ok(())
  }
}

/// Key generation, encapsulation, and decapsulation reproduce each vector
/// through the `Kem` functions, the prepared keys, and an allocated key.
#[test]
fn vectors_match_on_every_path() {
  for (index, vector) in vectors().iter().enumerate() {
    let (encapsulation_key, decapsulation_key) = XWing::generate_keypair(filled(&vector.seed)).expect("key generation");
    assert_eq!(
      decapsulation_key.as_bytes().as_slice(),
      vector.decapsulation_key,
      "vector {index}: the seed is the decapsulation key"
    );
    assert_eq!(
      encapsulation_key.as_bytes().as_slice(),
      vector.encapsulation_key,
      "vector {index}: encapsulation key"
    );
    let prepared = decapsulation_key.prepare();
    assert_eq!(
      prepared.encapsulation_key(),
      encapsulation_key,
      "vector {index}: prepared encapsulation key"
    );

    let parsed = XWingEncapsulationKey::try_from_slice(&vector.encapsulation_key).expect("vector key");
    let prepared_encapsulation_key = parsed.prepare();
    assert_eq!(prepared_encapsulation_key.encapsulation_key(), parsed);
    for (path, (ciphertext, shared_secret)) in [
      (
        "Kem",
        XWing::encapsulate(&parsed, filled(&vector.randomness)).expect("encapsulation"),
      ),
      (
        "prepared",
        prepared_encapsulation_key
          .encapsulate(filled(&vector.randomness))
          .expect("encapsulation"),
      ),
    ] {
      assert_eq!(
        ciphertext.as_bytes().as_slice(),
        vector.ciphertext,
        "vector {index}: {path} ciphertext"
      );
      assert_eq!(
        shared_secret.as_bytes().as_slice(),
        vector.shared_secret,
        "vector {index}: {path} encapsulated secret"
      );
    }

    let ciphertext = XWingCiphertext::try_from_slice(&vector.ciphertext).expect("vector ciphertext");
    let mut decapsulated = vec![
      ("Kem", XWing::decapsulate(&decapsulation_key, &ciphertext)),
      ("prepared", prepared.decapsulate(&ciphertext)),
    ];
    #[cfg(feature = "alloc")]
    decapsulated.push((
      "allocated",
      decapsulation_key.prepare_in(Global).decapsulate(&ciphertext),
    ));
    for (path, shared_secret) in decapsulated {
      assert_eq!(
        shared_secret.expect("decapsulation").as_bytes().as_slice(),
        vector.shared_secret,
        "vector {index}: {path} decapsulated secret"
      );
    }
  }
}

/// A small-order X25519 part makes the X25519 output all-zero, which X-Wing
/// pull request 39 rejects in both directions. A rejection changes no key.
#[test]
fn small_order_x25519_parts_are_rejected() {
  let vector = &vectors()[0];
  let decapsulation_key = XWingDecapsulationKey::try_from_slice(&vector.decapsulation_key).expect("vector key");
  let prepared = decapsulation_key.prepare();
  let valid_ciphertext = XWingCiphertext::try_from_slice(&vector.ciphertext).expect("vector ciphertext");
  let x25519_offset = XWing::CIPHERTEXT_SIZE - 32;
  let key_x25519_offset = XWing::ENCAPSULATION_KEY_SIZE - 32;

  for point in low_order_points() {
    let mut ciphertext = valid_ciphertext.to_bytes();
    ciphertext[x25519_offset..].copy_from_slice(&point);
    let ciphertext = XWingCiphertext::from_bytes(ciphertext);
    assert_eq!(
      XWing::decapsulate(&decapsulation_key, &ciphertext).err(),
      Some(XWingError::InvalidCiphertext),
      "{point:02x?}"
    );
    assert_eq!(
      prepared.decapsulate(&ciphertext).err(),
      Some(XWingError::InvalidCiphertext),
      "{point:02x?}"
    );
    assert_eq!(
      prepared
        .decapsulate(&valid_ciphertext)
        .expect("the key still decapsulates")
        .as_bytes()
        .as_slice(),
      vector.shared_secret
    );

    let mut encapsulation_key: [u8; XWing::ENCAPSULATION_KEY_SIZE] =
      vector.encapsulation_key.clone().try_into().expect("vector key length");
    encapsulation_key[key_x25519_offset..].copy_from_slice(&point);
    let encapsulation_key =
      XWingEncapsulationKey::try_from_slice(&encapsulation_key).expect("the ML-KEM part is valid");
    assert_eq!(
      XWing::encapsulate(&encapsulation_key, filled(&vector.randomness)).err(),
      Some(XWingError::InvalidEncapsulationKey),
      "{point:02x?}"
    );
    let prepared_encapsulation_key = encapsulation_key.prepare();
    assert_eq!(
      prepared_encapsulation_key.encapsulate(filled(&vector.randomness)).err(),
      Some(XWingError::InvalidEncapsulationKey),
      "{point:02x?}"
    );
  }
}

/// FIPS 203 section 7.2: an ML-KEM coefficient of at least q makes the key
/// invalid. Import rejects it, so no invalid key reaches encapsulation.
#[test]
fn unreduced_ml_kem_coefficient_is_rejected_at_import() {
  let mut bytes: [u8; XWing::ENCAPSULATION_KEY_SIZE] = vectors()[0]
    .encapsulation_key
    .clone()
    .try_into()
    .expect("vector key length");
  // Coefficient 0 becomes 4095, at least q = 3329.
  bytes[0] = 0xff;
  bytes[1] |= 0x0f;
  let invalid = Some(XWingError::InvalidEncapsulationKey);
  assert_eq!(XWingEncapsulationKey::try_from_slice(&bytes).err(), invalid);
  assert_eq!(XWingPreparedEncapsulationKey::try_from_slice(&bytes).err(), invalid);
}

/// Raw imports accept only the exact lengths.
#[test]
fn raw_imports_reject_wrong_lengths() {
  let bytes = [0u8; XWing::ENCAPSULATION_KEY_SIZE + 1];
  for length in [XWing::ENCAPSULATION_KEY_SIZE - 1, XWing::ENCAPSULATION_KEY_SIZE + 1] {
    assert_eq!(
      XWingEncapsulationKey::try_from_slice(&bytes[..length]).err(),
      Some(XWingError::InvalidEncapsulationKey)
    );
  }
  for length in [XWing::DECAPSULATION_KEY_SIZE - 1, XWing::DECAPSULATION_KEY_SIZE + 1] {
    assert_eq!(
      XWingDecapsulationKey::try_from_slice(&bytes[..length]).err(),
      Some(XWingError::InvalidDecapsulationKey)
    );
  }
  for length in [XWing::CIPHERTEXT_SIZE - 1, XWing::CIPHERTEXT_SIZE + 1] {
    assert_eq!(
      XWingCiphertext::try_from_slice(&bytes[..length]).err(),
      Some(XWingError::InvalidCiphertext)
    );
  }
}

/// A failed random source returns its error and no key or shared secret.
#[test]
fn random_source_failure_is_returned() {
  let failing = |_: &mut [u8]| Err(XWingError::RandomGenerationFailed);
  assert_eq!(
    XWing::generate_keypair(failing).err(),
    Some(XWingError::RandomGenerationFailed)
  );
  let encapsulation_key = XWingEncapsulationKey::try_from_slice(&vectors()[0].encapsulation_key).expect("vector key");
  assert_eq!(
    XWing::encapsulate(&encapsulation_key, failing).err(),
    Some(XWingError::RandomGenerationFailed)
  );
}
