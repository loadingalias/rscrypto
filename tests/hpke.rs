//! HPKE against its published vectors and against hostile inputs.
//!
//! The vectors are RFC 9180 Appendix A.1.1 (from the hpkewg/hpke repository),
//! the draft-ietf-hpke-hpke-06 Appendix D.2 to D.4 edge cases, and the
//! draft-ietf-hpke-pq-05 X-Wing vectors A.5 and A.12. Provenance is in
//! testdata/hpke/README.md.
#![cfg(all(
  feature = "hpke",
  feature = "x25519",
  feature = "aes-gcm",
  feature = "x-wing",
  feature = "chacha20poly1305"
))]

extern crate alloc;

use alloc::alloc::Global;

use rscrypto::{
  HpkeError, HpkeReceiver, HpkeSender, HpkeSuite, HpkeX25519HkdfSha256Aes128Gcm as X25519Suite, HpkeX25519PrivateKey,
  HpkeXWingHkdfSha256ChaCha20Poly1305 as XWingSuite, Kem, X25519PublicKey, X25519SecretKey, XWing, XWingCiphertext,
  XWingDecapsulationKey,
};
use serde_json::Value;

mod common;
use common::decode_hex_vec;

fn vectors(source: &str) -> Vec<Value> {
  serde_json::from_str::<Value>(source)
    .expect("vector JSON")
    .as_array()
    .expect("vector list")
    .clone()
}

fn field(case: &Value, name: &str) -> Vec<u8> {
  decode_hex_vec(case[name].as_str().expect("hex vector field"))
}

fn array<const N: usize>(case: &Value, name: &str) -> [u8; N] {
  field(case, name).try_into().expect("vector field length")
}

/// A random source that must be asked for exactly `bytes`.
fn filled(bytes: &[u8]) -> impl FnMut(&mut [u8]) -> Result<(), HpkeError> + '_ {
  |out| {
    out.copy_from_slice(bytes);
    Ok(())
  }
}

fn x25519_vectors() -> Vec<Value> {
  let mut cases = vectors(include_str!("../testdata/hpke/hpke-x25519-hkdfsha256-aes128gcm.json"));
  cases.extend(vectors(include_str!("../testdata/hpke/hpke-hpke-06-edge-cases.json")));
  assert_eq!(cases.len(), 4);
  cases
}

fn xwing_vector(kdf_id: u64) -> Value {
  vectors(include_str!("../testdata/hpke/hpke-pq-xwing.json"))
    .into_iter()
    .find(|case| case["kdf_id"] == kdf_id)
    .expect("X-Wing vector")
}

/// Seal every encryption in order, and check the exports, on the sender;
/// then open them in order and check the exports on the receiver.
fn assert_contexts<S: HpkeSuite>(case: &Value, sender: &mut HpkeSender<S>, receiver: &mut HpkeReceiver<S>) {
  for (sequence, encryption) in case["encryptions"].as_array().expect("encryptions").iter().enumerate() {
    let (aad, plaintext, ciphertext) = (
      field(encryption, "aad"),
      field(encryption, "pt"),
      field(encryption, "ct"),
    );
    let mut sealed = vec![0u8; ciphertext.len()];
    sender.seal(&aad, &plaintext, &mut sealed).expect("seal");
    assert_eq!(sealed, ciphertext, "sequence {sequence}: ciphertext");
    let mut opened = vec![0u8; plaintext.len()];
    receiver.open(&aad, &ciphertext, &mut opened).expect("open");
    assert_eq!(opened, plaintext, "sequence {sequence}: plaintext");
  }
  for export in case["exports"].as_array().expect("exports") {
    let expected = field(export, "exported_value");
    let context = field(export, "exporter_context");
    let mut sent = vec![0u8; expected.len()];
    let mut received = vec![0u8; expected.len()];
    sender.export(&context, &mut sent).expect("export");
    receiver.export(&context, &mut received).expect("export");
    assert_eq!(sent, expected);
    assert_eq!(received, expected);
  }
}

/// DeriveKeyPair, the sender's encapsulation, every encryption in sequence
/// order, and every export, for the X25519 suite.
#[test]
fn x25519_vectors_match() {
  for case in x25519_vectors() {
    let recipient = X25519Suite::derive_keypair(&field(&case, "ikmR"));
    assert_eq!(recipient.secret_key().as_bytes(), &array::<32>(&case, "skRm"));
    assert_eq!(recipient.public_key().as_bytes(), &array::<32>(&case, "pkRm"));
    let imported = HpkeX25519PrivateKey::new(X25519SecretKey::from_bytes(array(&case, "skRm")));
    let info = field(&case, "info");

    let (encapsulation, mut sender) =
      HpkeSender::<X25519Suite>::setup(&recipient.public_key(), &info, filled(&field(&case, "ikmE"))).expect("setup");
    assert_eq!(encapsulation.as_bytes(), &array::<32>(&case, "enc"));
    let mut receiver = HpkeReceiver::<X25519Suite>::setup(&encapsulation, &imported, &info).expect("setup");
    assert_contexts(&case, &mut sender, &mut receiver);
  }
}

/// The single-shot and in-place forms give the first encryption of the vector.
#[test]
fn x25519_single_shot_and_in_place_match() {
  let case = &x25519_vectors()[0];
  let encryption = &case["encryptions"][0];
  let (aad, plaintext, ciphertext) = (
    field(encryption, "aad"),
    field(encryption, "pt"),
    field(encryption, "ct"),
  );
  let info = field(case, "info");
  let recipient = X25519Suite::derive_keypair(&field(case, "ikmR"));

  let mut sealed = vec![0u8; ciphertext.len()];
  let encapsulation = X25519Suite::seal(
    &recipient.public_key(),
    &info,
    &aad,
    &plaintext,
    &mut sealed,
    filled(&field(case, "ikmE")),
  )
  .expect("seal");
  assert_eq!(sealed, ciphertext);
  let mut opened = vec![0u8; plaintext.len()];
  X25519Suite::open(&encapsulation, &recipient, &info, &aad, &ciphertext, &mut opened).expect("open");
  assert_eq!(opened, plaintext);

  let (_, mut sender) =
    HpkeSender::<X25519Suite>::setup(&recipient.public_key(), &info, filled(&field(case, "ikmE"))).expect("setup");
  let mut buffer = plaintext.clone();
  let tag = sender.seal_in_place_detached(&aad, &mut buffer).expect("seal");
  assert_eq!([buffer.as_slice(), tag.as_ref()].concat(), ciphertext);
  let mut receiver = HpkeReceiver::<X25519Suite>::setup(&encapsulation, &recipient, &info).expect("setup");
  receiver.open_in_place_detached(&aad, &mut buffer, &tag).expect("open");
  assert_eq!(buffer, plaintext);
}

/// The in-place forms build the same keys and contexts as the by-value
/// forms: the vector's keys, encapsulation, ciphertexts, and exports.
#[test]
fn in_place_forms_match_vectors() {
  let case = &x25519_vectors()[0];
  let recipient = X25519Suite::derive_keypair_in(&field(case, "ikmR"), Global);
  assert_eq!(recipient.secret_key().as_bytes(), &array::<32>(case, "skRm"));
  assert_eq!(recipient.public_key().as_bytes(), &array::<32>(case, "pkRm"));
  let info = field(case, "info");
  let (encapsulation, mut sender) =
    HpkeSender::<X25519Suite>::setup_in(&recipient.public_key(), &info, filled(&field(case, "ikmE")), Global)
      .expect("setup");
  assert_eq!(encapsulation.as_bytes(), &array::<32>(case, "enc"));
  let mut receiver = HpkeReceiver::<X25519Suite>::setup_in(&encapsulation, &recipient, &info, Global).expect("setup");
  assert_contexts(case, &mut sender, &mut receiver);

  let case = xwing_vector(1);
  let recipient = XWingSuite::derive_keypair_in(&field(&case, "ikmR"), Global);
  assert_eq!(recipient.decapsulation_key().as_bytes(), &array::<32>(&case, "skRm"));
  let info = field(&case, "info");
  let public_key = recipient.encapsulation_key().prepare();
  let (encapsulation, mut sender) =
    HpkeSender::<XWingSuite>::setup_in(&public_key, &info, filled(&field(&case, "ikmE")), Global).expect("setup");
  assert_eq!(encapsulation.as_bytes().as_slice(), field(&case, "enc"));
  let mut receiver = HpkeReceiver::<XWingSuite>::setup_in(&encapsulation, &recipient, &info, Global).expect("setup");
  assert_contexts(&case, &mut sender, &mut receiver);
}

/// draft-ietf-hpke-pq-05 A.5: the full schedule for the X-Wing suite.
#[test]
fn xwing_vector_matches() {
  let case = xwing_vector(1);
  let recipient = XWingSuite::derive_keypair(&field(&case, "ikmR"));
  assert_eq!(recipient.decapsulation_key().as_bytes(), &array::<32>(&case, "skRm"));
  assert_eq!(
    recipient.encapsulation_key().as_bytes().as_slice(),
    field(&case, "pkRm")
  );
  let info = field(&case, "info");

  let public_key = recipient.encapsulation_key().prepare();
  let (encapsulation, mut sender) =
    HpkeSender::<XWingSuite>::setup(&public_key, &info, filled(&field(&case, "ikmE"))).expect("setup");
  assert_eq!(encapsulation.as_bytes().as_slice(), field(&case, "enc"));
  let mut receiver = HpkeReceiver::<XWingSuite>::setup(&encapsulation, &recipient, &info).expect("setup");
  assert_contexts(&case, &mut sender, &mut receiver);
}

/// draft-ietf-hpke-pq-05 A.12 uses the SHAKE256 KDF, which rscrypto does not
/// implement, so only its KEM values apply: DeriveKeyPair, the
/// encapsulation, and the shared secret.
#[test]
fn xwing_kem_values_of_shake256_vector_match() {
  let case = xwing_vector(17);
  let recipient = XWingSuite::derive_keypair(&field(&case, "ikmR"));
  assert_eq!(recipient.decapsulation_key().as_bytes(), &array::<32>(&case, "skRm"));
  assert_eq!(
    recipient.encapsulation_key().as_bytes().as_slice(),
    field(&case, "pkRm")
  );
  let (encapsulation, shared_secret) = XWing::encapsulate(&recipient.encapsulation_key(), |out| {
    out.copy_from_slice(&field(&case, "ikmE"));
    Ok(())
  })
  .expect("encapsulation");
  assert_eq!(encapsulation.as_bytes().as_slice(), field(&case, "enc"));
  assert_eq!(shared_secret.as_bytes(), &array::<32>(&case, "shared_secret"));
}

/// RFC 9180 section 7.1.4: X25519 of a small-order point is all-zero, which
/// both directions reject. X-Wing rejects it in its own X25519 part.
#[test]
fn small_order_inputs_are_rejected() {
  let points: Vec<[u8; 32]> = include_str!("../testdata/xwing/low-order-points.txt")
    .lines()
    .map(|line| decode_hex_vec(line).try_into().expect("32-byte encoding"))
    .collect();
  assert_eq!(points.len(), 14);
  let recipient = X25519Suite::derive_keypair(&[0x11; 32]);
  let xwing_recipient = XWingSuite::derive_keypair(&[0x22; 32]);
  let (xwing_encapsulation, _) =
    HpkeSender::<XWingSuite>::setup(&xwing_recipient.encapsulation_key().prepare(), b"", filled(&[0x33; 64]))
      .expect("setup");

  for point in points {
    assert_eq!(
      HpkeSender::<X25519Suite>::setup(&X25519PublicKey::from_bytes(point), b"", filled(&[0x44; 32])).err(),
      Some(HpkeError::EncapsulationFailed),
      "{point:02x?}"
    );
    assert_eq!(
      HpkeReceiver::<X25519Suite>::setup(&X25519PublicKey::from_bytes(point), &recipient, b"").err(),
      Some(HpkeError::DecapsulationFailed),
      "{point:02x?}"
    );

    let mut key = xwing_recipient.encapsulation_key().to_bytes();
    key[XWing::ENCAPSULATION_KEY_SIZE - 32..].copy_from_slice(&point);
    let key = rscrypto::XWingPreparedEncapsulationKey::try_from_slice(&key).expect("valid ML-KEM part");
    assert_eq!(
      HpkeSender::<XWingSuite>::setup(&key, b"", filled(&[0x33; 64])).err(),
      Some(HpkeError::EncapsulationFailed),
      "{point:02x?}"
    );
    let mut encapsulation = xwing_encapsulation.to_bytes();
    encapsulation[XWing::CIPHERTEXT_SIZE - 32..].copy_from_slice(&point);
    assert_eq!(
      HpkeReceiver::<XWingSuite>::setup(&XWingCiphertext::from_bytes(encapsulation), &xwing_recipient, b"").err(),
      Some(HpkeError::DecapsulationFailed),
      "{point:02x?}"
    );
  }
}

/// A changed ciphertext, tag, aad, or info fails as one opaque error, and the
/// receiver keeps its sequence number, so the genuine message still opens.
#[test]
fn failed_open_leaves_the_receiver_usable() {
  let recipient = X25519Suite::derive_keypair(&[0x55; 32]);
  let (encapsulation, mut sender) =
    HpkeSender::<X25519Suite>::setup(&recipient.public_key(), b"info", filled(&[0x66; 32])).expect("setup");
  let mut ciphertext = [0u8; 4 + 16];
  sender.seal(b"aad", b"data", &mut ciphertext).expect("seal");

  let mut receiver = HpkeReceiver::<X25519Suite>::setup(&encapsulation, &recipient, b"info").expect("setup");
  let mut out = [0u8; 4];
  for index in [0, 3, 4, 19] {
    let mut modified = ciphertext;
    modified[index] ^= 1;
    assert_eq!(
      receiver.open(b"aad", &modified, &mut out).err(),
      Some(HpkeError::OpenFailed)
    );
  }
  assert_eq!(
    receiver.open(b"aae", &ciphertext, &mut out).err(),
    Some(HpkeError::OpenFailed)
  );
  assert_eq!(receiver.sequence(), 0);
  receiver
    .open(b"aad", &ciphertext, &mut out)
    .expect("the genuine message opens");
  assert_eq!(&out, b"data");
  assert_eq!(receiver.sequence(), 1);

  let mut wrong_info = HpkeReceiver::<X25519Suite>::setup(&encapsulation, &recipient, b"inf0").expect("setup");
  assert_eq!(
    wrong_info.open(b"aad", &ciphertext, &mut out).err(),
    Some(HpkeError::OpenFailed)
  );
}

/// Wrong buffer lengths and an oversized export fail before any state change.
#[test]
fn length_errors_change_nothing() {
  let recipient = X25519Suite::derive_keypair(&[0x77; 32]);
  let (encapsulation, mut sender) =
    HpkeSender::<X25519Suite>::setup(&recipient.public_key(), b"", filled(&[0x88; 32])).expect("setup");
  let mut receiver = HpkeReceiver::<X25519Suite>::setup(&encapsulation, &recipient, b"").expect("setup");
  assert_eq!(
    sender.seal(b"", b"data", &mut [0u8; 19]).err(),
    Some(HpkeError::InvalidLength)
  );
  assert_eq!(
    sender.seal(b"", b"data", &mut [0u8; 21]).err(),
    Some(HpkeError::InvalidLength)
  );
  assert_eq!(
    receiver.open(b"", &[0u8; 15], &mut []).err(),
    Some(HpkeError::InvalidLength)
  );
  assert_eq!(
    receiver.open(b"", &[0u8; 20], &mut [0u8; 5]).err(),
    Some(HpkeError::InvalidLength)
  );
  assert_eq!((sender.sequence(), receiver.sequence()), (0, 0));

  let mut longest = vec![0u8; 255 * 32];
  sender.export(b"", &mut longest).expect("the longest export");
  let mut too_long = vec![0u8; 255 * 32 + 1];
  assert_eq!(sender.export(b"", &mut too_long).err(), Some(HpkeError::ExportTooLong));
  assert!(
    too_long.iter().all(|&byte| byte == 0),
    "a rejected export writes nothing"
  );
}

/// A failed random source returns its error and no context.
#[test]
fn random_source_failure_is_returned() {
  let failing = |_: &mut [u8]| Err(HpkeError::RandomGenerationFailed);
  let recipient = X25519Suite::derive_keypair(&[0x99; 32]);
  assert_eq!(
    HpkeSender::<X25519Suite>::setup(&recipient.public_key(), b"", failing).err(),
    Some(HpkeError::RandomGenerationFailed)
  );
  let xwing = XWingDecapsulationKey::from_bytes([0xaa; 32])
    .encapsulation_key()
    .prepare();
  assert_eq!(
    HpkeSender::<XWingSuite>::setup(&xwing, b"", failing).err(),
    Some(HpkeError::RandomGenerationFailed)
  );
}
