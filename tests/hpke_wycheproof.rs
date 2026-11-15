//! The X25519 HPKE suite against the Wycheproof HPKE vectors.
//!
//! The files are `C2SP/wycheproof` `testvectors_v1/hpke_x25519_*` at
//! `65ca4947`; provenance is in testdata/hpke/README.md. Only the groups of
//! DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, and AES-128-GCM apply.
//!
//! The `acceptable` cases are a non-canonical recipient key, a non-canonical
//! encapsulation of large order, an encapsulation on the twist, and a
//! 1024-byte `info`. rscrypto accepts each of them, as RFC 7748 and RFC 9180
//! allow, so the tests require the `valid` result for them.
#![cfg(all(feature = "hpke", feature = "x25519", feature = "aes-gcm"))]

use rscrypto::{
  HpkeError, HpkeReceiver, HpkeSender, HpkeSuite, HpkeX25519HkdfSha256Aes128Gcm as Suite, HpkeX25519PrivateKey,
  X25519PublicKey, X25519SecretKey,
};
use serde_json::Value;

mod common;
use common::decode_hex_vec;

fn field(case: &Value, name: &str) -> Vec<u8> {
  decode_hex_vec(case[name].as_str().expect("hex vector field"))
}

/// The tests of every group of this suite in `source`.
fn suite_tests(source: &str) -> Vec<Value> {
  let file: Value = serde_json::from_str(source).expect("Wycheproof JSON");
  file["testGroups"]
    .as_array()
    .expect("test groups")
    .iter()
    .filter(|group| (&group["kemId"], &group["kdfId"], &group["aeadId"]) == (&32.into(), &1.into(), &1.into()))
    .flat_map(|group| group["tests"].as_array().expect("tests").clone())
    .collect()
}

fn is_valid(case: &Value) -> bool {
  let result = case["result"].as_str().expect("result");
  assert!(
    ["valid", "acceptable", "invalid"].contains(&result),
    "unknown result {result}"
  );
  result != "invalid"
}

fn filled(bytes: &[u8]) -> impl FnMut(&mut [u8]) -> Result<(), HpkeError> + '_ {
  |out| {
    out.copy_from_slice(bytes);
    Ok(())
  }
}

/// A 32-byte X25519 encoding; the typed import rejects every other length.
fn point(bytes: &[u8]) -> Option<X25519PublicKey> {
  Some(X25519PublicKey::from_bytes(bytes.try_into().ok()?))
}

fn recipient(case: &Value) -> HpkeX25519PrivateKey {
  let secret = field(case, "skRm").try_into().expect("32-byte private key");
  let recipient = HpkeX25519PrivateKey::new(X25519SecretKey::from_bytes(secret));
  assert_eq!(recipient.public_key().as_bytes().as_slice(), field(case, "pkRm"));
  recipient
}

/// `SetupBaseS` with `ikmE` and one `Seal` give exactly `enc` and `ct`.
fn assert_sender(public_key: &X25519PublicKey, case: &Value, aad: &[u8], plaintext: &[u8]) {
  let mut ciphertext = vec![0u8; plaintext.len().strict_add(Suite::TAG_SIZE)];
  let encapsulation = Suite::seal(
    public_key,
    &field(case, "info"),
    aad,
    plaintext,
    &mut ciphertext,
    filled(&field(case, "ikmE")),
  )
  .expect("seal");
  assert_eq!(encapsulation.as_bytes().as_slice(), field(case, "enc"));
  assert_eq!(ciphertext, field(case, "ct"));
}

/// Valid recipient keys encapsulate to the vector's `enc` and `ct`; a key of
/// the wrong length does not import, and a key of small order fails setup.
#[test]
fn encapsulation_vectors() {
  let mut counts = [0usize; 2];
  for case in suite_tests(include_str!("../testdata/hpke/hpke_x25519_encap_test.json")) {
    let id = &case["tcId"];
    let valid = is_valid(&case);
    counts[usize::from(valid)] += 1;
    let Some(public_key) = point(&field(&case, "pkRm")) else {
      assert!(!valid, "tcId {id}: a wrong-length key imports");
      continue;
    };
    if valid {
      assert_sender(&public_key, &case, &field(&case, "aad"), &field(&case, "pt"));
    } else {
      assert_eq!(
        HpkeSender::<Suite>::setup(&public_key, &field(&case, "info"), filled(&field(&case, "ikmE"))).err(),
        Some(HpkeError::EncapsulationFailed),
        "tcId {id}"
      );
    }
  }
  assert_eq!(counts, [17, 2]);
}

/// Each decryption vector opens to `pt` or fails with the error that its
/// cause names, and each honest one is also reproduced by the sender.
#[test]
fn decryption_vectors() {
  let mut counts = [0usize; 2];
  for case in suite_tests(include_str!("../testdata/hpke/hpke_x25519_decrypt_test.json")) {
    let id = &case["tcId"];
    let valid = is_valid(&case);
    counts[usize::from(valid)] += 1;
    let recipient = recipient(&case);
    let (info, aad, ciphertext) = (field(&case, "info"), field(&case, "aad"), field(&case, "ct"));
    let Some(encapsulation) = point(&field(&case, "enc")) else {
      assert!(!valid, "tcId {id}: a wrong-length encapsulation imports");
      continue;
    };
    let mut receiver = match HpkeReceiver::<Suite>::setup(&encapsulation, &recipient, &info) {
      Ok(receiver) => receiver,
      Err(error) => {
        assert!(!valid, "tcId {id}: setup failed");
        assert_eq!(error, HpkeError::DecapsulationFailed, "tcId {id}");
        continue;
      }
    };
    let mut plaintext = vec![0u8; ciphertext.len().saturating_sub(Suite::TAG_SIZE)];
    let opened = receiver.open(&aad, &ciphertext, &mut plaintext);
    if valid {
      assert_eq!(opened, Ok(()), "tcId {id}");
      assert_eq!(plaintext, field(&case, "pt"), "tcId {id}");
      if case.get("ikmE").is_some() {
        assert_sender(&recipient.public_key(), &case, &aad, &plaintext);
      }
    } else {
      let expected = if ciphertext.len() < Suite::TAG_SIZE {
        HpkeError::InvalidLength
      } else {
        HpkeError::OpenFailed
      };
      assert_eq!(opened.err(), Some(expected), "tcId {id}");
      assert_eq!(receiver.sequence(), 0, "tcId {id}");
    }
  }
  assert_eq!(counts, [35, 12]);
}

/// One receiver opens each entry in order. A failed open keeps the sequence
/// number, so the next entry opens at the same one. When a case carries
/// `ikmE`, the sender reproduces every valid ciphertext in sequence order.
#[test]
fn sequence_vectors() {
  let cases = suite_tests(include_str!("../testdata/hpke/hpke_x25519_sequence_test.json"));
  assert_eq!(cases.len(), 7);
  for case in cases {
    let id = &case["tcId"];
    assert!(is_valid(&case), "tcId {id}");
    let recipient = recipient(&case);
    let info = field(&case, "info");
    let encapsulation = point(&field(&case, "enc")).expect("32-byte encapsulation");
    let mut receiver = HpkeReceiver::<Suite>::setup(&encapsulation, &recipient, &info).expect("setup");
    let mut sealed = Vec::new();
    for entry in case["decryptions"].as_array().expect("decryptions") {
      let (aad, ciphertext) = (field(entry, "aad"), field(entry, "ct"));
      let sequence = entry["sequenceNumber"].as_u64().expect("sequence number");
      assert_eq!(receiver.sequence(), sequence, "tcId {id}");
      let mut plaintext = vec![0u8; ciphertext.len().strict_sub(Suite::TAG_SIZE)];
      let opened = receiver.open(&aad, &ciphertext, &mut plaintext);
      if is_valid(entry) {
        assert_eq!(opened, Ok(()), "tcId {id}, sequence {sequence}");
        assert_eq!(plaintext, field(entry, "pt"), "tcId {id}, sequence {sequence}");
        sealed.push((sequence, aad, plaintext, ciphertext));
      } else {
        assert_eq!(
          opened.err(),
          Some(HpkeError::OpenFailed),
          "tcId {id}, sequence {sequence}"
        );
      }
    }

    if case.get("ikmE").is_none() {
      continue;
    }
    sealed.sort_by_key(|entry| entry.0);
    let (encapsulation, mut sender) =
      HpkeSender::<Suite>::setup(&recipient.public_key(), &info, filled(&field(&case, "ikmE"))).expect("setup");
    assert_eq!(encapsulation.as_bytes().as_slice(), field(&case, "enc"), "tcId {id}");
    for (expected_sequence, (sequence, aad, plaintext, ciphertext)) in (0..).zip(sealed) {
      assert_eq!(
        sequence, expected_sequence,
        "tcId {id}: valid entries cover every sequence number"
      );
      assert_eq!(
        sender.seal_to_vec(&aad, &plaintext).expect("seal"),
        ciphertext,
        "tcId {id}"
      );
    }
  }
}
