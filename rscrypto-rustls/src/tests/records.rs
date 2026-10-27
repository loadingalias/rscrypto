use alloc::{boxed::Box, collections::BTreeMap, vec, vec::Vec};
use rustls::crypto::cipher::EncodableVersion;
use rustls::enums::ProtocolVersion;

use rustls::crypto::tls13::{HkdfExpander, OkmBlock};

use super::*;
use crate::{
  hash,
  tests::{fields, hex},
};

#[test]
fn rfc8448_complete_simple_handshake_symmetric_trace() {
  let mut transcript = hash::SHA256.start();
  let mut secrets: BTreeMap<Vec<u8>, Box<dyn HkdfExpander>> = BTreeMap::new();
  let mut records: [Option<(RecordKey, u64)>; 2] = [None, None];
  let (mut extracts, mut expands, mut finishes, mut encrypted_records) = (0_usize, 0_usize, 0_usize, 0_usize);

  for block in include_str!("../../tests/vectors/rfc8448.txt").split("\n\n") {
    if block.is_empty() {
      continue;
    }
    let step = block.lines().next().expect("step description");
    let v = fields(block);
    if step.contains("construct a") || step.contains("construct an") {
      // Each handshake message is added once, in the published transcript order.
      if !step.contains("NewSessionTicket") {
        assert_eq!(v.len(), 1);
        transcript.update(v.values().next().expect("handshake message"));
      }
    }
    if let Some(expected) = v.get("secret") {
      let salt = v.get("salt").map_or(&[][..], Vec::as_slice);
      let ikm = &v["IKM"];
      let tag = hash::HMAC_SHA256.with_key(salt).sign(&[ikm]);
      assert_eq!(tag.as_ref(), expected, "{step}");
      secrets.insert(expected.clone(), hash::HKDF_SHA256.extract_from_secret(Some(salt), ikm));
      extracts = extracts.strict_add(1);
    }
    if let Some(prk) = v.get("PRK") {
      let expander = secrets.get(prk).expect("PRK was produced by a previous trace step");
      if let Some(expected) = v.get("hash").filter(|value| value.len() == 32) {
        let actual = if step.contains("tls13 derived") {
          hash::SHA256.hash(b"")
        } else {
          transcript.fork_finish()
        };
        assert_eq!(actual.as_ref(), expected, "{step}");
      }
      let mut outputs = BTreeMap::new();
      for (info_name, result_name) in [
        ("info", "expanded"),
        ("key info", "key expanded"),
        ("iv info", "iv expanded"),
      ] {
        if let Some(info) = v.get(info_name) {
          let expected = &v[result_name];
          let mut actual = vec![0; expected.len()];
          let (length, label_and_context) = info.split_at(2);
          expander
            .expand_slice(&[length, label_and_context], &mut actual)
            .expect("published output bound");
          assert_eq!(&actual, expected, "{step}: {result_name}");
          outputs.insert(result_name, actual);
          expands = expands.strict_add(1);
        }
      }
      if let Some(expected) = v.get("finished") {
        let actual =
          hash::HKDF_SHA256.hmac_sign(&OkmBlock::new(&outputs["expanded"]), transcript.fork_finish().as_ref());
        assert_eq!(actual.as_ref(), expected, "{step}");
        finishes = finishes.strict_add(1);
      }
      if let Some(secret) = outputs.get("expanded") {
        secrets.insert(
          secret.clone(),
          hash::HKDF_SHA256.expander_for_okm(&OkmBlock::new(secret)),
        );
      }
      if outputs.contains_key("key expanded") {
        let index = if step.contains("{client}") || step.contains("read traffic") {
          0
        } else {
          1
        };
        records[index] = Some((
          RecordKey {
            cipher: Algorithm::Aes128.cipher(&outputs["key expanded"]),
            iv: Iv::from(<[u8; 12]>::try_from(outputs["iv expanded"].as_slice()).expect("IV length")),
          },
          0,
        ));
      }
    }
    if let Some(expected) = v.get("complete record").filter(|record| record[0] == 23) {
      let index = usize::from(step.contains("{server}"));
      let (key, sequence) = records[index].as_mut().expect("traffic key installed");
      let typ = if step.contains("application_data") {
        ContentType::ApplicationData
      } else if step.contains("alert") {
        ContentType::Alert
      } else {
        ContentType::Handshake
      };
      let plaintext = &v["payload"];
      let parts = [plaintext.as_slice()];
      let mut output = Vec::new();
      key
        .encrypt(
          Record {
            typ,
            version: EncodableVersion::Legacy(ProtocolVersion::TLSv1_3),
            payload: OutboundPlain::new(&parts),
          },
          *sequence,
          &mut output,
        )
        .expect("published record encryption");
      assert_eq!(&output, expected, "{step}");
      let mut ciphertext = expected[5..].to_vec();
      let opened = key
        .decrypt(
          Record::new(
            ContentType::ApplicationData,
            EncodableVersion::Legacy(ProtocolVersion::TLSv1_2),
            InboundOpaque(&mut ciphertext),
          ),
          *sequence,
        )
        .expect("published record decryption");
      assert_eq!(
        (opened.typ, opened.version, opened.payload),
        (
          typ,
          EncodableVersion::Legacy(ProtocolVersion::TLSv1_3),
          plaintext.as_slice()
        )
      );
      *sequence = sequence.strict_add(1);
      encrypted_records = encrypted_records.strict_add(1);
    }
  }
  assert_eq!((extracts, expands, finishes, encrypted_records), (3, 20, 2, 7));
}

#[test]
fn record_framing_fragmentation_and_failure_match_aws_lc() {
  let peer = rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER;
  for suite in crate::cipher_suite::ALL_CIPHER_SUITES {
    let ours = suite.aead_alg;
    let key = || {
      if ours.key_len() == 16 {
        AeadKey::from([0x19; 16])
      } else {
        AeadKey::from([0x19; 32])
      }
    };
    let peer = peer
      .tls13_cipher_suites
      .iter()
      .find(|s| s.common.suite == suite.common.suite)
      .expect("matching suite")
      .aead_alg;
    for seq in [0, 1, u64::MAX.strict_sub(1)] {
      for len in [0, 1, 31, 16_384] {
        let plain = vec![0xa3; len];
        let (a, b) = plain.split_at(len / 2);
        let chunks = [a, b];
        let message = || Record {
          typ: ContentType::ApplicationData,
          version: EncodableVersion::Legacy(ProtocolVersion::TLSv1_3),
          payload: OutboundPlain::new(&chunks),
        };
        let mut encrypt = ours.encrypter(key(), Iv::from([0x71; 12]));
        let mut peer = peer.encrypter(key(), Iv::from([0x71; 12]));
        let size = encrypt.encrypted_payload_len(len);
        let mut actual = vec![0xa5];
        let mut expected = Vec::new();
        encrypt.encrypt(message(), seq, &mut actual).expect("encrypt");
        peer.encrypt(message(), seq, &mut expected).expect("peer encrypt");
        assert_eq!(actual[0], 0xa5, "encryption must preserve existing output");
        assert_eq!(actual[1..], expected);
        assert_eq!(expected.len(), size.strict_add(5));
        let mut decrypt = ours.decrypter(key(), Iv::from([0x71; 12]));
        let mut payload = expected[5..].to_vec();
        let opened = decrypt
          .decrypt(
            Record::new(
              ContentType::ApplicationData,
              EncodableVersion::Legacy(ProtocolVersion::TLSv1_2),
              InboundOpaque(&mut payload),
            ),
            seq,
          )
          .expect("decrypt");
        assert_eq!(opened.payload, plain);
        assert_eq!(opened.typ, ContentType::ApplicationData);
        assert_eq!(opened.version, EncodableVersion::Legacy(ProtocolVersion::TLSv1_3));
        let mut corrupt = expected[5..].to_vec();
        *corrupt.last_mut().expect("tag") ^= 1;
        assert!(matches!(
          decrypt.decrypt(
            Record::new(
              ContentType::ApplicationData,
              EncodableVersion::Legacy(ProtocolVersion::TLSv1_2),
              InboundOpaque(&mut corrupt)
            ),
            seq
          ),
          Err(Error::DecryptError)
        ));
        assert!(corrupt[..len.strict_add(1)].iter().all(|&byte| byte == 0));
      }
    }
  }
}

#[test]
fn record_length_and_invalid_key_reject_before_releasing_plaintext() {
  let mut invalid = AES_128_GCM.encrypter(AeadKey::from([0; 32]), Iv::from([0; 12]));
  let message = || Record {
    typ: ContentType::ApplicationData,
    version: EncodableVersion::Legacy(ProtocolVersion::TLSv1_3),
    payload: (&b"plaintext"[..]).into(),
  };
  let mut out = vec![0xa5; 3];
  assert!(matches!(
    invalid.encrypt(message(), 0, &mut out),
    Err(Error::EncryptError)
  ));
  assert_eq!(out, [0xa5; 3]);
  assert!(matches!(
    AES_128_GCM.extract_keys(AeadKey::from([0; 32]), Iv::from([0; 12])),
    Err(UnsupportedOperationError)
  ));
  // An 8-byte IV yields no 12-byte nonce; the record fails before anything
  // is appended.
  let mut short_iv = AES_256_GCM.encrypter(AeadKey::from([0; 32]), Iv::new(&[0; 8]).expect("IV"));
  let mut out = vec![0xa5; 3];
  assert!(matches!(
    short_iv.encrypt(message(), 0, &mut out),
    Err(Error::EncryptError)
  ));
  assert_eq!(out, [0xa5; 3]);
  let mut encrypt = AES_256_GCM.encrypter(AeadKey::from([0; 32]), Iv::from([0; 12]));
  let large = vec![0; 16_385];
  let large_message = Record {
    typ: ContentType::ApplicationData,
    version: EncodableVersion::Legacy(ProtocolVersion::TLSv1_3),
    payload: large.as_slice().into(),
  };
  let mut out = Vec::new();
  assert!(matches!(
    encrypt.encrypt(large_message, 0, &mut out),
    Err(Error::EncryptError)
  ));
  assert!(out.is_empty());
}

#[test]
fn rfc9001_retry_integrity_tag() {
  // RFC 9001 §5.8 and Appendix A.4. Retry authenticates the original DCID
  // and Retry header as AAD with empty plaintext and the fixed IV.
  let cipher = Algorithm::Aes128.cipher(&hex("be0c690b9f66575a1d766b54e368c84e"));
  let nonce: [u8; 12] = hex("461599d35d632bf2239825bb").try_into().expect("Retry nonce");
  let aad = hex("088394c8f03e515708ff000000010008f067a5502a4262b5746f6b656e");
  assert_eq!(
    cipher.seal(nonce, &aad, &mut []).expect("Retry tag").as_slice(),
    hex("04a265ba2eff4d829058fb3f0f2496ba")
  );
}
