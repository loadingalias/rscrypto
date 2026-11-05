use super::*;
use crate::tests::{Fixed, fields, hex};
use alloc::{format, vec, vec::Vec};
use rustls::Error;
use rustls::crypto::cipher::{AeadKey as RustlsAeadKey, Iv};
use rustls::crypto::tls13::OkmBlock;
use rustls::crypto::{GetRandomFailed, SecureRandom};
use rustls::quic::Side;
use std::sync::Mutex;

/// Counts bytes 0, 1, 2, ... across requests, records each request, and fails request `fail_at`.
#[derive(Debug, Default)]
struct Source {
  state: Mutex<(u8, Vec<Vec<u8>>)>,
  fail_at: Option<usize>,
}

impl Source {
  fn failing_at(fail_at: usize) -> Self {
    Self {
      fail_at: Some(fail_at),
      ..Self::default()
    }
  }

  fn draws(&self) -> Vec<Vec<u8>> {
    self.state.lock().expect("draws").1.clone()
  }

  fn last(&self) -> Vec<u8> {
    self.draws().pop().expect("a draw")
  }
}

impl SecureRandom for Source {
  fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
    let mut state = self.state.lock().expect("draws");
    for byte in output.iter_mut() {
      *byte = state.0;
      state.0 = state.0.wrapping_add(1);
    }
    let failed = self.fail_at == Some(state.1.len());
    state.1.push(output.to_vec());
    if failed { Err(GetRandomFailed) } else { Ok(()) }
  }
}

fn lengths(source: &Source) -> Vec<usize> {
  source.draws().iter().map(Vec::len).collect()
}

/// Apply or remove header protection at `offset`, sampling four bytes past it (RFC 9001, section 5.4.2).
fn header_protection(key: &dyn rustls::quic::HeaderProtectionKey, packet: &mut [u8], offset: usize, protect: bool) {
  let sample: [u8; 16] = packet[offset.strict_add(4)..offset.strict_add(20)]
    .try_into()
    .expect("sample");
  let (first, rest) = packet.split_at_mut(offset);
  let number = &mut rest[..4];
  if protect {
    key.encrypt_in_place(&sample, &mut first[0], number)
  } else {
    key.decrypt_in_place(&sample, &mut first[0], number)
  }
  .expect("header protection");
}

#[test]
fn rfc9001_initial_packets_through_initial_keys_for_each_side() {
  let cid = hex("8394c8f03e515708");
  for (fixture, side, peer, number, length) in [
    (
      include_str!("../../tests/vectors/rfc9001-client.txt"),
      Side::Client,
      Side::Server,
      2,
      1162,
    ),
    (
      include_str!("../../tests/vectors/rfc9001-server.txt"),
      Side::Server,
      Side::Client,
      1,
      99,
    ),
  ] {
    let vector = fields(fixture);
    let header = &vector["header"];
    let offset = header.len().strict_sub(usize::from(header[0] & 3).strict_add(1));
    let mut plaintext = vector["frame"].clone();
    plaintext.resize(length, 0);

    let local = initial_keys(&cid, side).local;
    let mut packet = [header.as_slice(), plaintext.as_slice()].concat();
    let tag = local
      .packet
      .encrypt_in_place(number, header, &mut packet[header.len()..], None)
      .expect("RFC Initial");
    packet.extend_from_slice(tag.as_ref());
    header_protection(&*local.header, &mut packet, offset, true);
    assert_eq!(packet, vector["packet"], "RFC 9001 Appendix A protected packet");

    // The peer's remote keys open what this side's local keys sealed.
    let remote = initial_keys(&cid, peer).remote;
    header_protection(&*remote.header, &mut packet, offset, false);
    assert_eq!(&packet[..header.len()], header.as_slice());
    let opened = remote
      .packet
      .decrypt_in_place(number, header, &mut packet[header.len()..], None)
      .expect("peer opens the RFC Initial");
    assert_eq!(opened, plaintext.as_slice());
    assert_eq!(remote.packet.confidentiality_limit(), 1 << 23);
    assert_eq!(remote.packet.integrity_limit(), 1 << 52);

    // The same side's remote keys belong to the other direction.
    let mut packet = vector["packet"].clone();
    let wrong = initial_keys(&cid, side).remote;
    header_protection(&*wrong.header, &mut packet, offset, false);
    assert_ne!(&packet[..header.len()], header.as_slice());
  }
}

#[test]
fn rfc9001_retry_integrity_and_independent_tags_for_every_connection_id_length() {
  let cid = hex("8394c8f03e515708");
  let packet = hex("ff000000010008f067a5502a4262b5746f6b656e04a265ba2eff4d829058fb3f0f2496ba");
  let split = packet.len().strict_sub(RETRY_INTEGRITY_TAG_LEN);
  assert_eq!(
    retry_integrity_tag(&cid, &packet[..split])
      .expect("8-byte ID")
      .as_slice(),
    &packet[split..]
  );
  assert!(is_valid_retry(&cid, &packet));
  for index in 0..packet.len() {
    let mut corrupted = packet.clone();
    corrupted[index] ^= 1;
    assert!(!is_valid_retry(&cid, &corrupted));
  }
  assert!(!is_valid_retry(&[0; 8], &packet));
  assert!(!is_valid_retry(&cid[..7], &packet));
  for length in 0..RETRY_INTEGRITY_TAG_LEN {
    assert!(!is_valid_retry(&cid, &packet[..length]));
  }

  // AWS-LC computes the RFC 9001 section 5.8 tag over the pseudo-packet
  // independently, with the published key and nonce at packet number zero.
  let oracle = rustls_aws_lc_rs::cipher_suite::TLS13_AES_128_GCM_SHA256
    .quic
    .expect("QUIC AEAD")
    .packet_key(
      RustlsAeadKey::from(<[u8; 16]>::try_from(hex("be0c690b9f66575a1d766b54e368c84e")).expect("key")),
      Iv::from(<[u8; 12]>::try_from(hex("461599d35d632bf2239825bb")).expect("nonce")),
    );
  let retry = &packet[..split];
  for length in 0..=MAX_CONNECTION_ID_LEN {
    let original: Vec<u8> = (0..length)
      .map(|byte| u8::try_from(byte).expect("short").wrapping_mul(37))
      .collect();
    let pseudo = [&[u8::try_from(length).expect("short")], original.as_slice(), retry].concat();
    let expected = oracle
      .encrypt_in_place(0, &pseudo, &mut [], None)
      .expect("independent tag");
    let tag = retry_integrity_tag(&original, retry).expect("valid ID length");
    assert_eq!(tag.as_slice(), expected.as_ref());
    assert!(is_valid_retry(&original, &[retry, &tag].concat()));
  }
  let long = [0; MAX_CONNECTION_ID_LEN + 1];
  assert!(retry_integrity_tag(&long, retry).is_none());
  let tag = retry_integrity_tag(&long[..MAX_CONNECTION_ID_LEN], retry).expect("20 bytes");
  assert!(!is_valid_retry(&long, &[retry, &tag].concat()));
}

#[test]
fn stateless_reset_tokens_are_truncated_independent_hmac() {
  let source = Source::default();
  let key = StatelessResetKey::generate(&source).expect("reset key");
  assert_eq!(lengths(&source), [32]);
  let root: [u8; 32] = source.last().try_into().expect("key bytes");
  let peer = rustls_aws_lc_rs::cipher_suite::TLS13_AES_128_GCM_SHA256.hkdf_provider;
  let mut seen = alloc::collections::BTreeSet::new();
  for length in 0..=MAX_CONNECTION_ID_LEN {
    let id: Vec<u8> = (0..length)
      .map(|byte| u8::try_from(byte).expect("short") ^ 0xa5)
      .collect();
    let expected = peer.hmac_sign(&OkmBlock::new(&root), &id);
    let token = key.token(&id);
    assert_eq!(token.as_slice(), &expected.as_ref()[..StatelessResetKey::TOKEN_LEN]);
    assert!(seen.insert(token), "distinct IDs get distinct tokens");
  }
  assert_eq!(format!("{key:?}"), "StatelessResetKey { .. }");

  let failing = Source::failing_at(0);
  StatelessResetKey::generate(&failing).expect_err("failed source");
  assert_eq!(lengths(&failing), [32]);
}

#[test]
fn address_tokens_match_independent_peer_and_fail_closed() {
  const AAD: &[u8] = b"203.0.113.7:4433";
  let source = Source::default();
  let key = AddressTokenKey::generate(&source).expect("token key");
  let other = AddressTokenKey::generate(&Fixed(0x42)).expect("token key");
  let root = source.last();
  let hkdf = rustls_aws_lc_rs::cipher_suite::TLS13_AES_128_GCM_SHA256.hkdf_provider;
  let aead = rustls_aws_lc_rs::cipher_suite::TLS13_AES_256_GCM_SHA384
    .quic
    .expect("QUIC AEAD");
  let mut tokens = alloc::collections::BTreeSet::new();
  for length in [0, 1, 37, AddressTokenKey::MAX_PLAINTEXT_LEN] {
    let plaintext = vec![0x57; length];
    let mut token = plaintext.clone();
    key.seal(&source, &mut token, AAD).expect("seal");
    let nonce = source.last();
    assert_eq!(nonce.len(), AddressTokenKey::NONCE_LEN);
    assert_eq!(token.len(), length.strict_add(AddressTokenKey::OVERHEAD));
    assert_eq!(&token[token.len().strict_sub(16)..], nonce.as_slice());

    // AWS-LC derives the one-use key with HKDF-SHA256 and seals with AES-256-GCM.
    let mut derived = [0; 32];
    hkdf
      .expander_for_okm(&OkmBlock::new(&root))
      .expand_slice(&[b"rscrypto-rustls QUIC token v1", &nonce], &mut derived)
      .expect("independent HKDF");
    let mut expected = plaintext.clone();
    let tag = aead
      .packet_key(RustlsAeadKey::from(derived), Iv::from([0; 12]))
      .encrypt_in_place(0, AAD, &mut expected, None)
      .expect("independent AEAD");
    expected.extend_from_slice(tag.as_ref());
    assert_eq!(&token[..length.strict_add(16)], expected.as_slice());
    assert!(tokens.insert(token.clone()), "a fresh nonce for every token");

    let mut opened = token.clone();
    assert_eq!(key.open(&mut opened, AAD).expect("open"), plaintext.as_slice());
    let positions: Vec<usize> = if length < 64 {
      (0..token.len()).collect()
    } else {
      vec![0, length, length.strict_add(16), token.len().strict_sub(1)]
    };
    for position in positions {
      let mut corrupt = token.clone();
      corrupt[position] ^= 1;
      assert!(matches!(key.open(&mut corrupt, AAD), Err(Error::DecryptError)));
      assert!(
        corrupt[..length].iter().all(|&byte| byte == 0),
        "rejected plaintext is cleared"
      );
    }
    assert!(matches!(
      key.open(&mut token.clone(), b"198.51.100.1:4433"),
      Err(Error::DecryptError)
    ));
    assert!(matches!(other.open(&mut token.clone(), AAD), Err(Error::DecryptError)));
  }

  // Bounds fail before entropy is drawn or any byte changes.
  let draws = source.draws().len();
  let mut oversized = vec![0x33; AddressTokenKey::MAX_PLAINTEXT_LEN.strict_add(1)];
  assert!(matches!(
    key.seal(&source, &mut oversized, AAD),
    Err(Error::EncryptError)
  ));
  assert_eq!(oversized, vec![0x33; AddressTokenKey::MAX_PLAINTEXT_LEN.strict_add(1)]);
  assert_eq!(source.draws().len(), draws);
  for length in (0..AddressTokenKey::OVERHEAD).chain([AddressTokenKey::MAX_PLAINTEXT_LEN
    .strict_add(AddressTokenKey::OVERHEAD)
    .strict_add(1)])
  {
    let mut token = vec![0x44; length];
    assert!(matches!(key.open(&mut token, AAD), Err(Error::DecryptError)));
    assert!(
      token.iter().all(|&byte| byte == 0x44),
      "length rejection leaves the token unchanged"
    );
  }

  // Entropy failure: no token, no fallback, and the plaintext is unchanged.
  let failing = Source::failing_at(1);
  let key = AddressTokenKey::generate(&failing).expect("first draw succeeds");
  let mut data = vec![0x66; 9];
  assert!(matches!(
    key.seal(&failing, &mut data, AAD),
    Err(Error::FailedToGetRandomBytes)
  ));
  assert_eq!(data, vec![0x66; 9]);
  assert_eq!(lengths(&failing), [32, 16]);
  AddressTokenKey::generate(&Source::failing_at(0)).expect_err("failed source");
  assert_eq!(format!("{key:?}"), "AddressTokenKey { .. }");
}

#[test]
fn connection_ids_carry_independent_aes_tags_and_validate_only_under_their_key() {
  use aes::cipher::{Array, BlockCipherEncrypt as _, KeyInit as _};

  let source = Source::default();
  let key = ConnectionIdKey::generate(&source).expect("ID key");
  let tag_key: [u8; 16] = source.last().try_into().expect("key bytes");
  let other = ConnectionIdKey::generate(&crate::tests::RANDOM).expect("ID key");
  // RustCrypto AES-128 computes each tag block independently.
  let oracle = aes::Aes128::new(&Array::from(tag_key));
  let mut seen = alloc::collections::BTreeSet::new();
  for _ in 0..128 {
    let id = key.issue(&source).expect("issue");
    assert_eq!(id.len(), ConnectionIdKey::ID_LEN);
    assert_eq!(id[..3], source.last(), "the nonce is the caller's draw");
    let mut block = [0u8; 16];
    block[..3].copy_from_slice(&id[..3]);
    block[3..].copy_from_slice(b"rscrypto-cid1");
    let mut block = Array::from(block);
    oracle.encrypt_block(&mut block);
    assert_eq!(id[3..], block[..5], "the tag is truncated AES-128 under the key");
    assert!(key.validate(&id));
    assert!(!other.validate(&id), "different key");
    for bit in 0..64 {
      let mut forged = id;
      forged[bit / 8] ^= 1 << (bit % 8);
      assert!(!key.validate(&forged), "altered ID");
    }
    seen.insert(id);
  }
  assert_eq!(seen.len(), 128, "distinct nonces give distinct IDs");
  for length in [0, 7, 9, 20] {
    assert!(!key.validate(&vec![0; length]), "foreign length");
  }
  assert_eq!(lengths(&source)[..2], [16, 3]);
  assert_eq!(format!("{key:?}"), "ConnectionIdKey { .. }");

  let failing = Source::failing_at(1);
  let key = ConnectionIdKey::generate(&failing).expect("first draw succeeds");
  key.issue(&failing).expect_err("failed source");
  ConnectionIdKey::generate(&Source::failing_at(0)).expect_err("failed source");
}

#[test]
fn endpoint_keys_can_be_shared_across_threads() {
  fn shareable<T: Send + Sync + 'static>() {}
  shareable::<ConnectionIdKey>();
  shareable::<StatelessResetKey>();
  shareable::<AddressTokenKey>();
}
