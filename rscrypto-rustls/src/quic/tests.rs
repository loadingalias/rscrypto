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

/// Fills each request with the bytes of `0`, repeated from the start of every request.
#[derive(Debug)]
struct Bytes(&'static [u8]);

impl SecureRandom for Bytes {
  fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
    for (byte, source) in output.iter_mut().zip(self.0.iter().cycle()) {
      *byte = *source;
    }
    Ok(())
  }
}

/// A source whose every request fails.
#[derive(Debug)]
struct Failing;

impl SecureRandom for Failing {
  fn fill(&self, _output: &mut [u8]) -> Result<(), GetRandomFailed> {
    Err(GetRandomFailed)
  }
}

fn load_balancer(rotation: u8, server_id_len: usize, nonce_len: usize, first_octet: FirstOctet) -> LoadBalancerConfig {
  let key = LoadBalancerKey::from_bytes(hex("8f95f09245765f80256934e50c66207f").try_into().expect("key"));
  LoadBalancerConfig::new(
    RotationId::new(rotation).expect("rotation"),
    server_id_len,
    nonce_len,
    first_octet,
    key,
  )
  .expect("allowed lengths")
}

/// Every encrypted CID of draft-ietf-quic-load-balancers-22 Appendix B.2 and the section 5.4.2.4
/// example, with the first octet that encodes the rotation and the length (section 3.4).
///
/// Appendix B.1 is the no-key form, which this provider does not implement. Erratum: its second
/// row is malformed, with a nine-digit nonce and a CID that does not match its fields.
#[test]
fn load_balancer_ids_match_the_draft_vectors() {
  const B2_KEY: &[u8] = &[
    0x8f, 0x95, 0xf0, 0x92, 0x45, 0x76, 0x5f, 0x80, 0x25, 0x69, 0x34, 0xe5, 0x0c, 0x66, 0x20, 0x7f,
  ];
  const EXAMPLE_KEY: &[u8] = &[
    0xfd, 0xf7, 0x26, 0xa9, 0x89, 0x3e, 0xc0, 0x5c, 0x06, 0x32, 0xd3, 0x95, 0x66, 0x80, 0xba, 0xf0,
  ];
  let vectors: [(&'static [u8], u8, &str, &str, &str); 5] = [
    (EXAMPLE_KEY, 0, "31441a", "9c69c275", "0767947d29be054a"),
    (B2_KEY, 0, "ed793a", "ee080dbf", "0720b1d07b359d3c"),
    (
      B2_KEY,
      1,
      "ed793a51d49b8f5fab65",
      "ee080dbf48",
      "2fcc381bc74cb4fbad2823a3d1f8fed2",
    ),
    (
      B2_KEY,
      2,
      "ed793a51d49b8f5f",
      "ee080dbf48c0d1e5",
      "504dd2d05a7b0de9b2b9907afb5ecf8cc3",
    ),
    // Erratum: B.2 lists cr_bits 3 for this row, but its CID begins 0x12, which encodes
    // rotation 0 and length 18 (section 3.4). google/quiche's tests use configuration 0 too.
    (
      B2_KEY,
      0,
      "ed793a51d49b8f5fab",
      "ee080dbf48c0d1e55d",
      "125779c9cc86beb3a3a4a3ca96fce4bfe0cdbc",
    ),
  ];
  for (key, rotation, server_id, nonce, id) in vectors {
    let (server_id, nonce, id) = (hex(server_id), hex(nonce), hex(id));
    // A generated key is the source's 16 bytes in order.
    let key = LoadBalancerKey::generate(&Bytes(key)).expect("fixed key");
    let config = LoadBalancerConfig::new(
      RotationId::new(rotation).expect("rotation"),
      server_id.len(),
      nonce.len(),
      FirstOctet::Length,
      key,
    )
    .expect("draft lengths");
    let issued = config
      .encode(&server_id, &nonce, &Failing)
      .expect("Length mode draws nothing");
    assert_eq!(issued.as_slice(), id, "encode {}+{}", server_id.len(), nonce.len());
    assert_eq!(config.connection_id_len(), id.len());
    let decoded = config.decode(&id).expect("routable");
    assert_eq!(
      decoded.as_slice(),
      server_id,
      "decode {}+{}",
      server_id.len(),
      nonce.len()
    );
  }
}

/// Decoding refuses the IDs that section 4.1 calls unroutable for a configuration, and ignores
/// server-use octets after the encrypted block.
#[test]
fn load_balancer_decode_refuses_unroutable_ids() {
  let config = load_balancer(5, 3, 4, FirstOctet::Length);
  let id = config.encode(&[1, 2, 3], &[4, 5, 6, 7], &Failing).expect("encode");
  let id = id.as_slice();
  assert_eq!(id[0] >> 5, 5, "rotation bits");
  let server_id = config.decode(id).expect("routable");
  assert_eq!(server_id.as_slice(), [1, 2, 3]);

  let mut longer = id.to_vec();
  longer.extend_from_slice(&[0xee; 12]);
  assert_eq!(config.decode(&longer), Some(server_id), "server-use octets");

  for rotation in [0b000, 0b100, 0b110, 0b111] {
    let mut other = id.to_vec();
    other[0] = (rotation << 5) | (other[0] & 0x1f);
    assert_eq!(config.decode(&other), None, "rotation {rotation:#05b}");
  }
  assert_eq!(config.decode(&id[..id.len() - 1]), None, "short ID");
  assert_eq!(config.decode(&[]), None, "empty ID");
  assert_eq!(RotationId::new(0b111), None, "reserved rotation");
}

/// The first octet carries the rotation in its three high bits and the configured low bits.
#[test]
fn load_balancer_first_octet_carries_rotation_and_low_bits() {
  let lengths = load_balancer(6, 2, 6, FirstOctet::Length);
  assert_eq!(
    lengths.encode(&[1, 2], &[3; 6], &Failing).expect("encode").as_slice()[0],
    0b110_01000
  );

  let random = load_balancer(6, 2, 6, FirstOctet::Random);
  for (draw, expected) in [(0xff, 0b110_11111), (0x00, 0b110_00000), (0xa5, 0b110_00101)] {
    let source = Fixed(draw);
    let id = random.encode(&[1, 2], &[3; 6], &source).expect("encode");
    assert_eq!(id.as_slice()[0], expected, "draw {draw:#04x}");
    assert_eq!(
      id.as_slice()[1..],
      lengths.encode(&[1, 2], &[3; 6], &Failing).expect("encode").as_slice()[1..]
    );
  }
  assert!(matches!(
    random.encode(&[1, 2], &[3; 6], &Failing),
    Err(Error::FailedToGetRandomBytes)
  ));
}

/// Encoding refuses a server ID or nonce of the wrong length before it draws entropy, so a short
/// nonce never pads silently into a repeated one.
#[test]
fn load_balancer_encode_refuses_wrong_lengths() {
  let config = load_balancer(0, 2, 6, FirstOctet::Random);
  let source = Source::default();
  for (server_id, nonce) in [
    (&[1u8][..], &[0u8; 6][..]),
    (&[1, 2, 3], &[0; 6]),
    (&[1, 2], &[0; 5]),
    (&[1, 2], &[0; 7]),
  ] {
    assert!(matches!(
      config.encode(server_id, nonce, &source),
      Err(Error::EncryptError)
    ));
  }
  assert!(source.draws().is_empty(), "no entropy before the length check");
  assert!(
    LoadBalancerConfig::new(
      RotationId::new(0).expect("rotation"),
      1,
      3,
      FirstOctet::Length,
      LoadBalancerKey::from_bytes([0; 16])
    )
    .is_none()
  );
  LoadBalancerKey::generate(&Failing).expect_err("failed source");
  assert_eq!(
    format!("{config:?}"),
    "LoadBalancerConfig { rotation: RotationId(0), server_id_len: 2, nonce_len: 6, first_octet: Random, .. }"
  );
  assert_eq!(
    format!("{:?}", LoadBalancerKey::from_bytes([0x42; 16])),
    "LoadBalancerKey { .. }"
  );
}

#[test]
fn endpoint_keys_can_be_shared_across_threads() {
  fn shareable<T: Send + Sync + 'static>() {}
  shareable::<LoadBalancerConfig>();
  shareable::<StatelessResetKey>();
  shareable::<AddressTokenKey>();
}

/// Every allowed length pair issues the connection ID of an independent implementation:
/// google/quiche for 117 pairs, and hopf-quic for the 3 pairs with a nonce of more than 16 octets,
/// which quiche refuses. The lines step through rotations 0 to 6. Decode recovers each server ID,
/// and it ignores the server-use octets after the encrypted block up to the 20-octet QUIC limit.
/// The provenance is in `tests/vectors/quic_lb/README.md`.
#[test]
fn load_balancer_ids_match_independent_implementations() {
  let files = [
    include_str!("../../tests/vectors/quic_lb/quiche-vectors.txt"),
    include_str!("../../tests/vectors/quic_lb/hopf-vectors.txt"),
  ];
  let mut pairs = alloc::collections::BTreeSet::new();
  for line in files.iter().flat_map(|file| file.lines()) {
    let fields: Vec<&str> = line.split(' ').collect();
    let [rotation, key, server_id, nonce, id] = fields[..] else {
      panic!("five fields: {line}")
    };
    let (server_id, nonce, id) = (hex(server_id), hex(nonce), hex(id));
    let config = LoadBalancerConfig::new(
      RotationId::new(rotation.parse().expect("rotation")).expect("rotation"),
      server_id.len(),
      nonce.len(),
      FirstOctet::Length,
      LoadBalancerKey::from_bytes(hex(key).try_into().expect("key")),
    )
    .expect("allowed lengths");
    let issued = config.encode(&server_id, &nonce, &Failing).expect("Length mode draws nothing");
    assert_eq!(issued.as_slice(), id, "encode {line}");

    let mut longer = id.clone();
    longer.resize(20, 0xee);
    for received in [&id, &longer] {
      let decoded = config.decode(received).expect("routable");
      assert_eq!(decoded.as_slice(), server_id, "decode {} octets, {line}", received.len());
    }
    pairs.insert((server_id.len(), nonce.len()));
  }
  let allowed = (1..=19_usize)
    .flat_map(|server_id_len| (4..=19_usize.strict_sub(server_id_len)).map(move |nonce_len| (server_id_len, nonce_len)));
  assert!(pairs.into_iter().eq(allowed), "every allowed length pair");
}
