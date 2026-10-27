use super::*;

#[cfg(all(feature = "x25519", feature = "ecdsa-p256"))]
mod network;
#[cfg(all(feature = "x25519", feature = "ecdsa-p256"))]
mod session;
use crate::tests::{fields, hex};
use bytes::BytesMut;

/// Records each request length and fails the request numbered `fail_at`.
#[derive(Debug)]
struct Recording {
  lengths: std::sync::Mutex<Vec<usize>>,
  fail_at: Option<usize>,
}

impl Recording {
  const fn new(fail_at: Option<usize>) -> Self {
    Self {
      lengths: std::sync::Mutex::new(Vec::new()),
      fail_at,
    }
  }

  fn lengths(&self) -> Vec<usize> {
    self.lengths.lock().expect("recorded lengths").clone()
  }
}

impl rustls::crypto::SecureRandom for Recording {
  fn fill(&self, output: &mut [u8]) -> Result<(), rustls::crypto::GetRandomFailed> {
    let mut lengths = self.lengths.lock().expect("recorded lengths");
    output.fill(0x42);
    let failed = self.fail_at == Some(lengths.len());
    lengths.push(output.len());
    if failed {
      Err(rustls::crypto::GetRandomFailed)
    } else {
      Ok(())
    }
  }
}

#[test]
fn endpoint_construction_draws_every_key_from_the_caller_and_fails_closed() {
  // Reset key, connection-ID tag key, connection-ID generator seed, endpoint seed.
  const DRAWS: [usize; 4] = [32, 16, 32, 32];
  let source = Recording::new(None);
  let keys = EndpointKeys::generate(&source).expect("endpoint keys");
  assert_eq!(source.lengths(), DRAWS[..3]);
  drop(keys.endpoint_config(&source).expect("endpoint configuration"));
  assert_eq!(source.lengths(), DRAWS);
  for fail_at in 0..DRAWS.len() {
    let source = Recording::new(Some(fail_at));
    assert!(matches!(
      endpoint_config_with_random(&source),
      Err(Error::FailedToGetRandomBytes)
    ));
    assert_eq!(source.lengths(), DRAWS[..=fail_at], "no draw after a failure");
  }
}

#[test]
fn connection_ids_carry_an_independent_aes_tag_and_validate_only_under_their_keys() {
  use aes::cipher::{Array, BlockCipherEncrypt as _, KeyInit as _};
  use quinn_proto::ConnectionIdGenerator;

  let keys = Arc::new(cid::CidKeys::generate(&Recording::new(None)).expect("CID keys"));
  let other = Arc::new(cid::CidKeys::generate(&crate::tests::RANDOM).expect("CID keys"));
  let (mut first, mut second) = (keys.generator(), keys.generator());
  // The recording source fills 0x42, so the tag key is [0x42; 16]. RustCrypto
  // AES-128 computes the tag block independently.
  let oracle = aes::Aes128::new(&Array::from([0x42; 16]));
  let mut seen = alloc::collections::BTreeSet::new();
  for round in 0..128 {
    let id = if round % 2 == 0 {
      first.generate_cid()
    } else {
      second.generate_cid()
    };
    assert_eq!(id.len(), 8);
    let mut block = [0u8; 16];
    block[..3].copy_from_slice(&id[..3]);
    block[3..].copy_from_slice(b"rscrypto-cid1");
    let mut block = Array::from(block);
    oracle.encrypt_block(&mut block);
    assert_eq!(id[3..], block[..5], "the tag is truncated AES-128 under the tag key");
    first.validate(id).expect("issued by this key set");
    second.validate(id).expect("same key set");
    other.generator().validate(id).expect_err("different key set");
    for bit in 0..64 {
      let mut forged = [0u8; 8];
      forged.copy_from_slice(&id);
      forged[bit / 8] ^= 1 << (bit % 8);
      first.validate(ConnectionId::new(&forged)).expect_err("altered ID");
    }
    seen.insert(id.to_vec());
  }
  assert!(seen.len() > 120, "endpoint generators draw independent nonces");
  for length in [0, 7, 9, 20] {
    first
      .validate(ConnectionId::new(&vec![0; length]))
      .expect_err("foreign length");
  }
}

#[cfg(all(feature = "ecdsa-p256", feature = "x25519"))]
#[test]
fn initial_destination_ids_come_from_the_client_entropy_source() {
  #[derive(Debug)]
  struct Zero;
  impl rustls::crypto::SecureRandom for Zero {
    fn fill(&self, output: &mut [u8]) -> Result<(), rustls::crypto::GetRandomFailed> {
      output.fill(0);
      Ok(())
    }
  }
  static ZERO: Zero = Zero;
  static FAILING: Recording = Recording::new(Some(0));

  let mut provider = crate::tests::tls::provider();
  provider.secure_random = &ZERO;
  let (client, _) = crate::tests::tls::configs(
    provider.clone(),
    crate::tests::tls::provider(),
    crate::tests::tls::Credential::P256,
    false,
  );
  let mut endpoint = quinn_proto::Endpoint::new(
    Arc::new(endpoint_config_with_random(&crate::tests::RANDOM).expect("endpoint")),
    None,
    false,
  );
  let now = std::time::Instant::now();
  let (_, mut connection) = endpoint
    .connect(
      now,
      client_config(client).expect("client configuration"),
      "127.0.0.1:4433".parse().expect("address"),
      "localhost",
    )
    .expect("connect");
  let mut datagram = Vec::new();
  let transmit = connection.poll_transmit(now, 1, &mut datagram).expect("Initial packet");
  assert_eq!(transmit.size, datagram.len());
  // Long header: flags, version, then the destination connection ID length and value.
  assert_eq!(datagram[5], 20);
  // An all-zero seed makes the first ID the start of RFC 8439 Appendix A.1 test vector 1.
  assert_eq!(datagram[6..26], hex("76b8e0ada0f13d90405d6ae55386bd28bdd219b8"));

  provider.secure_random = &FAILING;
  let (client, _) = crate::tests::tls::configs(
    provider,
    crate::tests::tls::provider(),
    crate::tests::tls::Credential::P256,
    false,
  );
  assert!(matches!(client_config(client), Err(Error::FailedToGetRandomBytes)));
  assert_eq!(FAILING.lengths(), [32]);
}

#[cfg(all(feature = "ecdsa-p256", feature = "x25519"))]
#[test]
fn server_tokens_use_the_selected_tls_entropy_source() {
  use core::sync::atomic::{AtomicUsize, Ordering};
  use rustls::crypto::{GetRandomFailed, SecureRandom};

  #[derive(Debug)]
  struct Source(AtomicUsize);
  impl SecureRandom for Source {
    fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
      self.0.fetch_add(1, Ordering::Relaxed);
      output[..17].fill(0x42);
      Err(GetRandomFailed)
    }
  }
  static SOURCE: Source = Source(AtomicUsize::new(0));
  let mut provider = crate::tests::tls::provider();
  provider.secure_random = &SOURCE;
  let (_, server) = crate::tests::tls::configs(
    crate::tests::tls::provider(),
    provider,
    crate::tests::tls::Credential::P256,
    false,
  );
  assert_eq!(SOURCE.0.load(Ordering::Relaxed), 0);
  assert!(matches!(server_config(server), Err(Error::FailedToGetRandomBytes)));
  assert_eq!(SOURCE.0.load(Ordering::Relaxed), 1);
}

#[cfg(all(feature = "ecdsa-p256", feature = "x25519"))]
#[test]
fn unsupported_quic_versions_never_derive_initial_keys() {
  let (_, server) = crate::tests::tls::configs(
    crate::tests::tls::provider(),
    crate::tests::tls::provider(),
    crate::tests::tls::Credential::P256,
    false,
  );
  let server = server_config(server).expect("server configuration");
  let cid = ConnectionId::new(&[1; 8]);
  for version in [0, 2, 0x6b33_43cf, 0xff00_001d, 0xff00_0021] {
    assert!(server.crypto.initial_keys(version, cid).is_err());
  }
  drop(server.crypto.initial_keys(1, cid).expect("QUIC v1"));
}

#[test]
fn rfc9001_initial_packets_through_quinn_traits() {
  let cid = ConnectionId::new(&hex("8394c8f03e515708"));
  for (fixture, side, number, length) in [
    (
      include_str!("../../tests/vectors/rfc9001-client.txt"),
      Side::Client,
      2,
      1162,
    ),
    (
      include_str!("../../tests/vectors/rfc9001-server.txt"),
      Side::Server,
      1,
      99,
    ),
  ] {
    let vector = fields(fixture);
    let header = &vector["header"];
    let offset = header.len().strict_sub(usize::from(header[0] & 3).strict_add(1));
    let mut plaintext = vector["frame"].clone();
    plaintext.resize(length, 0);
    let mut packet = [header.as_slice(), plaintext.as_slice(), &[0; 16]].concat();
    let keys = initial_keys(cid, side);
    keys.packet.local.encrypt(number, &mut packet, header.len());
    keys.header.local.encrypt(offset, &mut packet);
    assert_eq!(packet, vector["packet"]);
    keys.header.local.decrypt(offset, &mut packet);
    assert_eq!(&packet[..header.len()], header);
    let mut payload = BytesMut::from(&packet[header.len()..]);
    keys
      .packet
      .local
      .decrypt(number, header, &mut payload)
      .expect("RFC Initial");
    assert_eq!(payload.as_ref(), plaintext);
    assert_eq!(keys.packet.local.confidentiality_limit(), 1 << 23);
    assert_eq!(keys.packet.local.integrity_limit(), 1 << 52);
  }
}

#[test]
fn rfc9001_retry_integrity_and_all_byte_tampering() {
  let cid = ConnectionId::new(&hex("8394c8f03e515708"));
  let packet = hex("ff000000010008f067a5502a4262b5746f6b656e04a265ba2eff4d829058fb3f0f2496ba");
  let split = packet.len().strict_sub(16);
  assert_eq!(retry_tag(cid, &packet[..split]), packet[split..]);
  for header_len in 0..=split {
    assert!(valid_retry(cid, &packet[..header_len], &packet[header_len..]));
  }
  for index in 0..packet.len() {
    let mut corrupted = packet.clone();
    corrupted[index] ^= 1;
    assert!(!valid_retry(cid, &[], &corrupted));
  }
  assert!(!valid_retry(ConnectionId::new(&[0; 8]), &[], &packet));
  for length in 0..16 {
    assert!(!valid_retry(cid, &[], &packet[..length]));
  }
}

#[cfg(feature = "chacha20poly1305")]
#[test]
fn rfc9001_chacha_short_packet_through_quinn_traits() {
  use quinn_proto::crypto::{HeaderKey, PacketKey};
  use rustls::crypto::cipher::{AeadKey, Iv};
  let key: [u8; 32] = hex("c6d98ff3441c3fe1b2182094f69caa2ed4b716b65488960a7a984979fb23e1c8")
    .try_into()
    .expect("key");
  let iv: [u8; 12] = hex("e0459b3474bdd0e44a41c144").try_into().expect("IV");
  let hp: [u8; 32] = hex("25a282b9e82f06f21f488917a4fc8f1b73573685608597d0efcb076b0ab7a7a4")
    .try_into()
    .expect("header key");
  let alg = crate::quic::CHACHA20_POLY1305;
  let key = packet::Packet(alg.packet_key(AeadKey::from(key), Iv::from(iv)));
  let header = packet::Header(alg.header_protection_key(AeadKey::from(hp)));
  let mut packet = [hex("4200bff401").as_slice(), &[0; 16]].concat();
  key.encrypt(654_360_564, &mut packet, 4);
  header.encrypt(1, &mut packet);
  assert_eq!(packet, hex("4cfe4189655e5cd55c41f69080575d7999c25a5bfb"));
  header.decrypt(1, &mut packet);
  let mut body = BytesMut::from(&packet[4..]);
  key.decrypt(654_360_564, &packet[..4], &mut body).expect("RFC packet");
  assert_eq!(body.as_ref(), &[1]);
  assert_eq!(key.confidentiality_limit(), u64::MAX);
  assert_eq!(key.integrity_limit(), 1 << 36);
}

#[test]
fn truncated_header_protection_is_untouched_and_does_not_panic() {
  let keys = initial_keys(ConnectionId::new(&[0; 8]), Side::Client);
  for length in 0..64 {
    for offset in [0, 1, length, usize::MAX] {
      let packet = vec![0xc3; length];
      let mut encrypted = packet.clone();
      let mut decrypted = packet.clone();
      if offset.checked_add(20).is_none_or(|end| end > length) || offset == 0 {
        keys.header.local.encrypt(offset, &mut encrypted);
        keys.header.local.decrypt(offset, &mut decrypted);
        assert_eq!(encrypted, packet);
        assert_eq!(decrypted, packet);
      }
    }
  }
}

#[test]
fn tls_failures_close_with_their_alert_or_internal_error() {
  use rustls::error::PeerMisbehaved;
  // RFC 9001, section 4.8: CRYPTO_ERROR is 0x0100 plus the TLS alert.
  for (error, alert) in [
    (Error::PeerMisbehaved(PeerMisbehaved::InvalidKeyShare), 0x2f),
    (Error::DecryptError, 0x14),
    (Error::FailedToGetRandomBytes, 0x50),
  ] {
    assert_eq!(crypto_error(&error).code, TransportErrorCode::crypto(alert));
  }
}

#[cfg(all(feature = "x25519", feature = "ml-kem"))]
#[test]
fn hybrid_decapsulation_failure_closes_with_internal_error() {
  // RFC 10024, section 4.2: internal_error (0x50).
  let error = crate::kx_group::decapsulation_failed(rscrypto::MlKemError::InvalidCiphertext);
  assert_eq!(crypto_error(&error).code, TransportErrorCode::crypto(0x50));
}
