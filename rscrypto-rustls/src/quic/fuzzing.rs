//! Internal fuzz entry points. Absent from ordinary builds and public API docs.

use alloc::vec::Vec;

use rustls::crypto::{GetRandomFailed, SecureRandom};
use rustls::quic::Side;

use super::{
  AddressTokenKey, ConnectionIdKey, MAX_CONNECTION_ID_LEN, RETRY_INTEGRITY_TAG_LEN, initial_keys, is_valid_retry,
  retry_integrity_tag,
};

/// Fixture entropy: every request is filled with one byte.
#[derive(Debug)]
struct Fixed(u8);

impl SecureRandom for Fixed {
  fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
    output.fill(self.0);
    Ok(())
  }
}

/// Check production Retry tags on arbitrary packets and connection-ID lengths.
///
/// A packet is valid exactly when its last 16 bytes are the tag of the rest,
/// and a connection ID longer than 20 bytes never yields a tag.
pub fn retry(data: &[u8]) {
  if data.len() > 4096 {
    return;
  }
  let cid = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];
  let expected = data
    .len()
    .checked_sub(RETRY_INTEGRITY_TAG_LEN)
    .is_some_and(|split| retry_integrity_tag(&cid, &data[..split]).is_some_and(|tag| tag.as_slice() == &data[split..]));
  assert_eq!(is_valid_retry(&cid, data), expected);
  let tag = retry_integrity_tag(&cid, data).expect("8-byte connection ID");
  let mut packet = [data, &tag].concat();
  assert!(is_valid_retry(&cid, &packet));
  let last = packet.len().strict_sub(1);
  packet[last] ^= 1;
  assert!(!is_valid_retry(&cid, &packet));

  let id_len = data.first().map_or(0, |byte| usize::from(*byte) % 22).min(data.len());
  let (id, rest) = data.split_at(id_len);
  assert_eq!(
    retry_integrity_tag(id, rest).is_some(),
    id.len() <= MAX_CONNECTION_ID_LEN
  );
}

/// Apply and remove production header protection with arbitrary sample and
/// packet-number lengths. A rejected input is left unchanged, and an accepted
/// one round-trips.
pub fn header(data: &[u8]) {
  if data.len() > 4096 {
    return;
  }
  let Some((&lengths, rest)) = data.split_first() else {
    return;
  };
  let Some((&first, rest)) = rest.split_first() else {
    return;
  };
  // Lengths around the valid values: samples of 0..=33 bytes and packet numbers of 0..=7 bytes.
  let sample_len = usize::from(lengths >> 3).min(rest.len());
  let (sample, rest) = rest.split_at(sample_len);
  let pn_len = usize::from(lengths & 7).min(rest.len());
  let pn = &rest[..pn_len];
  let keys = initial_keys(&[1; 8], Side::Client);
  for key in [&*keys.remote.header, &*keys.local.header] {
    let (mut masked_first, mut masked_pn) = (first, pn.to_vec());
    match key.decrypt_in_place(sample, &mut masked_first, &mut masked_pn) {
      Ok(()) => {
        key
          .encrypt_in_place(sample, &mut masked_first, &mut masked_pn)
          .expect("a removed mask can be applied again");
        assert_eq!((masked_first, masked_pn.as_slice()), (first, pn));
      }
      Err(_) => assert_eq!((masked_first, masked_pn.as_slice()), (first, pn)),
    }
  }
}

/// Open arbitrary address tokens and connection IDs, then check authentication
/// failures on real seals.
pub fn token(data: &[u8]) {
  if data.len() > 4144 {
    return;
  }
  const AAD: &[u8] = b"fuzz address context";
  let key = AddressTokenKey::generate(&Fixed(0x42)).expect("fixed fixture key");
  let mut bytes = data.to_vec();
  let _opened = key.open(&mut bytes, AAD);
  let ids = ConnectionIdKey::generate(&Fixed(0x24)).expect("fixed fixture key");
  let _issued = ids.validate(data);
  if data.len() <= 4096 {
    let mut token = data.to_vec();
    key.seal(&Fixed(0x13), &mut token, AAD).expect("bounded seal");
    let nonce = &token[token.len().strict_sub(16)..];
    assert_eq!(nonce, [0x13; 16], "the token ends with its entropy-drawn nonce");
    let mut bad = token.clone();
    bad[0] ^= 1;
    key.open(&mut bad, AAD).expect_err("tampered token");
    let mut other = token.clone();
    key.open(&mut other, b"other context").expect_err("wrong context");
    assert_eq!(key.open(&mut token, AAD).expect("valid token"), data);
  }
}

/// Parse arbitrary TLS handshake fragments through rustls's QUIC server with this provider.
#[cfg(all(feature = "getrandom", feature = "ecdsa-p256", feature = "x25519"))]
pub fn handshake(data: &[u8]) {
  use alloc::sync::Arc;
  use alloc::vec;
  use rustls::quic::{Quic, QuicEvent, Version};
  use rustls::{Connection, ServerSide, VecInput};
  use std::sync::OnceLock;

  if data.len() > 65_536 {
    return;
  }
  static CONFIG: OnceLock<Arc<rustls::ServerConfig>> = OnceLock::new();
  let config = CONFIG.get_or_init(|| {
    let mut config = rustls::ServerConfig::builder(Arc::new(crate::default_provider()))
      .with_no_client_auth()
      .with_single_cert(
        Arc::new(
          rustls::crypto::Identity::from_cert_chain(vec![rustls::pki_types::CertificateDer::from(
            include_bytes!("../../tests/vectors/credentials/p256-leaf.der").as_slice(),
          )])
          .expect("fixture identity"),
        ),
        rustls::pki_types::PrivateKeyDer::Pkcs8(rustls::pki_types::PrivatePkcs8KeyDer::from(
          include_bytes!("../../tests/vectors/credentials/p256-leaf-key.der").as_slice(),
        )),
      )
      .expect("fixed fixture");
    config.alpn_protocols = vec![b"provider-test".to_vec().into()];
    Arc::new(config)
  });
  let mut server =
    Connection::<ServerSide, Quic>::new_quic(config.clone(), Version::V1, Vec::new()).expect("valid configuration");
  let mut input = VecInput::default();
  for chunk in data.chunks(37) {
    let mut remaining = chunk;
    while !remaining.is_empty() {
      if input.read(&mut remaining).is_err() {
        return;
      }
      if server.read_hs(&mut input).is_err() {
        return;
      }
    }
    let _parameters = server.quic_transport_parameters();
    let flight: usize = server
      .events()
      .map(|event| match event {
        QuicEvent::Message(message) => message.len(),
        _ => 0,
      })
      .sum();
    assert!(flight <= 65_536, "bounded fixture flight");
  }
}
