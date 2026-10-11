//! Internal fuzz entry points. Absent from ordinary builds and public API docs.

use super::*;
use quinn_proto::crypto::{HandshakeTokenKey as _, Session as _};
use std::sync::OnceLock;

/// Exercise production Retry validation, with arbitrary framing and a valid-tag anchor.
pub fn retry(data: &[u8]) {
  if data.len() > 4096 {
    return;
  }
  let cid = ConnectionId::new(&[0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08]);
  let split = data.first().map_or(0, |byte| usize::from(*byte).min(data.len()));
  let _accepted = valid_retry(&cid, &data[..split], &data[split..]);
  let tag = retry_tag(&cid, data);
  let packet = [data, &tag].concat();
  assert!(valid_retry(&cid, &[], &packet));
  let mut bad = packet;
  let index = bad.len().strict_sub(1);
  bad[index] ^= 1;
  assert!(!valid_retry(&cid, &[], &bad));
}

/// Remove production header protection on arbitrary packets and offsets.
pub fn header(data: &[u8]) {
  if data.len() > 4096 {
    return;
  }
  let keys = initial_keys(&ConnectionId::new(&[1; 8]), Side::Client);
  let mut bytes = data.to_vec();
  for offset in [
    0,
    data.first().map_or(1, |byte| usize::from(*byte)),
    data.len(),
    usize::MAX,
  ] {
    bytes.copy_from_slice(data);
    keys.header.remote.decrypt(offset, &mut bytes);
  }
}

/// Open arbitrary address tokens, then check authentication failures on real seals.
pub fn token(data: &[u8]) {
  if data.len() > 4112 {
    return;
  }
  let key = token::TokenKey::generate_with(|out| {
    out.fill(0x42);
    Ok(())
  })
  .expect("fixed fixture key");
  let cipher = key.aead_from_hkdf(&[0x13; 16]);
  let mut bytes = data.to_vec();
  let _opened = cipher.open(&mut bytes, b"fuzz address context");
  if data.len() <= 4096 {
    let mut sealed = data.to_vec();
    cipher
      .seal(&mut sealed, b"fuzz address context")
      .expect("bounded first seal");
    let mut bad = sealed.clone();
    bad[0] ^= 1;
    cipher
      .open(&mut bad, b"fuzz address context")
      .expect_err("tampered token");
    assert_eq!(
      cipher.open(&mut sealed, b"fuzz address context").expect("valid token"),
      data
    );
  }
}

/// Parse arbitrary TLS handshake fragments through the production QUIC session.
#[cfg(all(feature = "ecdsa-p256", feature = "x25519"))]
pub fn handshake(data: &[u8]) {
  if data.len() > 65_536 {
    return;
  }
  static CONFIG: OnceLock<Arc<rustls::ServerConfig>> = OnceLock::new();
  let config = CONFIG.get_or_init(|| {
    let mut config = rustls::ServerConfig::builder_with_provider(Arc::new(crate::default_provider()))
      .with_protocol_versions(&[&rustls::version::TLS13])
      .expect("TLS 1.3")
      .with_no_client_auth()
      .with_single_cert(
        vec![rustls::pki_types::CertificateDer::from(
          include_bytes!("../../tests/vectors/credentials/p256-leaf.der").as_slice(),
        )],
        rustls::pki_types::PrivateKeyDer::Pkcs8(rustls::pki_types::PrivatePkcs8KeyDer::from(
          include_bytes!("../../tests/vectors/credentials/p256-leaf-key.der").as_slice(),
        )),
      )
      .expect("fixed fixture");
    config.alpn_protocols = vec![b"provider-test".to_vec()];
    Arc::new(config)
  });
  let server = quic::ServerConnection::new(config.clone(), quic::Version::V1, Vec::new()).expect("valid configuration");
  let mut session = Session::new(quic::Connection::Server(server), Side::Server);
  for chunk in data.chunks(37) {
    if let Err(error) = session.read_handshake(chunk) {
      assert_eq!(
        session.read_handshake(&[]).expect_err("terminal failure").code,
        error.code
      );
      assert!(session.handshake_data().is_none());
      let mut output = Vec::new();
      assert!(session.write_handshake(&mut output).is_none());
      assert!(output.is_empty());
      return;
    }
    let _parameters = session.transport_parameters();
    let mut output = Vec::new();
    let _keys = session.write_handshake(&mut output);
    assert!(output.len() <= 65_536, "bounded fixture flight");
  }
}
