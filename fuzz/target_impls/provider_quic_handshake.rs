extern crate alloc;

use alloc::sync::Arc;
use std::sync::OnceLock;

use rustls::crypto::Identity;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::quic::{Quic, QuicEvent, Version};
use rustls::{Connection, ServerConfig, ServerSide, VecInput};

/// Parse arbitrary TLS handshake fragments through rustls's QUIC server with this provider.
pub(super) fn run(data: &[u8]) {
  if data.len() > 65_536 {
    return;
  }
  static CONFIG: OnceLock<Arc<ServerConfig>> = OnceLock::new();
  let config = CONFIG.get_or_init(|| {
    let mut config = ServerConfig::builder(Arc::new(rscrypto_rustls::default_provider()))
      .with_no_client_auth()
      .with_single_cert(
        Arc::new(
          Identity::from_cert_chain(vec![CertificateDer::from(
            include_bytes!("../../rscrypto-rustls/tests/vectors/credentials/p256-leaf.der").as_slice(),
          )])
          .expect("fixture identity"),
        ),
        PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(
          include_bytes!("../../rscrypto-rustls/tests/vectors/credentials/p256-leaf-key.der").as_slice(),
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
