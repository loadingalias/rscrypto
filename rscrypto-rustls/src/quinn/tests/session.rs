use super::*;
use crate::tests::tls::{self, Credential};
use quinn_proto::crypto::ClientConfig as _;

fn configs() -> (rustls::ClientConfig, rustls::ServerConfig) {
  tls::configs(
    crate::default_provider(),
    crate::default_provider(),
    Credential::P256,
    true,
  )
}

fn sessions(
  client: rustls::ClientConfig,
  server: rustls::ServerConfig,
  server_params: Vec<u8>,
) -> [Box<dyn crypto::Session>; 2] {
  [
    Box::new(Session::new(
      Arc::new(client)
        .connect(ServerName::try_from("localhost").expect("name"))
        .build_quic(quic::Version::V1, Vec::new())
        .expect("client"),
      Side::Client,
    )),
    Box::new(Session::new(
      Connection::<ServerSide, quic::Quic>::new_quic(Arc::new(server), quic::Version::V1, server_params)
        .expect("server"),
      Side::Server,
    )),
  ]
}

fn drive(peers: &mut [Box<dyn crypto::Session>; 2]) -> Result<(), TransportError> {
  for _ in 0..32 {
    let mut progress = false;
    for source in 0..2 {
      let mut flight = Vec::new();
      let _keys = peers[source].write_handshake(&mut flight);
      if flight.len() >= 131_072 {
        return Err(TransportError::new(TransportErrorCode::INTERNAL_ERROR, String::new()));
      }
      if !flight.is_empty() {
        progress = true;
        for chunk in flight.chunks(37) {
          let _ready = peers[1usize.strict_sub(source)].read_handshake(chunk)?;
        }
      }
    }
    if !progress && peers.iter().all(|peer| !peer.is_handshaking()) {
      return Ok(());
    }
  }
  Err(TransportError::new(TransportErrorCode::INTERNAL_ERROR, String::new()))
}

#[test]
fn configuration_rejects_missing_alpn_and_invalid_early_data_limits() {
  let (client, server) = configs();
  for protocols in [vec![], vec![vec![]], vec![vec![b'a'; 256]]] {
    let mut client = client.clone();
    let mut server = server.clone();
    client.alpn_protocols = protocols.iter().cloned().map(Into::into).collect();
    server.alpn_protocols = client.alpn_protocols.clone();
    client_config(client).expect_err("invalid ALPN");
    server_config(server).expect_err("invalid ALPN");
  }
  let mut server = server;
  server.max_early_data_size = 1024;
  assert!(server_config(server).is_err(), "QUIC accepts only zero or u32::MAX");
}

#[test]
fn quic_omits_middlebox_compatibility_and_rejects_post_handshake_client_auth() {
  let (client, server) = configs();
  let mut peers = sessions(client, server, Vec::new());
  let mut hello = Vec::new();
  assert!(peers[0].write_handshake(&mut hello).is_none());
  // RFC 9001 section 8.4: QUIC omits the compatibility-mode session ID.
  // Handshake header (4), legacy_version (2), random (32), then its length.
  assert_eq!(hello[0], 1, "raw TLS ClientHello, without a record envelope");
  assert_eq!(hello[38], 0, "QUIC must not request middlebox compatibility");

  let (client, server) = configs();
  let mut peers = sessions(client, server, Vec::new());
  drive(&mut peers).expect("TLS 1.3 handshake");
  // RFC 9001 section 4.4 prohibits post-handshake CertificateRequest.
  let error = peers[0]
    .read_handshake(&[13, 0, 0, 3, 0, 0, 0])
    .expect_err("post-handshake authentication is forbidden in QUIC");
  assert_eq!(
    error.code,
    TransportErrorCode::crypto(u8::from(rustls::error::AlertDescription::UnexpectedMessage))
  );
  assert!(peers[0].handshake_data().is_none());
}

#[test]
fn no_common_alpn_and_forbidden_key_update_fail_closed() {
  let (client, mut server) = configs();
  server.alpn_protocols = vec![b"different-application".to_vec().into()];
  let mut peers = sessions(client, server, Vec::new());
  let error = drive(&mut peers).expect_err("disjoint ALPN");
  assert_eq!(
    error.code,
    TransportErrorCode::crypto(u8::from(rustls::error::AlertDescription::NoApplicationProtocol))
  );

  let (client, server) = configs();
  let mut peers = sessions(client, server, Vec::new());
  drive(&mut peers).expect("TLS 1.3 handshake");
  // RFC 9001 section 6 prohibits TLS KeyUpdate: QUIC updates packet keys instead.
  for peer in &mut peers {
    let error = peer.read_handshake(&[24, 0, 0, 1, 0]).expect_err("forbidden KeyUpdate");
    assert_eq!(
      error.code,
      TransportErrorCode::crypto(u8::from(rustls::error::AlertDescription::UnexpectedMessage))
    );
    assert_eq!(peer.read_handshake(&[]).expect_err("terminal failure").code, error.code);
    assert!(peer.handshake_data().is_none());
    assert!(peer.peer_identity().is_none());
    assert!(peer.early_crypto().is_none());
    assert!(peer.next_1rtt_keys().is_none());
    peer.transport_parameters().expect_err("terminal failure");
    let mut output = vec![];
    assert!(peer.write_handshake(&mut output).is_none());
    assert!(output.is_empty());
    assert!(peer.export_keying_material(&mut [0; 32], b"failed", b"").is_err());
  }
}

#[test]
fn malformed_transport_parameters_are_rejected_at_quinn_boundary() {
  let (client, server) = configs();
  // Duplicate max_idle_timeout parameters: each is individually well formed.
  let mut peers = sessions(client, server, vec![1, 1, 0, 1, 1, 0]);
  drive(&mut peers).expect("rustls transports opaque QUIC parameters");
  assert_eq!(
    peers[0].transport_parameters().expect_err("duplicate parameter").code,
    TransportErrorCode::TRANSPORT_PARAMETER_ERROR
  );
}

#[derive(Debug)]
struct FailedEntropy;
impl rustls::crypto::SecureRandom for FailedEntropy {
  fn fill(&self, out: &mut [u8]) -> Result<(), rustls::crypto::GetRandomFailed> {
    if let Some(byte) = out.first_mut() {
      *byte = 0xaa;
    }
    Err(rustls::crypto::GetRandomFailed)
  }
}

#[test]
fn partial_client_entropy_failure_returns_without_a_session() {
  let mut provider = crate::default_provider();
  provider.secure_random = &FailedEntropy;
  let (client, _) = tls::configs(provider, crate::default_provider(), Credential::P256, false);
  let client = Arc::new(Client(Arc::new(client)));
  let params = TransportParameters::read(Side::Server, &mut &[][..]).expect("empty peer parameters");
  let result = client.start_session(1, "localhost", &params);
  assert!(matches!(result, Err(ConnectError::EndpointStopping)));
}
