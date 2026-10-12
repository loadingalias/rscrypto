use alloc::sync::Arc;
use core::ops::Deref;
use core::time::Duration;

use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::{
  ClientConfig, ClientSide, Connection, ConnectionOutputs, Error, KeyingMaterialExporter, RootCertStore, ServerConfig,
  ServerSide, SideData, Tcp, VecInput,
};

#[derive(Debug)]
pub(crate) struct FixtureTime;

impl rustls::time_provider::TimeProvider for FixtureTime {
  fn current_time(&self) -> Option<UnixTime> {
    // 2026-10-22 UTC, inside the fixed 2026–2036 fixture validity period.
    Some(UnixTime::since_unix_epoch(Duration::from_secs(1_792_627_200)))
  }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Credential {
  #[cfg(feature = "ecdsa-p256")]
  P256,
  #[cfg(feature = "ecdsa-p384")]
  P384,
  #[cfg(feature = "ed25519")]
  Ed25519,
  #[cfg(feature = "rsa")]
  Rsa,
  #[cfg(feature = "ml-dsa")]
  MlDsa44,
  #[cfg(feature = "ml-dsa")]
  MlDsa65,
  #[cfg(feature = "ml-dsa")]
  MlDsa87,
}

impl Credential {
  pub(crate) fn scheme(self) -> rustls::crypto::SignatureScheme {
    match self {
      #[cfg(feature = "ecdsa-p256")]
      Self::P256 => rustls::crypto::SignatureScheme::ECDSA_NISTP256_SHA256,
      #[cfg(feature = "ecdsa-p384")]
      Self::P384 => rustls::crypto::SignatureScheme::ECDSA_NISTP384_SHA384,
      #[cfg(feature = "ed25519")]
      Self::Ed25519 => rustls::crypto::SignatureScheme::ED25519,
      #[cfg(feature = "rsa")]
      Self::Rsa => rustls::crypto::SignatureScheme::RSA_PSS_SHA256,
      #[cfg(feature = "ml-dsa")]
      Self::MlDsa44 => rustls::crypto::SignatureScheme::ML_DSA_44,
      #[cfg(feature = "ml-dsa")]
      Self::MlDsa65 => rustls::crypto::SignatureScheme::ML_DSA_65,
      #[cfg(feature = "ml-dsa")]
      Self::MlDsa87 => rustls::crypto::SignatureScheme::ML_DSA_87,
    }
  }
}

pub(crate) fn credentials(kind: Credential) -> (&'static [u8], &'static [u8], &'static [u8]) {
  macro_rules! files {
    ($kind:literal) => {
      (
        &include_bytes!(concat!("../../tests/vectors/credentials/", $kind, "-root.der"))[..],
        &include_bytes!(concat!("../../tests/vectors/credentials/", $kind, "-leaf.der"))[..],
        &include_bytes!(concat!("../../tests/vectors/credentials/", $kind, "-leaf-key.der"))[..],
      )
    };
  }
  match kind {
    #[cfg(feature = "ecdsa-p256")]
    Credential::P256 => files!("p256"),
    #[cfg(feature = "ecdsa-p384")]
    Credential::P384 => files!("p384"),
    #[cfg(feature = "ed25519")]
    Credential::Ed25519 => files!("ed25519"),
    #[cfg(feature = "rsa")]
    Credential::Rsa => files!("rsa3072"),
    #[cfg(feature = "ml-dsa")]
    Credential::MlDsa44 => files!("mldsa44"),
    #[cfg(feature = "ml-dsa")]
    Credential::MlDsa65 => files!("mldsa65"),
    #[cfg(feature = "ml-dsa")]
    Credential::MlDsa87 => files!("mldsa87"),
  }
}

pub(crate) fn configs(
  client: CryptoProvider,
  server: CryptoProvider,
  kind: Credential,
  client_auth: bool,
) -> (ClientConfig, ServerConfig) {
  configs_with_time(client, server, kind, client_auth, Arc::new(FixtureTime))
}

pub(crate) fn configs_with_time(
  client: CryptoProvider,
  server: CryptoProvider,
  kind: Credential,
  client_auth: bool,
  time: Arc<dyn rustls::time_provider::TimeProvider>,
) -> (ClientConfig, ServerConfig) {
  let (root, leaf, key) = credentials(kind);
  let mut roots = RootCertStore::empty();
  roots.add(CertificateDer::from(root.to_vec())).expect("TLS fixture");
  let server_provider = Arc::new(server);
  let verifier = rustls::server::WebPkiClientVerifier::builder(Arc::new(roots.clone()), &server_provider)
    .build()
    .expect("client verifier");
  let client = ClientConfig::builder_with_details(Arc::new(client), time.clone()).with_root_certificates(roots);
  let mut client = if client_auth {
    client
      .with_client_auth_cert(
        Arc::new(
          rustls::crypto::Identity::from_cert_chain(vec![CertificateDer::from(leaf.to_vec())])
            .expect("fixture identity"),
        ),
        PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key.to_vec())),
      )
      .expect("client credential")
  } else {
    client.with_no_client_auth().expect("client configuration")
  };
  client.alpn_protocols = vec![b"provider-test".to_vec().into()];
  client.max_fragment_size = Some(512);
  let server = ServerConfig::builder_with_details(server_provider, time);
  let server = if client_auth {
    server.with_client_cert_verifier(Arc::new(verifier))
  } else {
    server.with_no_client_auth()
  };
  let mut server = server
    .with_single_cert(
      Arc::new(
        rustls::crypto::Identity::from_cert_chain(vec![CertificateDer::from(leaf.to_vec())]).expect("fixture identity"),
      ),
      PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key.to_vec())),
    )
    .expect("TLS fixture");
  server.alpn_protocols = client.alpn_protocols.clone();
  server.max_fragment_size = Some(512);
  (client, server)
}

pub(crate) struct Peer<S: SideData> {
  pub(crate) connection: Connection<S, Tcp>,
  input: VecInput,
  pub(crate) output: Vec<u8>,
  pub(crate) plaintext: Vec<u8>,
  pub(crate) early: Vec<u8>,
  pub(crate) closed: bool,
  exporter: Option<KeyingMaterialExporter>,
}

impl<S: SideData> Peer<S> {
  fn new(connection: Connection<S, Tcp>, output: Vec<u8>) -> Self {
    Self {
      connection,
      input: VecInput::default(),
      output,
      plaintext: Vec::new(),
      early: Vec::new(),
      closed: false,
      exporter: None,
    }
  }

  pub(crate) fn export(&mut self, label: &[u8], context: Option<&[u8]>) -> [u8; 64] {
    let exporter = self
      .exporter
      .get_or_insert_with(|| self.connection.exporter().expect("handshake exporter"));
    exporter
      .derive(label, context, [0; 64])
      .expect("export keying material")
  }
}

impl<S: SideData> Deref for Peer<S> {
  type Target = ConnectionOutputs;
  fn deref(&self) -> &Self::Target {
    &self.connection
  }
}

pub(crate) fn connections_from_configs(
  client: Arc<ClientConfig>,
  server: Arc<ServerConfig>,
) -> (Peer<ClientSide>, Peer<ServerSide>) {
  let mut output = Vec::new();
  let client = client
    .connect(ServerName::try_from("localhost").expect("TLS fixture"))
    .build(&mut output)
    .expect("TLS fixture");
  (
    Peer::new(client, output),
    Peer::new(Connection::new(server).expect("TLS fixture"), Vec::new()),
  )
}

fn transfer<S: SideData>(
  source: &mut Peer<impl SideData>,
  destination: &mut Peer<S>,
  mut take_early: impl FnMut(&mut rustls::MessageHandler<'_, '_, S>, &mut Vec<u8>) -> Result<(), Error>,
) -> Result<usize, Error> {
  if source.output.len() > 131_072 {
    return Err(Error::General("test flight exceeded its size bound".into()));
  }
  let length = source.output.len();
  // Transport fragmentation is independent of the configured record size.
  for chunk in source.output.chunks(37) {
    let mut chunk = chunk;
    while !chunk.is_empty() {
      if destination.input.read(&mut chunk).expect("read flight") == 0 {
        return Err(Error::General("test receiver stalled".into()));
      }
      let mut messages = destination
        .connection
        .read_tls(&mut destination.input, &mut destination.output);
      take_early(&mut messages, &mut destination.early)?;
      destination.closed |= messages.handle_all(&mut destination.plaintext)?.peer_has_closed();
    }
  }
  source.output.clear();
  Ok(length)
}

pub(crate) fn drive(client: &mut Peer<ClientSide>, server: &mut Peer<ServerSide>) -> Result<(), Error> {
  for _ in 0..128 {
    let progress = transfer(client, server, |messages, early| {
      while let Some(payload) = messages.next_early_data() {
        early.extend_from_slice(payload?.bytes());
      }
      Ok(())
    })?
    .strict_add(transfer(server, client, |_, _| Ok(()))?);
    if client.output.is_empty()
      && server.output.is_empty()
      && !client.connection.is_handshaking()
      && !server.connection.is_handshaking()
    {
      return Ok(());
    }
    if progress == 0 {
      return Err(Error::General("test handshake stalled".into()));
    }
  }
  Err(Error::General("test handshake exceeded its flight count".into()))
}

pub(crate) fn application_exchange(client: &mut Peer<ClientSide>, server: &mut Peer<ServerSide>) {
  let request = b"client application data".repeat(100);
  let response = b"server application data".repeat(100);
  client
    .connection
    .write(request.as_slice().into(), &mut client.output)
    .expect("write request");
  server
    .connection
    .write(response.as_slice().into(), &mut server.output)
    .expect("write response");
  drive(client, server).expect("application transfer");
  assert_eq!(server.plaintext, request);
  assert_eq!(client.plaintext, response);
  client.plaintext.clear();
  server.plaintext.clear();
}
