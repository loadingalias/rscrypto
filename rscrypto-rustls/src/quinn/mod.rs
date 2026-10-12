//! QUIC v1 integration with Quinn using rscrypto cryptography.
//!
//! Build rustls configurations explicitly with this provider and TLS 1.3, set
//! the application's ALPN protocols, then pass them to [`client_config`] or
//! [`server_config`]. Construct the endpoint with [`endpoint_config`]. Enable
//! only `runtime-tokio` (or another runtime) on the application's `quinn` crate;
//! its `rustls-*`, `ring`, and `aws-lc-rs` features are unnecessary.
//! The pinned Quinn manifest fix keeps the browser-WASM graph free of C backends.
//!
//! Initial packets and Retry always use RFC 9001 AES-128-GCM, independently of
//! the negotiated suite. Rustls owns handshake secrets and key updates. Quinn
//! enforces the packet keys' confidentiality and integrity limits. Application
//! policy still owns trust, ALPN, resumption, admission, and replay protection.

use alloc::{boxed::Box, collections::VecDeque, string::String, sync::Arc, vec::Vec};
use core::any::Any;
use quinn_proto::crypto::{self, ExportKeyingMaterialError, KeyPair, Keys, UnsupportedVersion};
use quinn_proto::transport_parameters::TransportParameters;
use quinn_proto::{ConnectError, ConnectionId, Side, TransportError, TransportErrorCode};
use rscrypto::aead::{Nonce96, expert::AeadWithNonce};
use rscrypto::{Aes128Gcm, Aes128GcmKey, Aes128GcmTag};
use rustls::crypto::CipherSuite;
use rustls::crypto::kx::NamedGroup;
use rustls::enums::ApplicationProtocol;
use rustls::pki_types::ServerName;
use rustls::{ClientSide, Connection, KeyingMaterialExporter, ServerSide, SideData, VecInput};
use rustls::{Error, quic};

mod packet;
mod token;

#[cfg(any(fuzzing, rscrypto_internal_fuzzing))]
#[doc(hidden)]
pub mod fuzzing;

/// Negotiated QUIC handshake data, returned by Quinn's `handshake_data()`.
#[derive(Clone, Debug)]
pub struct HandshakeData {
  /// Negotiated application protocol. QUIC requires a nonempty ALPN selection.
  pub protocol: Vec<u8>,
  /// Client's server name, available on the server side.
  pub server_name: Option<String>,
  /// Negotiated TLS 1.3 cipher suite.
  pub cipher_suite: CipherSuite,
  /// Negotiated key exchange, if available at this handshake stage.
  pub key_exchange_group: Option<NamedGroup>,
  /// Whether rustls performed a full or resumed handshake, once it is complete.
  pub handshake_kind: Option<rustls::HandshakeKind>,
}

/// Create a QUIC v1 endpoint configuration with a fresh clearing reset key.
///
/// Retain this configuration across endpoints which should share reset tokens.
/// Fails if the OS random source fails.
pub fn endpoint_config() -> Result<quinn_proto::EndpointConfig, Error> {
  endpoint_config_with_random(crate::random::SECURE_RANDOM)
}

/// Create a QUIC v1 endpoint using `random` for the reset and CID filter keys.
///
/// The source must provide cryptographically secure random bytes. A failure
/// returns an error without fallback. This source is used only during construction;
/// TLS and Quinn's transport randomness are configured separately.
pub fn endpoint_config_with_random(
  random: &dyn rustls::crypto::SecureRandom,
) -> Result<quinn_proto::EndpointConfig, Error> {
  let mut config = quinn_proto::EndpointConfig::new(Arc::new(token::ResetKey::generate_with(|out| random.fill(out))?));
  // Quinn's default factory gives each endpoint a different CID filter key.
  // Keep that filter stable with the reset key when this configuration is cloned.
  // This non-cryptographic filter is not the reset token's authentication key.
  let mut cid_key = [0; 8];
  random.fill(&mut cid_key)?;
  let cid_key = u64::from_le_bytes(cid_key);
  config.cid_generator(move || Box::new(quinn_proto::HashedConnectionIdGenerator::from_key(cid_key)));
  config.supported_versions(vec![1]);
  Ok(config)
}

fn validate(alpn: &[ApplicationProtocol<'_>], provider: &rustls::crypto::CryptoProvider) -> Result<(), Error> {
  if alpn.is_empty()
    || alpn
      .iter()
      .any(|protocol| protocol.as_ref().is_empty() || protocol.as_ref().len() > 255)
  {
    return Err(Error::General(
      "QUIC requires nonempty ALPN protocols of at most 255 bytes".into(),
    ));
  }
  if provider.tls13_cipher_suites.is_empty() || !provider.tls12_cipher_suites.is_empty() {
    return Err(Error::General("QUIC requires a TLS 1.3-only provider".into()));
  }
  Ok(())
}

/// Adapt a TLS 1.3 client configuration with explicit ALPN to QUIC v1.
///
/// Use this crate's provider in `tls`. Early data stays as configured by the
/// caller and is disabled by rustls by default. Quinn 0.11 cannot express a TLS
/// initialization error in `ConnectError`; a later entropy failure fails closed
/// as `ConnectError::EndpointStopping`, without panicking.
pub fn client_config(tls: rustls::ClientConfig) -> Result<quinn_proto::ClientConfig, Error> {
  validate(&tls.alpn_protocols, tls.provider())?;
  let mut config = quinn_proto::ClientConfig::new(Arc::new(Client(Arc::new(tls))));
  config.version(1);
  Ok(config)
}

/// Adapt a TLS 1.3 server configuration with explicit ALPN to QUIC v1.
///
/// Creates a fresh clearing address-token key using the TLS provider's entropy
/// source, propagating failures without fallback. Retain the returned configuration
/// to accept its outstanding tokens. Set rustls's `ticketer` to [`crate::ticket::ticketer`]
/// for stateless resumption. QUIC early data must stay disabled or explicitly use
/// `max_early_data_size = u32::MAX`, with application replay protection.
pub fn server_config(tls: rustls::ServerConfig) -> Result<quinn_proto::ServerConfig, Error> {
  validate(&tls.alpn_protocols, tls.provider())?;
  let tls = Arc::new(tls);
  // Pinned rustls validates only immutable configuration here: versions,
  // QUIC suites, early-data limits, and maximum fragment size. No entropy is
  // requested until processing ClientHello. Preserve that invariant on upgrades.
  Connection::<ServerSide, quic::Quic>::new_quic(Arc::clone(&tls), quic::Version::V1, Vec::new())?;
  let token_key = token::TokenKey::generate_with(|out| tls.provider().secure_random.fill(out))?;
  Ok(quinn_proto::ServerConfig::new(
    Arc::new(Server(tls)),
    Arc::new(token_key),
  ))
}

struct Client(Arc<rustls::ClientConfig>);
struct Server(Arc<rustls::ServerConfig>);

impl crypto::ClientConfig for Client {
  fn start_session(
    self: Arc<Self>,
    version: u32,
    server_name: &str,
    params: &TransportParameters,
  ) -> Result<Box<dyn crypto::Session>, ConnectError> {
    if version != 1 {
      return Err(ConnectError::UnsupportedVersion);
    }
    let name =
      ServerName::try_from(server_name.to_owned()).map_err(|_| ConnectError::InvalidServerName(server_name.into()))?;
    let mut encoded = Vec::new();
    params.write(&mut encoded);
    let connection = self
      .0
      .connect(name)
      .build_quic(quic::Version::V1, encoded)
      .map_err(|_| ConnectError::EndpointStopping)?;
    Ok(Box::new(Session::new(connection, Side::Client)))
  }
}

impl crypto::ServerConfig for Server {
  fn initial_keys(&self, version: u32, dst_cid: &ConnectionId) -> Result<Keys, UnsupportedVersion> {
    if version != 1 {
      return Err(UnsupportedVersion);
    }
    Ok(initial_keys(dst_cid, Side::Server))
  }

  fn retry_tag(&self, version: u32, orig_dst_cid: &ConnectionId, packet: &[u8]) -> [u8; 16] {
    assert_eq!(
      version, 1,
      "Quinn calls Retry only for a version accepted by initial_keys"
    );
    retry_tag(orig_dst_cid, packet)
  }

  fn start_session(self: Arc<Self>, version: u32, params: &TransportParameters) -> Box<dyn crypto::Session> {
    assert_eq!(version, 1, "Quinn starts only a version accepted by initial_keys");
    let mut encoded = Vec::new();
    params.write(&mut encoded);
    let connection = Connection::<ServerSide, quic::Quic>::new_quic(Arc::clone(&self.0), quic::Version::V1, encoded)
      .expect("immutable TLS configuration was validated by server_config");
    Box::new(Session::new(connection, Side::Server))
  }
}

fn initial_keys(cid: &ConnectionId, side: Side) -> Keys {
  let suite = crate::cipher_suite::TLS13_AES_128_GCM_SHA256;
  let side = match side {
    Side::Client => quic::Side::Client,
    Side::Server => quic::Side::Server,
  };
  packet::keys(quic::Keys::initial(
    quic::Version::V1,
    quic::Suite::try_from(suite).expect("static QUIC suite"),
    cid,
    side,
  ))
}

// RFC 9001 section 5.8: public fixed Retry key and nonce for QUIC v1.
const RETRY_KEY: [u8; 16] = [
  0xbe, 0x0c, 0x69, 0x0b, 0x9f, 0x66, 0x57, 0x5a, 0x1d, 0x76, 0x6b, 0x54, 0xe3, 0x68, 0xc8, 0x4e,
];
const RETRY_NONCE: [u8; 12] = [0x46, 0x15, 0x99, 0xd3, 0x5d, 0x63, 0x2b, 0xf2, 0x23, 0x98, 0x25, 0xbb];

fn retry_aad(cid: &ConnectionId, header: &[u8], body: &[u8]) -> Vec<u8> {
  let mut aad = Vec::with_capacity(
    1usize
      .strict_add(cid.len())
      .strict_add(header.len())
      .strict_add(body.len()),
  );
  aad.push(u8::try_from(cid.len()).expect("QUIC connection IDs are at most 20 bytes"));
  aad.extend_from_slice(cid);
  aad.extend_from_slice(header);
  aad.extend_from_slice(body);
  aad
}

fn retry_tag(cid: &ConnectionId, packet: &[u8]) -> [u8; 16] {
  let cipher = Aes128Gcm::new(&Aes128GcmKey::from_bytes(RETRY_KEY));
  *cipher
    .encrypt_in_place(&Nonce96::from_bytes(RETRY_NONCE), &retry_aad(cid, packet, &[]), &mut [])
    .expect("Retry has an empty plaintext")
    .as_bytes()
}

fn valid_retry(cid: &ConnectionId, header: &[u8], payload: &[u8]) -> bool {
  let Some(length) = payload.len().checked_sub(16) else {
    return false;
  };
  let (body, tag) = payload.split_at(length);
  let Ok(tag) = <[u8; 16]>::try_from(tag) else {
    return false;
  };
  Aes128Gcm::new(&Aes128GcmKey::from_bytes(RETRY_KEY))
    .decrypt_in_place(
      &Nonce96::from_bytes(RETRY_NONCE),
      &retry_aad(cid, header, body),
      &mut [],
      &Aes128GcmTag::from_bytes(tag),
    )
    .is_ok()
}

trait EndpointSide: SideData {
  fn server_name(&self) -> Option<String>;
  fn early_data_accepted(&self) -> Option<bool>;
}

impl EndpointSide for ClientSide {
  fn server_name(&self) -> Option<String> {
    None
  }
  fn early_data_accepted(&self) -> Option<bool> {
    Some(self.is_early_data_accepted())
  }
}

impl EndpointSide for ServerSide {
  fn server_name(&self) -> Option<String> {
    self.server_name().map(|name| String::from(name.as_ref()))
  }
  fn early_data_accepted(&self) -> Option<bool> {
    None
  }
}

struct Session<S: EndpointSide> {
  connection: Connection<S, quic::Quic>,
  input: VecInput,
  pending: VecDeque<quic::QuicEvent>,
  exporter: Option<KeyingMaterialExporter>,
  side: Side,
  next: Option<quic::Secrets>,
  notified: bool,
  failed: Option<TransportError>,
}

impl<S: EndpointSide> Session<S> {
  fn new(mut connection: Connection<S, quic::Quic>, side: Side) -> Self {
    let pending = connection.events().collect();
    Self {
      connection,
      input: VecInput::default(),
      pending,
      exporter: None,
      side,
      next: None,
      notified: false,
      failed: None,
    }
  }
}

impl<S: EndpointSide> crypto::Session for Session<S>
where
  Connection<S, quic::Quic>: Send + Sync,
{
  fn initial_keys(&self, cid: &ConnectionId, side: Side) -> Keys {
    initial_keys(cid, side)
  }

  fn handshake_data(&self) -> Option<Box<dyn Any>> {
    if self.failed.is_some() {
      return None;
    }
    Some(Box::new(HandshakeData {
      protocol: self.connection.alpn_protocol()?.as_ref().to_vec(),
      server_name: self.connection.side().server_name(),
      cipher_suite: self.connection.negotiated_cipher_suite()?.suite(),
      key_exchange_group: self
        .connection
        .negotiated_key_exchange_group()
        .map(|group| group.name()),
      handshake_kind: self.connection.handshake_kind(),
    }))
  }

  fn peer_identity(&self) -> Option<Box<dyn Any>> {
    if self.failed.is_some() {
      return None;
    }
    let rustls::crypto::Identity::X509(identity) = self.connection.peer_identity()?.identity() else {
      return None;
    };
    let certificates: Vec<_> = core::iter::once(identity.end_entity.clone())
      .chain(identity.intermediates.iter().cloned())
      .collect();
    Some(Box::new(certificates))
  }

  fn early_crypto(&self) -> Option<(Box<dyn crypto::HeaderKey>, Box<dyn crypto::PacketKey>)> {
    if self.failed.is_some() {
      return None;
    }
    let keys = self.connection.zero_rtt_keys()?;
    Some((
      Box::new(packet::Header(keys.header)),
      Box::new(packet::Packet(keys.packet)),
    ))
  }

  fn early_data_accepted(&self) -> Option<bool> {
    self.connection.side().early_data_accepted()
  }

  fn is_handshaking(&self) -> bool {
    self.connection.is_handshaking()
  }

  fn read_handshake(&mut self, input: &[u8]) -> Result<bool, TransportError> {
    if let Some(error) = &self.failed {
      return Err(error.clone());
    }
    let result = (|| {
      let mut remaining = input;
      while !remaining.is_empty() {
        self
          .input
          .read(&mut remaining)
          .map_err(|_| Error::PeerSentOversizedRecord)?;
        self.connection.read_hs(&mut self.input)?;
      }
      self.pending.extend(self.connection.events());
      if !self.connection.is_handshaking() && self.exporter.is_none() {
        self.exporter = Some(self.connection.exporter()?);
      }
      Ok::<(), Error>(())
    })()
    .map_err(|error| {
      let alert =
        rustls::error::AlertDescription::try_from(&error).unwrap_or(rustls::error::AlertDescription::InternalError);
      TransportError::from(TransportErrorCode::crypto(u8::from(alert)))
    })
    .and_then(|()| {
      // A resumed client can restore cached transport parameters before
      // EncryptedExtensions selects ALPN. Check at handshake completion,
      // before Quinn can publish an established connection.
      if !self.connection.is_handshaking() && self.connection.alpn_protocol().is_none() {
        return Err(
          TransportErrorCode::crypto(u8::from(rustls::error::AlertDescription::NoApplicationProtocol)).into(),
        );
      }
      let ready = !self.notified && self.handshake_data().is_some();
      self.notified |= ready;
      Ok(ready)
    });
    if let Err(error) = &result {
      self.failed = Some(error.clone());
    }
    result
  }

  fn transport_parameters(&self) -> Result<Option<TransportParameters>, TransportError> {
    if let Some(error) = &self.failed {
      return Err(error.clone());
    }
    self
      .connection
      .quic_transport_parameters()
      .map(|mut encoded| {
        TransportParameters::read(self.side, &mut encoded)
          .map_err(|_| TransportErrorCode::TRANSPORT_PARAMETER_ERROR.into())
      })
      .transpose()
  }

  fn write_handshake(&mut self, output: &mut Vec<u8>) -> Option<Keys> {
    if self.failed.is_some() {
      return None;
    }
    while let Some(event) = self.pending.pop_front() {
      match event {
        quic::QuicEvent::Message(message) => output.extend_from_slice(&message),
        quic::QuicEvent::KeyChange(quic::KeyChange::Handshake { keys }) => return Some(packet::keys(keys)),
        quic::QuicEvent::KeyChange(quic::KeyChange::OneRtt { keys, next }) => {
          self.next = Some(next);
          return Some(packet::keys(keys));
        }
        _ => {}
      }
    }
    None
  }

  fn next_1rtt_keys(&mut self) -> Option<KeyPair<Box<dyn crypto::PacketKey>>> {
    if self.failed.is_some() {
      return None;
    }
    let keys = self.next.as_mut()?.next_packet_keys();
    Some(KeyPair {
      local: Box::new(packet::Packet(keys.local)),
      remote: Box::new(packet::Packet(keys.remote)),
    })
  }

  fn is_valid_retry(&self, cid: &ConnectionId, header: &[u8], payload: &[u8]) -> bool {
    valid_retry(cid, header, payload)
  }

  fn export_keying_material(
    &self,
    output: &mut [u8],
    label: &[u8],
    context: &[u8],
  ) -> Result<(), ExportKeyingMaterialError> {
    if self.failed.is_some() {
      return Err(ExportKeyingMaterialError);
    }
    self
      .exporter
      .as_ref()
      .ok_or(ExportKeyingMaterialError)?
      .derive(label, Some(context), output)
      .map(|_| ())
      .map_err(|_| ExportKeyingMaterialError)
  }
}

#[cfg(test)]
mod tests;
