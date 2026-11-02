use crate::{MAX_PLAINTEXT, Pipe, Profile, Workload, credentials};
use alloc::sync::Arc;
use core::sync::atomic::{AtomicUsize, Ordering};
use openssl::{
  nid::Nid,
  pkey::Id,
  pkey::PKey,
  ssl::{
    self, ErrorCode, Ssl, SslContext, SslContextBuilder, SslMethod, SslMode, SslOptions, SslSession,
    SslSessionCacheMode, SslStream, SslVerifyMode, SslVersion,
  },
  x509::X509,
};
use rustls::enums::ProtocolVersion;
use std::{io::Write, sync::Mutex};

fn ssl_version(version: ProtocolVersion) -> SslVersion {
  if version == ProtocolVersion::TLSv1_2 {
    SslVersion::TLS1_2
  } else {
    SslVersion::TLS1_3
  }
}

#[expect(clippy::panic, reason = "the benchmark catalog must select known workload values")]
fn standard_suite(name: &str) -> &'static str {
  match name {
    "AES128-GCM-SHA256" => "TLS_AES_128_GCM_SHA256",
    "AES256-GCM-SHA384" => "TLS_AES_256_GCM_SHA384",
    "CHACHA20-POLY1305-SHA256" => "TLS_CHACHA20_POLY1305_SHA256",
    _ => panic!("unknown suite"),
  }
}

pub(crate) struct Configs {
  client: SslContext,
  server: SslContext,
  session: Arc<Mutex<Option<SslSession>>>,
  version: ProtocolVersion,
}

impl Configs {
  #[expect(clippy::panic, reason = "the benchmark catalog must select known workload values")]
  pub(crate) fn new(
    profile: Profile<'_>,
    workload: Workload,
    version: ProtocolVersion,
    tickets: Arc<AtomicUsize>,
  ) -> Self {
    assert!(
      openssl::version::version().starts_with("OpenSSL 4.0.3 "),
      "benchmark peer requires OpenSSL 4.0.3"
    );
    let credentials::Credential {
      root,
      leaf,
      key,
      intermediate,
    } = credentials::der(profile.credential, workload);
    let root = X509::from_der(root).expect("root DER");
    let leaf = X509::from_der(leaf).expect("leaf DER");
    let key = PKey::private_key_from_der(key).expect("key DER");
    let session = Arc::new(Mutex::new(None));
    let context = |server: bool| {
      let mut context = SslContextBuilder::new(SslMethod::tls()).expect("TLS context");
      context
        .set_min_proto_version(Some(ssl_version(version)))
        .expect("TLS version");
      context
        .set_max_proto_version(Some(ssl_version(version)))
        .expect("TLS version");
      if version == ProtocolVersion::TLSv1_2 {
        let auth = if profile.credential.starts_with("rsa") {
          "RSA"
        } else {
          "ECDSA"
        };
        let cipher = match profile.suite {
          "CHACHA20-POLY1305-SHA256" => "CHACHA20-POLY1305",
          other => other,
        };
        context
          .set_cipher_list(&format!("ECDHE-{auth}-{cipher}"))
          .expect("cipher suite");
        context.set_options(SslOptions::NO_TICKET);
        if server {
          context.set_options(SslOptions::CIPHER_SERVER_PREFERENCE);
        }
      } else {
        context
          .set_ciphersuites(standard_suite(profile.suite))
          .expect("cipher suite");
      }
      assert!(!profile.group.contains(':'), "one key exchange group");
      let groups = if version == ProtocolVersion::TLSv1_2 {
        // Match rustls's offer: KX first, then the ECDSA certificate's curve.
        let certificate = match profile.credential {
          "p256" => "secp256r1",
          "p384" => "secp384r1",
          _ => profile.group,
        };
        if certificate == profile.group {
          profile.group.to_owned()
        } else {
          format!("{}:{certificate}", profile.group)
        }
      } else if !server && workload == Workload::HelloRetryRequest {
        "*X25519:secp256r1".to_owned()
      } else {
        profile.group.to_owned()
      };
      context.set_groups_list(&groups).expect("group");
      context
        .set_sigalgs_list(match profile.credential {
          "p256" => "ecdsa_secp256r1_sha256",
          "p384" => "ecdsa_secp384r1_sha384",
          "ed25519" => "ed25519",
          "rsa2048" | "rsa3072" | "rsa4096" => "rsa_pss_rsae_sha256:rsa_pkcs1_sha256",
          "rsa3072-sha384" | "rsa4096-sha384" => "rsa_pss_rsae_sha384:rsa_pkcs1_sha256",
          "rsa3072-sha512" | "rsa4096-sha512" => "rsa_pss_rsae_sha512:rsa_pkcs1_sha256",
          "mldsa44" => "mldsa44",
          "mldsa65" => "mldsa65",
          "mldsa87" => "mldsa87",
          _ => panic!("unknown credential"),
        })
        .expect("matched CertificateVerify scheme");
      if server || workload != Workload::ServerAuth {
        context.set_certificate(&leaf).expect("certificate");
        context.set_private_key(&key).expect("private key");
        context.check_private_key().expect("matching key");
        if let Some(intermediate) = intermediate {
          context
            .add_extra_chain_cert(X509::from_der(intermediate).expect("intermediate DER"))
            .expect("chain");
        }
      }
      // Send exactly the configured chain, without verifying it locally on each connection.
      context.set_mode(SslMode::NO_AUTO_CHAIN);
      context.cert_store_mut().add_cert(root.clone()).expect("trust root");
      context.set_verify(if server && workload == Workload::ServerAuth {
        SslVerifyMode::NONE
      } else if server {
        SslVerifyMode::PEER | SslVerifyMode::FAIL_IF_NO_PEER_CERT
      } else {
        SslVerifyMode::PEER
      });
      context
        .verify_param_mut()
        .set_time(credentials::TIME.try_into().expect("fixture timestamp"));
      context.set_alpn_protos(b"\x0dprovider-test").expect("ALPN");
      if server {
        context.set_alpn_select_callback(|_, offered| {
          ssl::select_next_proto(b"\x0dprovider-test", offered).ok_or(ssl::AlpnError::ALERT_FATAL)
        });
        context
          .set_session_id_context(b"provider-bench")
          .expect("session scope");
        // This setter controls TLS 1.3 only. Every TLS 1.3 resumed-workload
        // connection issues one ticket; NO_TICKET above disables TLS 1.2 tickets.
        context
          .set_num_tickets(usize::from(profile.resumed))
          .expect("ticket count");
        context.set_session_cache_mode(if version == ProtocolVersion::TLSv1_2 && !profile.resumed {
          SslSessionCacheMode::OFF
        } else {
          SslSessionCacheMode::SERVER
        });
        context.set_session_cache_size(256);
        if workload == Workload::Stateful {
          // In TLS 1.3 NO_TICKET selects opaque references to the server cache.
          // NewSessionTicket is still sent once, just as in rustls's stateful mode.
          context.set_options(SslOptions::NO_TICKET);
        }
      } else {
        // The callback alone keeps one session. Count TLS 1.3 ticket acceptance
        // like the rustls store; TLS 1.2 reuses its session ID without renewal.
        context.set_session_cache_mode(if version == ProtocolVersion::TLSv1_2 && !profile.resumed {
          SslSessionCacheMode::OFF
        } else {
          SslSessionCacheMode::CLIENT | SslSessionCacheMode::NO_INTERNAL_STORE
        });
        let sessions = session.clone();
        let tickets = tickets.clone();
        context.set_new_session_callback(move |_, ticket| {
          if version == ProtocolVersion::TLSv1_3 {
            tickets.fetch_add(1, Ordering::Relaxed);
          }
          *sessions.lock().expect("ticket cache") = Some(ticket);
        });
      }
      context.build()
    };
    Self {
      client: context(false),
      server: context(true),
      session,
      version,
    }
  }

  pub(crate) fn endpoints(&self, left: Pipe, right: Pipe, resumed: bool) -> (Endpoint, Endpoint) {
    let mut client = Ssl::new(&self.client).expect("client SSL");
    client.set_hostname("localhost").expect("SNI");
    client.param_mut().set_host("localhost").expect("hostname verification");
    client.set_connect_state();
    if resumed {
      let mut cache = self.session.lock().expect("session cache");
      // TLS 1.2 session IDs are reusable; TLS 1.3 consumes a ticket and renews it.
      let session = if self.version == ProtocolVersion::TLSv1_2 {
        cache.clone()
      } else {
        cache.take()
      }
      .expect("seeded session");
      // SAFETY: this session came only from this exact client SslContext's
      // callback. Its owned reference stays valid for the call; SSL retains it.
      unsafe { client.set_session(&session) }.expect("resume with same context");
    }
    let mut server = Ssl::new(&self.server).expect("server SSL");
    server.set_accept_state();
    (Endpoint::new(client, left), Endpoint::new(server, right))
  }
}

pub(crate) struct Endpoint {
  stream: SslStream<Pipe>,
  pub(crate) pipe: Pipe,
  /// Decrypted application bytes, retained across reads like the rustls endpoint's.
  /// Allocated and zeroed once, so a timed read neither allocates nor clears memory.
  plaintext: Vec<u8>,
  filled: usize,
}

impl Endpoint {
  pub(crate) fn close(&mut self) {
    // Send the real close_notify before freeing SSL. OpenSSL invalidates
    // cached sessions when SSL is freed without shutdown; rustls has no
    // such invalidation. Both measured peers send the same notification.
    self.stream.shutdown().expect("OpenSSL close notification");
  }
  fn new(ssl: Ssl, pipe: Pipe) -> Self {
    Self {
      stream: SslStream::new(ssl, pipe.clone()).expect("memory BIO"),
      pipe,
      plaintext: vec![0; MAX_PLAINTEXT],
      filled: 0,
    }
  }
  pub(crate) fn handshaking(&self) -> bool {
    !self.stream.ssl().is_init_finished()
  }
  pub(crate) fn resumed(&self) -> bool {
    self.stream.ssl().session_reused()
  }
  #[expect(
    clippy::panic,
    reason = "an unexpected production TLS error invalidates this benchmark fixture"
  )]
  pub(crate) fn step(&mut self) {
    if self.handshaking()
      && let Err(error) = self.stream.do_handshake()
    {
      assert_eq!(error.code(), ErrorCode::WANT_READ, "OpenSSL handshake: {error}");
    }
    // Read only received records, as the rustls endpoint does. SSL_read
    // decrypts into the retained buffer: the first of two plaintext copies.
    while !self.handshaking() && self.pipe.pending() {
      assert!(self.filled < self.plaintext.len(), "OpenSSL plaintext buffer is full");
      match self.stream.ssl_read(&mut self.plaintext[self.filled..]) {
        Ok(count) => {
          assert!(count > 0);
          self.filled = self.filled.strict_add(count);
        }
        Err(error) if error.code() == ErrorCode::WANT_READ => break,
        Err(error) => panic!("OpenSSL application input: {error}"),
      }
    }
  }
  /// Copy the decrypted bytes out of the plaintext buffer: the second plaintext copy.
  pub(crate) fn receive(&mut self, bytes: &mut [u8]) {
    assert_eq!(self.filled, bytes.len(), "application length");
    bytes.copy_from_slice(&self.plaintext[..self.filled]);
    self.filled = 0;
  }
  pub(crate) fn exporter(&self, bytes: &mut [u8]) {
    self
      .stream
      .ssl()
      .export_keying_material(bytes, "provider-bench", Some(b"matched"))
      .expect("exporter");
  }
  /// Verify protocol, suite, EMS, full-handshake key exchange, ALPN and identity.
  pub(crate) fn check(
    &self,
    profile: Profile<'_>,
    credential: credentials::Credential,
    authenticated: bool,
    version: ProtocolVersion,
    resumed: bool,
    side: &str,
  ) {
    let ssl = self.stream.ssl();
    assert_eq!(ssl.version2(), Some(ssl_version(version)), "{side} protocol version");
    if version == ProtocolVersion::TLSv1_2 {
      assert_eq!(ssl.extms_support(), Some(true), "{side} EMS");
      if !resumed && side == "client" {
        let key = ssl.peer_tmp_key().expect("server ECDHE public key");
        match profile.group {
          "X25519" => assert_eq!(key.id(), Id::X25519, "selected KX"),
          curve => assert_eq!(
            key.ec_key().expect("ECDHE key").group().curve_name(),
            Some(if curve == "secp256r1" {
              Nid::X9_62_PRIME256V1
            } else {
              Nid::SECP384R1
            }),
            "selected KX",
          ),
        }
      }
    }
    assert_eq!(
      ssl.current_cipher().map(|cipher| cipher.protocol_id()),
      Some(crate::selected_suite(profile, version).0.to_be_bytes()),
      "{side} cipher suite"
    );
    assert_eq!(ssl.selected_alpn_protocol(), Some(credentials::ALPN), "{side} ALPN");
    if !authenticated {
      assert!(ssl.peer_certificate().is_none(), "{side} must have no peer identity");
      assert_eq!(
        ssl.peer_cert_chain().map_or(0, |chain| chain.len()),
        0,
        "{side} must have no peer chain"
      );
      return;
    }
    let peer = ssl.peer_certificate().expect("peer certificate");
    assert_eq!(peer.to_der().expect("peer DER"), credential.leaf, "{side} peer leaf");
    // A client's chain includes the peer's leaf; a server's does not.
    let client = side == "client";
    let chain = ssl
      .peer_cert_chain()
      .map(|chain| {
        chain
          .iter()
          .map(|cert| cert.to_der().expect("peer DER"))
          .collect::<Vec<_>>()
      })
      .unwrap_or_default();
    let expected = if client { Some(credential.leaf) } else { None }
      .into_iter()
      .chain(credential.intermediate)
      .map(<[u8]>::to_vec)
      .collect::<Vec<_>>();
    // OpenSSL restores the peer certificate but not the transmitted chain on resumption.
    if profile.resumed && chain.is_empty() {
      return;
    }
    assert_eq!(chain, expected, "{side} peer chain");
  }
  pub(crate) fn send(&mut self, bytes: &[u8]) {
    self.stream.write_all(bytes).expect("OpenSSL application output");
  }
}
