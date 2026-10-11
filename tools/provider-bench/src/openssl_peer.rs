use crate::{Pipe, Profile, credentials};
use alloc::{collections::VecDeque, sync::Arc};
use openssl::{
  pkey::PKey,
  ssl::{
    self, ErrorCode, Ssl, SslContext, SslContextBuilder, SslMethod, SslSession, SslSessionCacheMode, SslStream,
    SslVerifyMode, SslVersion,
  },
  x509::X509,
};
use std::{io::Write, sync::Mutex};

pub(crate) struct Configs {
  client: SslContext,
  server: SslContext,
  session: Arc<Mutex<Option<SslSession>>>,
}

impl Configs {
  #[expect(clippy::panic, reason = "the benchmark catalog must select known workload values")]
  pub(crate) fn new(profile: Profile<'_>) -> Self {
    assert!(
      openssl::version::version().starts_with("OpenSSL 4.0.3 "),
      "benchmark peer requires OpenSSL 4.0.3"
    );
    let (root, leaf, key) = credentials::der(profile.credential);
    let root = X509::from_der(root).expect("root DER");
    let leaf = X509::from_der(leaf).expect("leaf DER");
    let key = PKey::private_key_from_der(key).expect("key DER");
    let session = Arc::new(Mutex::new(None));
    let context = |server: bool| {
      let mut context = SslContextBuilder::new(SslMethod::tls()).expect("TLS context");
      context
        .set_min_proto_version(Some(SslVersion::TLS1_3))
        .expect("TLS version");
      context
        .set_max_proto_version(Some(SslVersion::TLS1_3))
        .expect("TLS version");
      context
        .set_ciphersuites(match profile.suite {
          "AES128-GCM-SHA256" => "TLS_AES_128_GCM_SHA256",
          "AES256-GCM-SHA384" => "TLS_AES_256_GCM_SHA384",
          "CHACHA20-POLY1305-SHA256" => "TLS_CHACHA20_POLY1305_SHA256",
          _ => panic!("unknown suite"),
        })
        .expect("cipher suite");
      context.set_groups_list(profile.group).expect("group");
      context
        .set_sigalgs_list(match profile.credential {
          "p256" => "ecdsa_secp256r1_sha256",
          "p384" => "ecdsa_secp384r1_sha384",
          "ed25519" => "ed25519",
          "rsa3072" => "rsa_pss_rsae_sha256:rsa_pkcs1_sha256",
          "mldsa44" => "mldsa44",
          "mldsa65" => "mldsa65",
          "mldsa87" => "mldsa87",
          _ => panic!("unknown credential"),
        })
        .expect("matched CertificateVerify scheme");
      context.set_certificate(&leaf).expect("certificate");
      context.set_private_key(&key).expect("private key");
      context.check_private_key().expect("matching key");
      context.cert_store_mut().add_cert(root.clone()).expect("trust root");
      context.set_verify(if server {
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
        context
          .set_num_tickets(if profile.resumed { 2 } else { 0 })
          .expect("ticket count");
        context.set_session_cache_mode(SslSessionCacheMode::SERVER);
      } else {
        context.set_session_cache_mode(if profile.resumed {
          SslSessionCacheMode::CLIENT
        } else {
          SslSessionCacheMode::OFF
        });
        let sessions = session.clone();
        context.set_new_session_callback(move |_, ticket| {
          *sessions.lock().expect("ticket cache") = Some(ticket);
        });
      }
      context.build()
    };
    Self {
      client: context(false),
      server: context(true),
      session,
    }
  }

  pub(crate) fn endpoints(&self, left: Pipe, right: Pipe, resumed: bool) -> (Endpoint, Endpoint) {
    let mut client = Ssl::new(&self.client).expect("client SSL");
    client.set_hostname("localhost").expect("SNI");
    client.param_mut().set_host("localhost").expect("hostname verification");
    client.set_connect_state();
    if resumed {
      let session = self
        .session
        .lock()
        .expect("ticket cache")
        .take()
        .expect("TLS 1.3 ticket");
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
  received: VecDeque<u8>,
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
      received: VecDeque::new(),
    }
  }
  pub(crate) fn handshaking(&self) -> bool {
    !self.stream.ssl().is_init_finished()
  }
  pub(crate) fn resumed(&self) -> bool {
    self.stream.ssl().session_reused()
  }
  #[expect(clippy::panic, reason = "an unexpected production TLS error invalidates this benchmark fixture")]
  pub(crate) fn step(&mut self) {
    if self.handshaking()
      && let Err(error) = self.stream.do_handshake()
    {
      assert_eq!(error.code(), ErrorCode::WANT_READ, "OpenSSL handshake: {error}");
    }
    if !self.handshaking() {
      let mut bytes = [0; 16_384];
      loop {
        match self.stream.ssl_read(&mut bytes) {
          Ok(count) => {
            assert!(count > 0);
            self.received.extend(&bytes[..count]);
          }
          Err(error) if error.code() == ErrorCode::WANT_READ => break,
          Err(error) => panic!("OpenSSL application input: {error}"),
        }
      }
    }
  }
  pub(crate) fn exporter(&self, bytes: &mut [u8]) {
    assert_eq!(self.stream.ssl().selected_alpn_protocol(), Some(credentials::ALPN));
    self
      .stream
      .ssl()
      .export_keying_material(bytes, "provider-bench", Some(b"matched"))
      .expect("exporter");
  }
  pub(crate) fn send(&mut self, bytes: &[u8]) {
    self.stream.write_all(bytes).expect("OpenSSL application output");
  }
  pub(crate) fn receive(&mut self, output: &mut [u8]) {
    assert!(self.received.len() >= output.len());
    for byte in output {
      *byte = self.received.pop_front().expect("decrypted application bytes");
    }
  }
}
