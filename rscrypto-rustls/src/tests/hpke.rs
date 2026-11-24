use rustls::Error;
use rustls::crypto::hpke::{EncapsulatedSecret, Hpke, HpkePrivateKey, HpkePublicKey};
use rustls::crypto::{GetRandomFailed, SecureRandom};
use rustls_aws_lc_rs::hpke::DH_KEM_X25519_HKDF_SHA256_AES_128 as PEER;

use crate::hpke::X25519HkdfSha256Aes128Gcm;

const OURS: X25519HkdfSha256Aes128Gcm =
  X25519HkdfSha256Aes128Gcm::new(rustls_ring::DEFAULT_TLS13_PROVIDER.secure_random);

#[test]
fn both_hpke_roles_exchange_single_shots_and_successive_messages_with_aws_lc() -> Result<(), Error> {
  for (sender, recipient) in [(&OURS as &dyn Hpke, PEER as &dyn Hpke), (PEER, &OURS)] {
    let (public, private) = recipient.generate_key_pair()?;
    let (enc, ciphertext) = sender.seal(b"info", b"outer", b"inner", &public)?;
    assert_eq!(
      recipient.open(&enc, b"info", b"outer", &ciphertext, &private)?,
      b"inner"
    );

    let (enc, mut sealer) = sender.setup_sealer(b"info", &public)?;
    let mut opener = recipient.setup_opener(&enc, b"info", &private)?;
    for (aad, message) in [(b"first".as_slice(), b"".as_slice()), (b"retry", b"second inner hello")] {
      let ciphertext = sealer.seal(aad, message)?;
      assert_eq!(opener.open(aad, &ciphertext)?, message);
    }
  }
  Ok(())
}

#[test]
fn invalid_hpke_inputs_fail_and_an_unauthenticated_message_preserves_the_sequence() -> Result<(), Error> {
  let (public, private) = OURS.generate_key_pair()?;
  let (enc, ciphertext) = PEER.seal(b"info", b"outer", b"inner", &public)?;
  for bytes in [vec![0; 31], vec![0; 32], vec![0; 33]] {
    assert!(matches!(
      OURS.setup_sealer(b"info", &HpkePublicKey(bytes.clone())),
      Err(Error::EncryptError)
    ));
    assert!(matches!(
      OURS.setup_opener(&EncapsulatedSecret(bytes), b"info", &private),
      Err(Error::DecryptError)
    ));
  }
  for length in [31, 33] {
    assert!(matches!(
      OURS.setup_opener(&enc, b"info", &HpkePrivateKey::from(vec![0; length])),
      Err(Error::DecryptError)
    ));
  }
  assert!(matches!(
    OURS.open(&enc, b"wrong", b"outer", &ciphertext, &private),
    Err(Error::DecryptError)
  ));
  let mut opener = OURS.setup_opener(&enc, b"info", &private)?;
  assert!(matches!(opener.open(b"changed", &ciphertext), Err(Error::DecryptError)));
  assert!(matches!(opener.open(b"outer", &[]), Err(Error::DecryptError)));
  assert_eq!(opener.open(b"outer", &ciphertext)?, b"inner");
  Ok(())
}

#[test]
fn incomplete_entropy_aborts_encapsulation_and_recipient_key_generation() -> Result<(), Error> {
  #[derive(Debug)]
  struct Interrupted;
  impl SecureRandom for Interrupted {
    fn fill(&self, out: &mut [u8]) -> Result<(), GetRandomFailed> {
      out[0] = 0x42;
      Err(GetRandomFailed)
    }
  }
  let suite = X25519HkdfSha256Aes128Gcm::new(&Interrupted);
  let (public, _) = PEER.generate_key_pair()?;
  assert!(matches!(
    suite.setup_sealer(b"info", &public),
    Err(Error::FailedToGetRandomBytes)
  ));
  assert!(matches!(suite.generate_key_pair(), Err(Error::FailedToGetRandomBytes)));
  Ok(())
}

#[cfg(feature = "default")]
#[test]
fn a_server_without_ech_sees_the_public_name_and_the_client_rejects_the_connection() -> Result<(), Error> {
  use alloc::sync::Arc;

  use rustls::client::EchConfig;
  use rustls::pki_types::{CertificateDer, EchConfigListBytes, ServerName, pem::PemObject};
  use rustls::{ClientConfig, Connection, RootCertStore};

  use super::tls::{self, Credential, FixtureTime, Peer};

  // Public output of OpenSSL 4.0.3: openssl ech -public_name localhost -suite 0x20,1,1.
  let list = EchConfigListBytes::from_pem_slice(b"-----BEGIN ECHCONFIG-----\nADz+DQA4pwAgACBV8rmx209HPUG0RJp4SbYNL+Hl9ICSCl+0E2LPOb7yHwAEAAEA\nAQAJbG9jYWxob3N0AAA=\n-----END ECHCONFIG-----\n").expect("ECHConfigList");
  let config = EchConfig::new(list, crate::hpke::ALL_SUPPORTED_SUITES)?;
  let provider = crate::default_provider();
  let (root, _, _) = tls::credentials(Credential::P256);
  let mut roots = RootCertStore::empty();
  roots.add(CertificateDer::from(root))?;
  let client = ClientConfig::builder_with_details(Arc::new(provider.clone()), Arc::new(FixtureTime))
    .with_ech(config.into())
    .with_root_certificates(roots)
    .with_no_client_auth()?;
  let (_, server) = tls::configs(provider, rustls_ring::DEFAULT_TLS13_PROVIDER, Credential::P256, false);
  let mut output = Vec::new();
  let conn = Arc::new(client)
    .connect(ServerName::try_from("private.example").expect("DNS name"))
    .build(&mut output)?;
  let mut client = Peer::new(conn, output);
  let mut server = Peer::new(Connection::new(Arc::new(server))?, Vec::new());
  assert!(matches!(tls::drive(&mut client, &mut server), Err(Error::RejectedEch(rejection)) if !rejection.can_retry()));
  assert_eq!(
    server.connection.side().server_name().map(AsRef::as_ref),
    Some("localhost")
  );
  Ok(())
}

#[cfg(feature = "chacha20-rng")]
#[test]
fn connection_scoped_entropy_seals_ech_but_refuses_a_shared_recipient_key() -> Result<(), Error> {
  use crate::replay::{ScopedRandom, Stream, scope};

  let suite = X25519HkdfSha256Aes128Gcm::new(&ScopedRandom);
  let (public, private) = PEER.generate_key_pair()?;
  let stream = Stream::new(rscrypto::ChaCha20Rng::from_seed([3; 32]));
  let (enc, ciphertext) = scope(&stream, || {
    assert!(matches!(suite.generate_key_pair(), Err(Error::FailedToGetRandomBytes)));
    suite.seal(b"info", b"outer", b"inner", &public)
  })?;
  assert_eq!(PEER.open(&enc, b"info", b"outer", &ciphertext, &private)?, b"inner");
  Ok(())
}
