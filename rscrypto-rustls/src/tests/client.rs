use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use rustls::crypto::kx::NamedGroup;
use rustls::crypto::{GetRandomFailed, SecureRandom};
use rustls::pki_types::{PrivateKeyDer, PrivatePkcs8KeyDer, ServerName};
use rustls::{Error, HandshakeKind};

use super::tls::{self, Credential};
use crate::ClientProvider;

#[derive(Debug)]
struct Counting {
  calls: AtomicUsize,
  failing: AtomicBool,
}

impl SecureRandom for Counting {
  fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
    self.calls.fetch_add(1, Ordering::Relaxed);
    if self.failing.load(Ordering::Relaxed) {
      output.fill(0x42);
      return Err(GetRandomFailed);
    }
    super::RANDOM.fill(output)
  }
}

static SOURCE: Counting = Counting {
  calls: AtomicUsize::new(0),
  failing: AtomicBool::new(false),
};
static CLIENT: ClientProvider = ClientProvider::new(&SOURCE);

fn credential() -> Credential {
  [
    #[cfg(feature = "ecdsa-p256")]
    Credential::P256,
    #[cfg(feature = "ecdsa-p384")]
    Credential::P384,
    #[cfg(feature = "ed25519")]
    Credential::Ed25519,
    #[cfg(feature = "rsa")]
    Credential::Rsa,
  ][0]
}

#[test]
fn client_provider_offers_the_catalog_and_uses_only_the_caller_source() {
  let provider = CLIENT.provider();
  assert!(provider.tls12_cipher_suites.is_empty());
  assert_eq!(
    provider
      .tls13_cipher_suites
      .iter()
      .map(|suite| suite.common.suite)
      .collect::<Vec<_>>(),
    crate::cipher_suite::ALL_CIPHER_SUITES
      .iter()
      .map(|suite| suite.common.suite)
      .collect::<Vec<_>>()
  );
  let expected = [
    #[cfg(all(feature = "x25519", feature = "ml-kem"))]
    NamedGroup::X25519MLKEM768,
    #[cfg(feature = "x25519")]
    NamedGroup::X25519,
    #[cfg(feature = "p256-ecdh")]
    NamedGroup::secp256r1,
    #[cfg(feature = "p384-ecdh")]
    NamedGroup::secp384r1,
  ];
  assert_eq!(
    provider.kx_groups.iter().map(|group| group.name()).collect::<Vec<_>>(),
    expected
  );
  #[cfg(feature = "getrandom")]
  assert_eq!(
    crate::kx_group::ALL_KX_GROUPS
      .iter()
      .map(|group| group.name())
      .collect::<Vec<_>>(),
    expected
  );
  assert_eq!(
    provider.signature_verification_algorithms.supported_schemes(),
    crate::verify::SUPPORTED_ALGORITHMS.supported_schemes()
  );

  // Each group, against independent servers, draws entropy only from the caller.
  for peer in [
    rustls_ring::DEFAULT_TLS13_PROVIDER,
    rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER,
  ] {
    for group in CLIENT.provider().kx_groups.iter() {
      if !peer.kx_groups.iter().any(|candidate| candidate.name() == group.name()) {
        continue; // ring has no hybrid group.
      }
      let mut client = CLIENT.provider();
      client
        .kx_groups
        .to_mut()
        .retain(|candidate| candidate.name() == group.name());
      let (client, server) = tls::configs(client, peer.clone(), credential(), false);
      // The client hello, with its random and key share, is built with the connection.
      let calls = SOURCE.calls.load(Ordering::Relaxed);
      let (mut client, mut server) = tls::connections_from_configs(Arc::new(client), Arc::new(server));
      tls::drive(&mut client, &mut server).expect("client-provider handshake");
      assert_eq!(client.handshake_kind(), Some(HandshakeKind::Full));
      assert!(SOURCE.calls.load(Ordering::Relaxed) > calls);
      tls::application_exchange(&mut client, &mut server);
    }
  }

  // A failing source fails the client hello; nothing falls back to another source.
  let (client, _) = tls::configs(
    CLIENT.provider(),
    rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER,
    credential(),
    false,
  );
  SOURCE.failing.store(true, Ordering::Relaxed);
  let result = Arc::new(client)
    .connect(ServerName::try_from("localhost").expect("name"))
    .build(&mut Vec::new());
  SOURCE.failing.store(false, Ordering::Relaxed);
  assert!(matches!(result, Err(Error::FailedToGetRandomBytes)));
}

#[test]
fn client_provider_loads_no_keys_and_builds_no_ticketers() {
  let provider = CLIENT.provider();
  let (_, _, key) = tls::credentials(credential());
  assert!(matches!(
    provider
      .key_provider
      .load_private_key(PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key.to_vec()))),
    Err(Error::General(_))
  ));
  assert!(matches!(provider.ticketer_factory.ticketer(), Err(Error::General(_))));
}
