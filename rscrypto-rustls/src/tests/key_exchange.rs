use rustls::Error;
use rustls::crypto::kx::NamedGroup;
use rustls::error::PeerMisbehaved;

#[cfg(feature = "getrandom")]
use super::{
  ALL_KX_GROUPS, MLKEM768, MLKEM1024, SECP256R1, SECP256R1MLKEM768, SECP384R1, SECP384R1MLKEM1024, X25519,
  X25519MLKEM768,
};
#[cfg(not(feature = "getrandom"))]
use caller::{
  ALL_KX_GROUPS, MLKEM768, MLKEM1024, SECP256R1, SECP256R1MLKEM768, SECP384R1, SECP384R1MLKEM1024, X25519,
  X25519MLKEM768,
};
use rustls::crypto::kx::{StartedKeyExchange, SupportedKxGroup};

#[test]
fn caller_entropy_reaches_x25519_key_generation() {
  use rustls::crypto::{GetRandomFailed, SecureRandom};

  #[derive(Debug)]
  struct Alice;
  impl SecureRandom for Alice {
    fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
      // RFC 7748 section 6.1, Alice's private input.
      output.copy_from_slice(&[
        0x77, 0x07, 0x6d, 0x0a, 0x73, 0x18, 0xa5, 0x7d, 0x3c, 0x16, 0xc1, 0x72, 0x51, 0xb2, 0x66, 0x45, 0xdf, 0x4c,
        0x2f, 0x87, 0xeb, 0xc0, 0x99, 0x2a, 0xb1, 0x77, 0xfb, 0xa5, 0x1d, 0xb9, 0x2c, 0x2a,
      ]);
      Ok(())
    }
  }
  let exchange = super::X25519Group::new(&Alice).start().expect("fixed RFC input");
  assert_eq!(
    exchange.pub_key(),
    &[
      0x85, 0x20, 0xf0, 0x09, 0x89, 0x30, 0xa7, 0x54, 0x74, 0x8b, 0x7d, 0xdc, 0xb4, 0x3e, 0xf7, 0x5a, 0x0d, 0xbf, 0x3a,
      0x0d, 0x26, 0x38, 0x1a, 0xf4, 0xeb, 0xa4, 0xa9, 0x8e, 0xaa, 0x9b, 0x4e, 0x6a,
    ]
  );
}

#[test]
fn group_inventory_and_both_roles_agree_with_independent_peers() {
  let aws = rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER;
  let ring = rustls_ring::DEFAULT_TLS13_PROVIDER;
  let expected = [
    (NamedGroup::X25519MLKEM768, 1216, 1120, 64),
    (NamedGroup::X25519, 32, 32, 32),
    (NamedGroup::secp256r1, 65, 65, 32),
    (NamedGroup::secp384r1, 97, 97, 48),
  ];
  assert_eq!(ALL_KX_GROUPS.len(), expected.len());
  for (ours, (name, client_len, server_len, secret_len)) in ALL_KX_GROUPS.iter().zip(expected) {
    assert_eq!(ours.name(), name);
    assert!(crate::tests::tls::provider().tls12_cipher_suites.is_empty());
    assert!(ours.ffdhe_group().is_none());
    assert_eq!(ours.fips(), rustls::pki_types::FipsStatus::Unvalidated);
    for peer in aws
      .kx_groups
      .iter()
      .chain(ring.kx_groups.iter())
      .filter(|p| p.name() == name)
    {
      for (client, server) in [(*ours, *peer), (*peer, *ours)] {
        let client = client.start().expect("key exchange");
        assert_eq!(client.pub_key().len(), client_len);
        assert_eq!(client.group(), name);
        assert!(client.ffdhe_group().is_none());
        let server = server.start_and_complete(client.pub_key()).expect("key exchange");
        assert_eq!(server.group, name);
        assert_eq!(server.pub_key.len(), server_len);
        assert_eq!(server.secret.secret_bytes().len(), secret_len);
        let agreed = client.into_single().complete(&server.pub_key).expect("key exchange");
        assert_eq!(agreed.secret_bytes(), server.secret.secret_bytes());
      }
    }
  }
}

#[test]
fn classical_groups_reject_malformed_and_small_order_peers() {
  let mut one = [0; 32];
  one[0] = 1;
  for peer in [&[][..], &[1; 31], &[1; 33], &[0; 32], &one] {
    assert!(matches!(
      X25519.start().expect("key exchange").into_single().complete(peer),
      Err(Error::PeerMisbehaved(PeerMisbehaved::InvalidKeyShare))
    ));
  }
  for (group, length) in [(SECP256R1, 65), (SECP384R1, 97)] {
    let mut off_curve = vec![0; length];
    off_curve[0] = 4;
    let mut noncanonical = vec![0xff; length];
    noncanonical[0] = 4;
    let valid = group.start().expect("key exchange").pub_key().to_vec();
    let mut compressed = valid[..length / 2 + 1].to_vec();
    compressed[0] = 2;
    for peer in [&[][..], &valid[..length - 1], &off_curve, &noncanonical, &compressed] {
      assert!(matches!(
        group.start().expect("key exchange").into_single().complete(peer),
        Err(Error::PeerMisbehaved(PeerMisbehaved::InvalidKeyShare))
      ));
    }
  }
}

#[test]
#[expect(
  clippy::panic,
  reason = "the selected hybrid group must return its component exchange"
)]
fn hybrid_layout_and_classical_fallback_match_rfc10024() {
  let client = X25519MLKEM768.start().expect("key exchange");
  let StartedKeyExchange::Hybrid(client) = client else {
    panic!("hybrid exchange")
  };
  let (name, classical) = client.component();
  assert_eq!(name, NamedGroup::X25519);
  assert_eq!(classical, &client.as_key_exchange().pub_key()[1184..]);
  let server = rustls_ring::kx_group::X25519
    .start_and_complete(classical)
    .expect("key exchange");
  let agreed = client.complete_component(&server.pub_key).expect("key exchange");
  assert_eq!(agreed.secret_bytes(), server.secret.secret_bytes());

  // The independent hybrid peer verifies component order in both shares and
  // in the 64-byte shared secret. This case distinguishes implicit rejection
  // of a well-formed altered ciphertext from an invalid classical share.
  let client = X25519MLKEM768.start().expect("key exchange");
  let mut server = rustls_aws_lc_rs::kx_group::X25519MLKEM768
    .start_and_complete(client.pub_key())
    .expect("key exchange");
  server.pub_key[0] ^= 1;
  let agreed = client.into_single().complete(&server.pub_key).expect("key exchange");
  assert_ne!(&agreed.secret_bytes()[..32], &server.secret.secret_bytes()[..32]);
  assert_eq!(&agreed.secret_bytes()[32..], &server.secret.secret_bytes()[32..]);
}

#[test]
fn hybrid_rejects_invalid_lengths_encapsulation_keys_and_classical_shares() {
  let client = X25519MLKEM768.start().expect("key exchange");
  for peer in [&[][..], &client.pub_key()[..1215], &[0; 1217]] {
    assert!(matches!(
      X25519MLKEM768.start_and_complete(peer),
      Err(Error::PeerMisbehaved(PeerMisbehaved::InvalidKeyShare))
    ));
  }
  let mut invalid = client.pub_key().to_vec();
  // Encode coefficient q = 3329, the first noncanonical field element.
  invalid[0] = 1;
  invalid[1] = (invalid[1] & 0xf0) | 0x0d;
  assert!(matches!(
    X25519MLKEM768.start_and_complete(&invalid),
    Err(Error::PeerMisbehaved(PeerMisbehaved::InvalidKeyShare))
  ));
  invalid.copy_from_slice(client.pub_key());
  invalid[1184..].fill(0);
  assert!(matches!(
    X25519MLKEM768.start_and_complete(&invalid),
    Err(Error::PeerMisbehaved(PeerMisbehaved::InvalidKeyShare))
  ));
  for server in [&[][..], &[0; 1119], &[0; 1121], &[0; 1120]] {
    assert!(matches!(
      X25519MLKEM768
        .start()
        .expect("key exchange")
        .into_single()
        .complete(server),
      Err(Error::PeerMisbehaved(PeerMisbehaved::InvalidKeyShare))
    ));
  }
}

#[cfg(all(
  feature = "hkdf",
  feature = "aes-gcm",
  feature = "chacha20poly1305",
  feature = "ecdsa-p256"
))]
#[test]
fn negotiation_uses_hybrid_fallback_and_hello_retry_request() {
  use crate::tests::tls;
  use rustls::HandshakeKind;

  let all = ALL_KX_GROUPS.to_vec();
  for (client_groups, server_groups, group, kind) in [
    (
      all.clone(),
      all.clone(),
      NamedGroup::X25519MLKEM768,
      HandshakeKind::Full,
    ),
    (all.clone(), vec![X25519], NamedGroup::X25519, HandshakeKind::Full),
    (
      all.clone(),
      vec![SECP256R1],
      NamedGroup::secp256r1,
      HandshakeKind::FullWithHelloRetryRequest,
    ),
    (
      all,
      vec![SECP384R1],
      NamedGroup::secp384r1,
      HandshakeKind::FullWithHelloRetryRequest,
    ),
    (
      vec![X25519, X25519MLKEM768],
      vec![X25519MLKEM768],
      NamedGroup::X25519MLKEM768,
      HandshakeKind::FullWithHelloRetryRequest,
    ),
  ] {
    let mut client_provider = crate::tests::tls::provider();
    client_provider.kx_groups = client_groups.into();
    let mut server_provider = crate::tests::tls::provider();
    server_provider.kx_groups = server_groups.into();
    let (client, server) = tls::configs(client_provider, server_provider, tls::Credential::P256, false);
    let (mut client, mut server) =
      tls::connections_from_configs(alloc::sync::Arc::new(client), alloc::sync::Arc::new(server));
    tls::drive(&mut client, &mut server).expect("key exchange");
    for side in [&*client, &*server] {
      assert_eq!(
        side.negotiated_key_exchange_group().expect("key exchange").name(),
        group
      );
      assert_eq!(side.handshake_kind(), Some(kind));
    }
    tls::application_exchange(&mut client, &mut server);
  }
}

/// rustls's own RFC 10024 composer over aws-lc-rs and ring components: an
/// independent implementation of the share layout and secret order for groups
/// that no pinned rustls provider ships (SecP384r1MLKEM1024), and a second
/// classical implementation for SecP256r1MLKEM768.
static PEER_SECP384R1MLKEM1024_AWS: rustls::crypto::kx::Hybrid = rustls::crypto::kx::Hybrid {
  classical: rustls_aws_lc_rs::kx_group::SECP384R1,
  post_quantum: rustls_aws_lc_rs::kx_group::MLKEM1024,
  name: NamedGroup::secp384r1MLKEM1024,
  layout: rustls::crypto::kx::HybridLayout {
    classical_share_len: 97,
    post_quantum_client_share_len: 1568,
    post_quantum_server_share_len: 1568,
    post_quantum_first: false,
  },
};
static PEER_SECP384R1MLKEM1024_RING: rustls::crypto::kx::Hybrid = rustls::crypto::kx::Hybrid {
  classical: rustls_ring::kx_group::SECP384R1,
  ..PEER_SECP384R1MLKEM1024_AWS
};
static PEER_SECP256R1MLKEM768_RING: rustls::crypto::kx::Hybrid = rustls::crypto::kx::Hybrid {
  classical: rustls_ring::kx_group::SECP256R1,
  post_quantum: rustls_aws_lc_rs::kx_group::MLKEM768,
  name: NamedGroup::secp256r1MLKEM768,
  layout: rustls::crypto::kx::HybridLayout {
    classical_share_len: 65,
    post_quantum_client_share_len: 1184,
    post_quantum_server_share_len: 1088,
    post_quantum_first: false,
  },
};

/// One opt-in group: ours, independent peers, and RFC lengths in bytes.
type OptInGroup = (
  &'static dyn SupportedKxGroup,
  Vec<&'static dyn SupportedKxGroup>,
  NamedGroup,
  usize, // client share
  usize, // server share
  usize, // shared secret
  usize, // classical share, first; zero for pure ML-KEM
);

fn opt_in_groups() -> [OptInGroup; 4] {
  [
    (
      SECP256R1MLKEM768,
      vec![
        rustls_aws_lc_rs::kx_group::SECP256R1MLKEM768,
        &PEER_SECP256R1MLKEM768_RING,
      ],
      NamedGroup::secp256r1MLKEM768,
      1249,
      1153,
      64,
      65,
    ),
    (
      SECP384R1MLKEM1024,
      vec![&PEER_SECP384R1MLKEM1024_AWS, &PEER_SECP384R1MLKEM1024_RING],
      NamedGroup::secp384r1MLKEM1024,
      1665,
      1665,
      80,
      97,
    ),
    (
      MLKEM768,
      vec![rustls_aws_lc_rs::kx_group::MLKEM768],
      NamedGroup::MLKEM768,
      1184,
      1088,
      32,
      0,
    ),
    (
      MLKEM1024,
      vec![rustls_aws_lc_rs::kx_group::MLKEM1024],
      NamedGroup::MLKEM1024,
      1568,
      1568,
      32,
      0,
    ),
  ]
}

#[test]
fn opt_in_groups_agree_with_independent_peers_in_both_roles() {
  for (ours, peers, name, client_len, server_len, secret_len, _) in opt_in_groups() {
    assert_eq!(ours.name(), name);
    assert!(ours.ffdhe_group().is_none());
    assert_eq!(ours.fips(), rustls::pki_types::FipsStatus::Unvalidated);
    // Opt-in: no default list offers these groups.
    assert!(ALL_KX_GROUPS.iter().all(|group| group.name() != name));
    for peer in peers {
      for (client, server) in [(ours, peer), (peer, ours)] {
        let client = client.start().expect("key exchange");
        assert_eq!(client.pub_key().len(), client_len);
        assert_eq!(client.group(), name);
        let server = server.start_and_complete(client.pub_key()).expect("key exchange");
        assert_eq!(server.group, name);
        assert_eq!(server.pub_key.len(), server_len);
        assert_eq!(server.secret.secret_bytes().len(), secret_len);
        let agreed = client.into_single().complete(&server.pub_key).expect("key exchange");
        assert_eq!(agreed.secret_bytes(), server.secret.secret_bytes());
      }
    }
  }
}

#[test]
#[expect(clippy::panic, reason = "each NIST-curve hybrid must return its component exchange")]
fn nist_hybrids_put_the_classical_component_first_per_rfc10024() {
  for (ours, peers, name, _, _, _, classical_len) in opt_in_groups() {
    if classical_len == 0 {
      continue;
    }
    let (classical_group, component_peer) = match name {
      NamedGroup::secp256r1MLKEM768 => (NamedGroup::secp256r1, rustls_ring::kx_group::SECP256R1),
      _ => (NamedGroup::secp384r1, rustls_ring::kx_group::SECP384R1),
    };
    let StartedKeyExchange::Hybrid(client) = ours.start().expect("key exchange") else {
      panic!("hybrid exchange")
    };
    let (component, share) = client.component();
    assert_eq!(component, classical_group);
    assert_eq!(share, &client.as_key_exchange().pub_key()[..classical_len]);
    // A classical-only server completes the separately offered component.
    let server = component_peer.start_and_complete(share).expect("key exchange");
    let agreed = client.complete_component(&server.pub_key).expect("key exchange");
    assert_eq!(agreed.secret_bytes(), server.secret.secret_bytes());

    // A well-formed altered ciphertext is rejected implicitly: the ECDHE secret
    // (first) still agrees and the ML-KEM secret (second) differs. This fixes
    // the ciphertext's position after the classical share and the secret order.
    let client = ours.start().expect("key exchange");
    let mut server = peers[0].start_and_complete(client.pub_key()).expect("key exchange");
    let secret_classical = server.secret.secret_bytes().len() - 32;
    server.pub_key[classical_len] ^= 1;
    let agreed = client.into_single().complete(&server.pub_key).expect("key exchange");
    assert_eq!(
      &agreed.secret_bytes()[..secret_classical],
      &server.secret.secret_bytes()[..secret_classical]
    );
    assert_ne!(
      &agreed.secret_bytes()[secret_classical..],
      &server.secret.secret_bytes()[secret_classical..]
    );
  }
}

#[test]
fn opt_in_groups_reject_invalid_lengths_keys_and_classical_shares() {
  fn invalid(result: Result<impl Sized, Error>) -> bool {
    matches!(result, Err(Error::PeerMisbehaved(PeerMisbehaved::InvalidKeyShare)))
  }
  for (ours, peers, name, client_len, server_len, _, classical_len) in opt_in_groups() {
    let client = ours.start().expect("key exchange");
    let share = client.pub_key().to_vec();
    for peer in [&[][..], &share[..client_len - 1], &[share.as_slice(), &[0]].concat()] {
      assert!(invalid(ours.start_and_complete(peer)), "{name:?} client share length");
    }
    // Encode coefficient q = 3329, the first noncanonical field element, at
    // the start of the encapsulation key (FIPS 203 section 7.2 modulus check).
    let mut bad_key = share.clone();
    bad_key[classical_len] = 1;
    bad_key[classical_len + 1] = (bad_key[classical_len + 1] & 0xf0) | 0x0d;
    assert!(invalid(ours.start_and_complete(&bad_key)), "{name:?} encapsulation key");
    if classical_len > 0 {
      // An off-curve or non-uncompressed classical share is illegal.
      let mut bad_classical = share.clone();
      bad_classical[1..classical_len].fill(0);
      assert!(invalid(ours.start_and_complete(&bad_classical)), "{name:?} classical");
      bad_classical.copy_from_slice(&share);
      bad_classical[0] = 2;
      assert!(invalid(ours.start_and_complete(&bad_classical)), "{name:?} compressed");
    }
    let server = peers[0].start_and_complete(&share).expect("key exchange");
    for reply in [
      &[][..],
      &server.pub_key[..server_len - 1],
      &[server.pub_key.as_slice(), &[0]].concat(),
    ] {
      let client = ours.start().expect("key exchange");
      assert!(
        invalid(client.into_single().complete(reply)),
        "{name:?} server share length"
      );
    }
    if classical_len > 0 {
      let mut bad_classical = server.pub_key;
      bad_classical[1..classical_len].fill(0);
      let client = ours.start().expect("key exchange");
      assert!(
        invalid(client.into_single().complete(&bad_classical)),
        "{name:?} server classical"
      );
    }
  }
}

#[test]
fn opt_in_groups_fail_on_entropy_after_validating_peer_keys() {
  use core::sync::atomic::{AtomicUsize, Ordering};
  use rustls::crypto::{GetRandomFailed, SecureRandom};

  #[derive(Debug)]
  struct Failure {
    calls: AtomicUsize,
    fail_at: usize,
  }
  impl SecureRandom for Failure {
    fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
      output.fill(0x42);
      if self.calls.fetch_add(1, Ordering::Relaxed) == self.fail_at {
        Err(GetRandomFailed)
      } else {
        Ok(())
      }
    }
  }
  type Constructor = fn(&'static dyn SecureRandom) -> Box<dyn SupportedKxGroup>;
  // (constructor, reference group, entropy requests per start and per server completion)
  let groups: [(Constructor, &'static dyn SupportedKxGroup, usize, usize); 4] = [
    (
      |random| Box::new(super::P256MlKem768Group::new(random)),
      SECP256R1MLKEM768,
      2,
      2,
    ),
    (
      |random| Box::new(super::P384MlKem1024Group::new(random)),
      SECP384R1MLKEM1024,
      2,
      2,
    ),
    (|random| Box::new(super::MlKem768Group::new(random)), MLKEM768, 1, 1),
    (|random| Box::new(super::MlKem1024Group::new(random)), MLKEM1024, 1, 1),
  ];
  for (construct, reference, start_requests, server_requests) in groups {
    for fail_at in 0..start_requests {
      let random: &'static Failure = Box::leak(Box::new(Failure {
        calls: AtomicUsize::new(0),
        fail_at,
      }));
      assert!(matches!(construct(random).start(), Err(Error::FailedToGetRandomBytes)));
      assert_eq!(random.calls.load(Ordering::Relaxed), fail_at + 1);
    }
    let client = reference.start().expect("key exchange");
    for fail_at in 0..server_requests {
      let random: &'static Failure = Box::leak(Box::new(Failure {
        calls: AtomicUsize::new(0),
        fail_at,
      }));
      let group = construct(random);
      assert!(matches!(
        group.start_and_complete(&client.pub_key()[1..]),
        Err(Error::PeerMisbehaved(PeerMisbehaved::InvalidKeyShare))
      ));
      assert_eq!(random.calls.load(Ordering::Relaxed), 0, "validation precedes entropy");
      assert!(matches!(
        group.start_and_complete(client.pub_key()),
        Err(Error::FailedToGetRandomBytes)
      ));
      assert_eq!(random.calls.load(Ordering::Relaxed), fail_at + 1);
    }
  }
}

#[cfg(all(
  feature = "hkdf",
  feature = "aes-gcm",
  feature = "chacha20poly1305",
  feature = "ecdsa-p256"
))]
#[test]
fn opt_in_groups_negotiate_tls13_with_independent_peers() {
  use crate::tests::tls;
  use rustls::HandshakeKind;

  for (ours, peers, name, ..) in opt_in_groups() {
    for peer in peers {
      for ours_is_client in [true, false] {
        let mut provider = tls::provider();
        provider.kx_groups = vec![ours].into();
        let mut peer_provider = rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER;
        peer_provider.kx_groups = vec![peer].into();
        let (client, server) = if ours_is_client {
          tls::configs(provider, peer_provider, tls::Credential::P256, false)
        } else {
          tls::configs(peer_provider, provider, tls::Credential::P256, false)
        };
        let (mut client, mut server) =
          tls::connections_from_configs(alloc::sync::Arc::new(client), alloc::sync::Arc::new(server));
        tls::drive(&mut client, &mut server).expect("handshake");
        for side in [&*client, &*server] {
          assert_eq!(side.negotiated_key_exchange_group().expect("group").name(), name);
          assert_eq!(side.handshake_kind(), Some(HandshakeKind::Full));
        }
        tls::application_exchange(&mut client, &mut server);
      }
    }
  }
  // A client that prefers the default hybrid reaches an opt-in-only server
  // through HelloRetryRequest.
  for (ours, _, name, ..) in opt_in_groups() {
    let client_provider = tls::provider();
    let mut server_provider = tls::provider();
    server_provider.kx_groups = vec![ours].into();
    let mut client_provider = client_provider;
    client_provider.kx_groups.to_mut().push(ours);
    let (client, server) = tls::configs(client_provider, server_provider, tls::Credential::P256, false);
    let (mut client, mut server) =
      tls::connections_from_configs(alloc::sync::Arc::new(client), alloc::sync::Arc::new(server));
    tls::drive(&mut client, &mut server).expect("handshake");
    for side in [&*client, &*server] {
      assert_eq!(side.negotiated_key_exchange_group().expect("group").name(), name);
      assert_eq!(side.handshake_kind(), Some(HandshakeKind::FullWithHelloRetryRequest));
    }
  }
}

/// The groups of [`super::ALL_KX_GROUPS`] and the opt-in groups over test entropy.
#[cfg(not(feature = "getrandom"))]
mod caller {
  use super::super::{
    MlKem768Group, MlKem1024Group, P256Group, P256MlKem768Group, P384Group, P384MlKem1024Group, X25519Group,
    X25519MlKem768Group,
  };
  use crate::tests::RANDOM;
  use rustls::crypto::kx::SupportedKxGroup;

  pub(super) static SECP256R1MLKEM768: &dyn SupportedKxGroup = &P256MlKem768Group::new(&RANDOM);
  pub(super) static SECP384R1MLKEM1024: &dyn SupportedKxGroup = &P384MlKem1024Group::new(&RANDOM);
  pub(super) static MLKEM768: &dyn SupportedKxGroup = &MlKem768Group::new(&RANDOM);
  pub(super) static MLKEM1024: &dyn SupportedKxGroup = &MlKem1024Group::new(&RANDOM);
  pub(super) static X25519MLKEM768: &dyn SupportedKxGroup = &X25519MlKem768Group::new(&RANDOM);
  pub(super) static X25519: &dyn SupportedKxGroup = &X25519Group::new(&RANDOM);
  pub(super) static SECP256R1: &dyn SupportedKxGroup = &P256Group::new(&RANDOM);
  pub(super) static SECP384R1: &dyn SupportedKxGroup = &P384Group::new(&RANDOM);
  pub(super) static ALL_KX_GROUPS: &[&dyn SupportedKxGroup] = &[X25519MLKEM768, X25519, SECP256R1, SECP384R1];
}
