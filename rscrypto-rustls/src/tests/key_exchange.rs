use rustls::{Error, NamedGroup, PeerMisbehaved, ProtocolVersion};

use super::{ALL_KX_GROUPS, SECP256R1, SECP384R1, X25519, X25519MLKEM768};

#[test]
fn group_inventory_and_both_roles_agree_with_independent_peers() {
  let aws = rustls::crypto::aws_lc_rs::default_provider();
  let ring = rustls::crypto::ring::default_provider();
  let expected = [
    (NamedGroup::X25519MLKEM768, 1216, 1120, 64),
    (NamedGroup::X25519, 32, 32, 32),
    (NamedGroup::secp256r1, 65, 65, 32),
    (NamedGroup::secp384r1, 97, 97, 48),
  ];
  assert_eq!(ALL_KX_GROUPS.len(), expected.len());
  for (ours, (name, client_len, server_len, secret_len)) in ALL_KX_GROUPS.iter().zip(expected) {
    assert_eq!(ours.name(), name);
    assert!(ours.usable_for_version(ProtocolVersion::TLSv1_3));
    assert!(!ours.usable_for_version(ProtocolVersion::TLSv1_2));
    assert!(ours.ffdhe_group().is_none());
    assert!(!ours.fips());
    for peer in aws.kx_groups.iter().chain(&ring.kx_groups).filter(|p| p.name() == name) {
      for (client, server) in [(*ours, *peer), (*peer, *ours)] {
        let client = client.start().expect("key exchange");
        assert_eq!(client.pub_key().len(), client_len);
        assert_eq!(client.group(), name);
        assert!(client.ffdhe_group().is_none());
        let server = server.start_and_complete(client.pub_key()).expect("key exchange");
        assert_eq!(server.group, name);
        assert_eq!(server.pub_key.len(), server_len);
        assert_eq!(server.secret.secret_bytes().len(), secret_len);
        let agreed = client.complete(&server.pub_key).expect("key exchange");
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
      X25519.start().expect("key exchange").complete(peer),
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
        group.start().expect("key exchange").complete(peer),
        Err(Error::PeerMisbehaved(PeerMisbehaved::InvalidKeyShare))
      ));
    }
  }
}

#[test]
fn hybrid_layout_and_classical_fallback_match_rfc10024() {
  let client = X25519MLKEM768.start().expect("key exchange");
  let (name, classical) = client.hybrid_component().expect("classical component");
  assert_eq!(name, NamedGroup::X25519);
  assert_eq!(classical, &client.pub_key()[1184..]);
  let server = rustls::crypto::ring::kx_group::X25519
    .start_and_complete(classical)
    .expect("key exchange");
  let agreed = client.complete_hybrid_component(&server.pub_key).expect("key exchange");
  assert_eq!(agreed.secret_bytes(), server.secret.secret_bytes());

  // The independent hybrid peer verifies component order in both shares and
  // in the 64-byte shared secret. This case distinguishes implicit rejection
  // of a well-formed altered ciphertext from an invalid classical share.
  let client = X25519MLKEM768.start().expect("key exchange");
  let mut server = rustls::crypto::aws_lc_rs::kx_group::X25519MLKEM768
    .start_and_complete(client.pub_key())
    .expect("key exchange");
  server.pub_key[0] ^= 1;
  let agreed = client.complete(&server.pub_key).expect("key exchange");
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
      X25519MLKEM768.start().expect("key exchange").complete(server),
      Err(Error::PeerMisbehaved(PeerMisbehaved::InvalidKeyShare))
    ));
  }
}

#[cfg(all(feature = "hkdf", feature = "aes-gcm", feature = "chacha20poly1305"))]
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
    let mut client_provider = tls::symmetric_provider();
    client_provider.kx_groups = client_groups;
    let mut server_provider = tls::symmetric_provider();
    server_provider.kx_groups = server_groups;
    let (mut client, mut server) = tls::connections(client_provider, server_provider);
    tls::drive(&mut client, &mut server).expect("key exchange");
    for side in [&client, &server] {
      assert_eq!(
        side.negotiated_key_exchange_group().expect("key exchange").name(),
        group
      );
      assert_eq!(side.handshake_kind(), Some(kind));
    }
    tls::application_exchange(&mut client, &mut server);
  }
}
