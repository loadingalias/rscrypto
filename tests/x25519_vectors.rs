#![cfg(feature = "x25519")]

use rscrypto::{X25519PublicKey, X25519SecretKey, X25519SharedSecret};
use x25519_dalek::{PublicKey as DalekPublicKey, StaticSecret as DalekStaticSecret};

mod common;
#[path = "common/array.rs"]
mod hex_array;

fn decode_hex_32(hex: &str) -> [u8; 32] {
  hex_array::decode_hex_array::<32>(hex)
}

fn iterative_x25519(iterations: usize) -> [u8; 32] {
  let mut k = X25519SecretKey::from_bytes(X25519PublicKey::basepoint().to_bytes());
  let mut u = X25519PublicKey::basepoint();

  for _ in 0..iterations {
    let next = k
      .diffie_hellman(&u)
      .expect("RFC 7748 iterative ladder must produce a nonzero secret")
      .expose_secret()
      .expose();
    u = X25519PublicKey::from_bytes(*k.as_bytes());
    k = X25519SecretKey::from_bytes(next);
  }

  k.expose_secret().expose()
}

#[test]
fn rfc_7748_scalar_multiplication_vectors_match() {
  let scalar = X25519SecretKey::from_bytes(decode_hex_32(
    "a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4",
  ));
  let public = X25519PublicKey::from_bytes(decode_hex_32(
    "e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c",
  ));
  let expected = decode_hex_32("c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552");

  let shared = scalar
    .diffie_hellman(&public)
    .expect("first RFC 7748 scalar-multiplication vector must produce a nonzero secret");
  assert_eq!(*shared.as_bytes(), expected);

  let scalar = X25519SecretKey::from_bytes(decode_hex_32(
    "4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d",
  ));
  let public = X25519PublicKey::from_bytes(decode_hex_32(
    "e5210f12786811d3f4b7959d0538ae2c31dbe7106fc03c3efc4cd549c715a493",
  ));
  let expected = decode_hex_32("95cbde9476e8907d7aade45cb4b873f88b595a68799fa152e6f8f7647aac7957");

  let shared = scalar
    .diffie_hellman(&public)
    .expect("second RFC 7748 scalar-multiplication vector must produce a nonzero secret");
  assert_eq!(*shared.as_bytes(), expected);
}

#[test]
fn rfc_7748_iterated_ladder_vectors_match() {
  assert_eq!(
    iterative_x25519(1),
    decode_hex_32("422c8e7a6227d7bca1350b3e2bb7279f7897b87bb6854b783c60e80311ae3079")
  );
  assert_eq!(
    iterative_x25519(1_000),
    decode_hex_32("684cf59ba83309552800ef566f2f4d3c1c3887c49360e3875f2eb94d99532c51")
  );
}

#[test]
fn rfc_7748_diffie_hellman_vector_matches() {
  let alice = X25519SecretKey::from_bytes(decode_hex_32(
    "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a",
  ));
  let bob = X25519SecretKey::from_bytes(decode_hex_32(
    "5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb",
  ));
  let alice_public = X25519PublicKey::from_bytes(decode_hex_32(
    "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a",
  ));
  let bob_public = X25519PublicKey::from_bytes(decode_hex_32(
    "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f",
  ));
  let expected = decode_hex_32("4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742");

  let alice_shared = alice
    .diffie_hellman(&bob_public)
    .expect("Alice RFC 7748 exchange must produce a nonzero secret");
  let bob_shared = bob
    .diffie_hellman(&alice_public)
    .expect("Bob RFC 7748 exchange must produce a nonzero secret");

  assert_eq!(alice.public_key(), alice_public);
  assert_eq!(bob.public_key(), bob_public);
  assert_eq!(*alice_shared.as_bytes(), expected);
  assert!(alice_shared.ct_eq(&bob_shared).declassify());
}

#[test]
fn low_order_points_return_all_zero_error() {
  let secret = X25519SecretKey::from_bytes([0x42; X25519SecretKey::LENGTH]);
  let low_order = X25519PublicKey::from_bytes([0u8; X25519PublicKey::LENGTH]);

  secret
    .diffie_hellman(&low_order)
    .expect_err("X25519 must reject a low-order point");
  X25519SharedSecret::diffie_hellman(&secret, &low_order)
    .expect_err("X25519 shared-secret API must reject a low-order point");
}

#[test]
fn non_canonical_public_inputs_are_accepted_and_reduced() {
  let secret = X25519SecretKey::from_bytes([0x24; X25519SecretKey::LENGTH]);
  let public = X25519PublicKey::from_bytes(decode_hex_32(
    "edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
  ));

  secret
    .diffie_hellman(&public)
    .expect_err("u = p must reduce to zero and fail the all-zero check");
  assert_eq!(
    public.to_bytes(),
    decode_hex_32("edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f")
  );
}

#[test]
fn public_keys_and_shared_secrets_match_x25519_dalek() {
  for seed in 0u8..32 {
    let mut secret_bytes = [0u8; 32];
    for (index, byte) in secret_bytes.iter_mut().enumerate() {
      let index = u8::try_from(index).expect("32-byte key index must fit in one byte");
      *byte = seed.wrapping_mul(17).wrapping_add(index.wrapping_mul(29));
    }

    let ours_secret = X25519SecretKey::from_bytes(secret_bytes);
    let ours_public = ours_secret.public_key();

    let dalek_secret = DalekStaticSecret::from(secret_bytes);
    let dalek_public = DalekPublicKey::from(&dalek_secret);
    assert_eq!(ours_public.to_bytes(), dalek_public.to_bytes());

    let mut peer_bytes = [0u8; 32];
    for (index, byte) in peer_bytes.iter_mut().enumerate() {
      let index = u8::try_from(index).expect("32-byte peer-key index must fit in one byte");
      *byte = seed
        .wrapping_mul(23)
        .wrapping_add(index.wrapping_mul(11))
        .wrapping_add(5);
    }
    peer_bytes[31] |= 0x80;

    let ours_peer = X25519PublicKey::from_bytes(peer_bytes);
    let dalek_peer = DalekPublicKey::from(peer_bytes);
    let ours_shared = ours_secret.diffie_hellman(&ours_peer);
    let dalek_shared = dalek_secret.diffie_hellman(&dalek_peer).to_bytes();

    if dalek_shared.iter().all(|&byte| byte == 0) {
      ours_shared.expect_err("rscrypto must reject every all-zero X25519 shared secret");
    } else {
      let ours_shared = ours_shared.expect("nonzero x25519-dalek shared secret must be accepted");
      assert_eq!(*ours_shared.as_bytes(), dalek_shared);
    }
  }
}

#[test]
fn public_key_matches_basepoint_diffie_hellman() {
  for seed in 0u8..32 {
    let mut secret_bytes = [0u8; 32];
    for (index, byte) in secret_bytes.iter_mut().enumerate() {
      let index = u8::try_from(index).expect("32-byte key index must fit in one byte");
      *byte = seed.wrapping_mul(41).wrapping_add(index.wrapping_mul(7));
    }

    let secret = X25519SecretKey::from_bytes(secret_bytes);
    let public = secret.public_key();
    let via_ladder = secret
      .diffie_hellman(&X25519PublicKey::basepoint())
      .expect("basepoint exchange must produce a nonzero X25519 secret");

    assert_eq!(public.to_bytes(), *via_ladder.as_bytes());
  }
}

#[test]
fn secret_and_shared_debug_are_masked() {
  let secret = X25519SecretKey::from_bytes([0x42; 32]);
  let public = secret.public_key();
  let shared = secret
    .diffie_hellman(&public)
    .expect("self-derived public key must produce a nonzero X25519 secret");

  assert_eq!(format!("{secret:?}"), "X25519SecretKey(****)");
  assert_eq!(format!("{shared:?}"), "X25519SharedSecret(****)");
  assert!(format!("{public:?}").starts_with("X25519PublicKey("));
}
