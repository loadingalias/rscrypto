#![cfg(feature = "p384-ecdh")]

use p384::elliptic_curve::sec1::ToSec1Point as _;
use rscrypto::{P384EphemeralSecret, P384PublicKey};

mod common;
use common::decode_hex_vec as decode;

#[derive(Clone, Copy)]
struct NistCavpVector<'a> {
  private: &'a str,
  own_x: &'a str,
  own_y: &'a str,
  peer_x: &'a str,
  peer_y: &'a str,
  shared: &'a str,
}

fn nist_cavp_vectors() -> Vec<NistCavpVector<'static>> {
  let corpus = include_str!("../testdata/auth/nist/KAS_ECC_CDH_PrimitiveTest_P-384.rsp");
  let mut vectors = Vec::new();
  for block in corpus.split("\n\n").filter(|block| block.starts_with("COUNT = ")) {
    let field = |name: &str| {
      block
        .lines()
        .find_map(|line| line.strip_prefix(name)?.strip_prefix(" = "))
        .expect(name)
    };
    vectors.push(NistCavpVector {
      private: field("dIUT"),
      own_x: field("QIUTx"),
      own_y: field("QIUTy"),
      peer_x: field("QCAVSx"),
      peer_y: field("QCAVSy"),
      shared: field("ZIUT"),
    });
  }
  assert_eq!(vectors.len(), 25, "complete NIST P-384 CAVP component corpus");
  vectors
}

fn array<const N: usize>(bytes: &[u8]) -> [u8; N] {
  bytes.try_into().expect("test vector has the required fixed width")
}

fn scalar_bytes(bytes: &[u8]) -> [u8; 48] {
  let first_significant = bytes.iter().position(|&byte| byte != 0).unwrap_or(bytes.len());
  let significant = &bytes[first_significant..];
  assert!(significant.len() <= 48, "test scalar exceeds the P-384 scalar width");
  let mut scalar = [0u8; 48];
  scalar[48usize.strict_sub(significant.len())..].copy_from_slice(significant);
  scalar
}

fn secret(bytes: [u8; 48]) -> P384EphemeralSecret {
  P384EphemeralSecret::try_generate_with(|candidate| {
    candidate.copy_from_slice(&bytes);
    Ok::<(), core::convert::Infallible>(())
  })
  .expect("test scalar must be canonical and nonzero")
}

fn sec1(x: &str, y: &str) -> [u8; 97] {
  let mut encoded = [0u8; 97];
  encoded[0] = 0x04;
  encoded[1..49].copy_from_slice(&decode(x));
  encoded[49..].copy_from_slice(&decode(y));
  encoded
}

fn rustcrypto_shared(private: &[u8; 48], public: &p384::PublicKey) -> [u8; 48] {
  let secret = p384::SecretKey::from_slice(private).expect("oracle scalar must parse");
  let shared = p384::ecdh::diffie_hellman(secret.to_nonzero_scalar(), public.as_affine());
  array(shared.raw_secret_bytes().as_slice())
}

#[test]
fn nist_cavp_component_vectors_match_public_keys_and_shared_secrets() {
  for vector in nist_cavp_vectors() {
    let private = array(&decode(vector.private));
    let ours = secret(private);
    assert_eq!(ours.public_key().to_sec1_bytes(), sec1(vector.own_x, vector.own_y));

    let peer = P384PublicKey::from_sec1_bytes(&sec1(vector.peer_x, vector.peer_y)).expect("NIST peer point must parse");
    assert_eq!(
      ours.diffie_hellman(&peer).as_bytes(),
      &array::<48>(&decode(vector.shared))
    );
  }
}

#[test]
fn portable_authority_matches_rustcrypto_across_scalar_edges() {
  let mut one = [0u8; 48];
  one[47] = 1;
  let mut two = [0u8; 48];
  two[47] = 2;
  let mut order_minus_one = [0xffu8; 48];
  order_minus_one[24..].copy_from_slice(&decode("c7634d81f4372ddf581a0db248b0a77aecec196accc52972"));
  let mut order_minus_two = order_minus_one;
  order_minus_two[47] = 0x71;
  let mut high_bit_only = [0u8; 48];
  high_bit_only[0] = 0x80;
  let mut top_digit_carry = [0u8; 48];
  top_digit_carry[0] = 0x0f;
  top_digit_carry[1..].fill(0xf8);
  let scalars = [
    one,
    two,
    [0x11; 48],
    [0x42; 48],
    [0x7f; 48],
    [0xa5; 48],
    high_bit_only,
    top_digit_carry,
    order_minus_two,
    order_minus_one,
  ];

  for &left in &scalars {
    let oracle_left = p384::SecretKey::from_slice(&left).expect("oracle scalar must parse");
    assert_eq!(
      secret(left).public_key().as_sec1_bytes().as_slice(),
      oracle_left.public_key().to_sec1_point(false).as_bytes()
    );

    for &right in &scalars {
      let oracle_right = p384::SecretKey::from_slice(&right).expect("oracle scalar must parse");
      let peer = secret(right).public_key();
      let ours = secret(left).diffie_hellman(&peer);
      assert_eq!(
        ours.as_bytes(),
        &rustcrypto_shared(&left, &oracle_right.public_key()),
        "scalars {left:02x?} * {right:02x?}"
      );
    }
  }
}

#[test]
fn complete_wycheproof_ecpoint_corpus_obeys_the_canonical_sec1_contract() {
  let document: serde_json::Value = serde_json::from_str(include_str!(
    "../testdata/auth/wycheproof/ecdh_secp384r1_ecpoint_test.json"
  ))
  .expect("pinned Wycheproof corpus must parse");
  assert_eq!(document["numberOfTests"], 790);

  let mut valid = 0usize;
  for group in document["testGroups"].as_array().expect("test groups") {
    assert_eq!(group["curve"], "secp384r1");
    assert_eq!(group["encoding"], "ecpoint");
    for case in group["tests"].as_array().expect("test cases") {
      let id = case["tcId"].as_u64().expect("test case id");
      let public_bytes = decode(case["public"].as_str().expect("public encoding"));
      let oracle_public = p384::PublicKey::from_sec1_bytes(&public_bytes);
      let canonical = public_bytes.len() == 97 && public_bytes.first() == Some(&0x04) && oracle_public.is_ok();
      let ours = P384PublicKey::from_sec1_bytes(&public_bytes);
      assert_eq!(ours.is_ok(), canonical, "Wycheproof tcId {id}: parser/oracle mismatch");

      if case["result"] == "valid" {
        valid = valid.strict_add(1);
        let peer = ours.expect("Wycheproof valid point must parse");
        let private = scalar_bytes(&decode(case["private"].as_str().expect("private scalar")));
        let expected = array::<48>(&decode(case["shared"].as_str().expect("shared secret")));
        let shared = secret(private).diffie_hellman(&peer);
        assert_eq!(
          shared.as_bytes(),
          &expected,
          "Wycheproof tcId {id}: shared secret mismatch"
        );
        assert_eq!(
          shared.as_bytes(),
          &rustcrypto_shared(&private, &oracle_public.expect("valid Wycheproof oracle point")),
          "Wycheproof tcId {id}: oracle mismatch"
        );
      }
    }
  }
  assert!(valid > 0, "Wycheproof corpus must exercise agreement");
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn ring_independent_implementation_agrees_with_rscrypto() {
  let rng = ring::rand::SystemRandom::new();
  for scalar in [[0x42; 48], [0x7f; 48], [0xa5; 48]] {
    let ours = secret(scalar);
    let ours_public = ours.public_key();
    let ring_secret = ring::agreement::EphemeralPrivateKey::generate(&ring::agreement::ECDH_P384, &rng)
      .expect("ring must generate a P-384 scalar");
    let ring_public = ring_secret
      .compute_public_key()
      .expect("ring must derive a P-384 public key");
    let peer = P384PublicKey::from_sec1_bytes(ring_public.as_ref()).expect("ring public key must parse");
    let ours_shared = ours.diffie_hellman(&peer);

    let ring_peer = ring::agreement::UnparsedPublicKey::new(&ring::agreement::ECDH_P384, ours_public.as_sec1_bytes());
    let ring_shared = ring::agreement::agree_ephemeral(ring_secret, &ring_peer, array)
      .expect("ring must accept the rscrypto public key");
    assert_eq!(ours_shared.as_bytes(), &ring_shared);
  }
}
