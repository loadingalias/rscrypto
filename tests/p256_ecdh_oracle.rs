#![cfg(feature = "p256-ecdh")]

use p256::elliptic_curve::sec1::ToSec1Point as _;
use rscrypto::{P256EphemeralSecret, P256PublicKey};

const NIST_CAVP_P256_ANCHORS: [(&str, &str, &str, &str, &str, &str); 3] = [
  (
    "7d7dc5f71eb29ddaf80d6214632eeae03d9058af1fb6d22ed80badb62bc1a534",
    "ead218590119e8876b29146ff89ca61770c4edbbf97d38ce385ed281d8a6b230",
    "28af61281fd35e2fa7002523acc85a429cb06ee6648325389f59edfce1405141",
    "700c48f77f56584c5cc632ca65640db91b6bacce3a4df6b42ce7cc838833d287",
    "db71e509e3fd9b060ddb20ba5c51dcc5948d46fbf640dfe0441782cab85fa4ac",
    "46fc62106420ff012e54a434fbdd2d25ccc5852060561e68040dd7778997bd7b",
  ),
  (
    "38f65d6dce47676044d58ce5139582d568f64bb16098d179dbab07741dd5caf5",
    "119f2f047902782ab0c9e27a54aff5eb9b964829ca99c06b02ddba95b0a3f6d0",
    "8f52b726664cac366fc98ac7a012b2682cbd962e5acb544671d41b9445704d1d",
    "809f04289c64348c01515eb03d5ce7ac1a8cb9498f5caa50197e58d43a86a7ae",
    "b29d84e811197f25eba8f5194092cb6ff440e26d4421011372461f579271cda3",
    "057d636096cb80b67a8c038c890e887d1adfa4195e9b3ce241c8a778c59cda67",
  ),
  (
    "1accfaf1b97712b85a6f54b148985a1bdc4c9bec0bd258cad4b3d603f49f32c8",
    "d9f2b79c172845bfdb560bbb01447ca5ecc0470a09513b6126902c6b4f8d1051",
    "f815ef5ec32128d3487834764678702e64e164ff7315185e23aff5facd96d7bc",
    "a2339c12d4a03c33546de533268b4ad667debf458b464d77443636440ee7fec3",
    "ef48a3ab26e20220bcda2c1851076839dae88eae962869a497bf73cb66faf536",
    "2d457b78b4614132477618a5b077965ec90730a8c81a1c75d6d4ec68005d67ec",
  ),
];

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
  let corpus = include_str!("../testdata/auth/nist/KAS_ECC_CDH_PrimitiveTest_P-256.rsp");
  let mut vectors = Vec::new();
  for block in corpus.split("\n\n").filter(|block| block.starts_with("COUNT = ")) {
    let mut peer_x = None;
    let mut peer_y = None;
    let mut private = None;
    let mut own_x = None;
    let mut own_y = None;
    let mut shared = None;
    for line in block.lines().skip(1) {
      let Some((name, value)) = line.split_once(" = ") else {
        continue;
      };
      assert!(
        matches!(name, "QCAVSx" | "QCAVSy" | "dIUT" | "QIUTx" | "QIUTy" | "ZIUT"),
        "unexpected NIST P-256 CAVP field: {name}"
      );
      match name {
        "QCAVSx" => peer_x = Some(value),
        "QCAVSy" => peer_y = Some(value),
        "dIUT" => private = Some(value),
        "QIUTx" => own_x = Some(value),
        "QIUTy" => own_y = Some(value),
        "ZIUT" => shared = Some(value),
        _ => {}
      }
    }
    vectors.push(NistCavpVector {
      private: private.expect("NIST vector private scalar"),
      own_x: own_x.expect("NIST vector own x-coordinate"),
      own_y: own_y.expect("NIST vector own y-coordinate"),
      peer_x: peer_x.expect("NIST vector peer x-coordinate"),
      peer_y: peer_y.expect("NIST vector peer y-coordinate"),
      shared: shared.expect("NIST vector shared secret"),
    });
  }
  assert_eq!(vectors.len(), 25, "complete NIST P-256 CAVP component corpus");
  vectors
}

fn decode(hex: &str) -> Vec<u8> {
  assert_eq!(hex.len() % 2, 0, "hex input must contain complete bytes");
  let (pairs, remainder) = hex.as_bytes().as_chunks::<2>();
  assert!(remainder.is_empty(), "hex input must contain complete bytes");
  pairs
    .iter()
    .map(|pair| {
      let high = char::from(pair[0]).to_digit(16).expect("hex high nibble");
      let low = char::from(pair[1]).to_digit(16).expect("hex low nibble");
      u8::try_from((high << 4) | low).expect("decoded hex byte")
    })
    .collect()
}

fn array<const N: usize>(bytes: &[u8]) -> [u8; N] {
  bytes.try_into().expect("test vector has the required fixed width")
}

fn scalar_bytes(bytes: &[u8]) -> [u8; 32] {
  let first_significant = bytes.iter().position(|&byte| byte != 0).unwrap_or(bytes.len());
  let significant = &bytes[first_significant..];
  assert!(significant.len() <= 32, "test scalar exceeds the P-256 scalar width");
  let mut scalar = [0u8; 32];
  scalar[32usize.strict_sub(significant.len())..].copy_from_slice(significant);
  scalar
}

fn secret(bytes: [u8; 32]) -> P256EphemeralSecret {
  P256EphemeralSecret::try_generate_with(|candidate| {
    candidate.copy_from_slice(&bytes);
    Ok::<(), core::convert::Infallible>(())
  })
  .expect("test scalar must be canonical and nonzero")
}

fn crrl_scalar(bytes: [u8; 32]) -> crrl::p256::Scalar {
  let mut little_endian = bytes;
  little_endian.reverse();
  crrl::p256::Scalar::decode(&little_endian).expect("test scalar must be valid for CRRL")
}

fn sec1(x: &str, y: &str) -> [u8; 65] {
  let mut encoded = [0u8; 65];
  encoded[0] = 0x04;
  encoded[1..33].copy_from_slice(&decode(x));
  encoded[33..].copy_from_slice(&decode(y));
  encoded
}

#[test]
fn nist_cavp_component_vectors_match_public_keys_and_shared_secrets() {
  let vectors = nist_cavp_vectors();
  for vector in &vectors {
    let private = array(&decode(vector.private));
    let ours = secret(private);
    assert_eq!(ours.public_key().to_sec1_bytes(), sec1(vector.own_x, vector.own_y));

    let peer = P256PublicKey::from_sec1_bytes(&sec1(vector.peer_x, vector.peer_y)).expect("NIST peer point must parse");
    assert_eq!(
      ours.diffie_hellman(&peer).as_bytes(),
      &array::<32>(&decode(vector.shared))
    );
  }

  for (private, own_x, own_y, peer_x, peer_y, shared) in NIST_CAVP_P256_ANCHORS {
    assert!(vectors.iter().any(|vector| {
      (
        vector.private,
        vector.own_x,
        vector.own_y,
        vector.peer_x,
        vector.peer_y,
        vector.shared,
      ) == (private, own_x, own_y, peer_x, peer_y, shared)
    }));
  }
}

#[test]
fn nist_cavp_component_vectors_match_independent_pure_rust_implementations() {
  for vector in nist_cavp_vectors() {
    let private = array(&decode(vector.private));
    let expected_public = sec1(vector.own_x, vector.own_y);
    let peer = sec1(vector.peer_x, vector.peer_y);
    let expected_shared = array::<32>(&decode(vector.shared));

    let scalar = crrl_scalar(private);
    assert_eq!(
      crrl::p256::Point::mulgen(&scalar).encode_uncompressed(),
      expected_public
    );
    let crrl_peer = crrl::p256::Point::decode(&peer).expect("NIST peer point must parse in CRRL");
    assert_eq!(&(crrl_peer * scalar).encode_uncompressed()[1..33], &expected_shared);

    let mut libcrux_public = [0u8; 64];
    assert!(libcrux_p256::dh_initiator(&mut libcrux_public, &private));
    assert_eq!(libcrux_public, expected_public[1..]);
    let mut libcrux_shared = [0u8; 64];
    assert!(libcrux_p256::dh_responder(&mut libcrux_shared, &peer[1..], &private));
    assert_eq!(libcrux_shared[..32], expected_shared);
  }
}

#[test]
fn leading_zero_and_all_zero_shared_coordinates_retain_the_full_width() {
  let private = array(&decode(
    "0a0d622a47e48f6bc1038ace438c6f528aa00ad2bd1da5f13ee46bf5f633d71a",
  ));
  let cases = [
    (
      "0458fd4168a87795603e2b04390285bdca6e57de6027fe211dd9d25e2212d29e62080d36bd224d7405509295eed02a17150e03b314f96da37445b0d1d29377d12c",
      "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
      "04924fb33985c8a687fc04c9dd05e531ca0e0223aa58d58351e922ef482043d30cf504745e769b6dcbefe404da37f717b3109d2af23450fcfe2f075c2dabbe7194",
      "00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00f9",
    ),
  ];

  for (public, expected) in cases {
    let peer = P256PublicKey::from_sec1_bytes(&decode(public)).expect("Wycheproof public key must parse");
    let shared = secret(private).diffie_hellman(&peer);
    assert_eq!(shared.as_bytes(), &array::<32>(&decode(expected)));
  }
}

#[test]
fn portable_authority_matches_rustcrypto_for_public_derivation_and_agreement() {
  let scalars = [
    [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ],
    [0x11; 32],
    [0x42; 32],
    [0x7f; 32],
    [0xa5; 32],
  ];

  for &left in &scalars {
    let oracle_left = p256::SecretKey::from_slice(&left).expect("oracle scalar must parse");
    let expected_public = oracle_left.public_key().to_sec1_point(false);
    assert_eq!(
      secret(left).public_key().as_sec1_bytes().as_slice(),
      expected_public.as_bytes()
    );

    for &right in &scalars {
      let oracle_right = p256::SecretKey::from_slice(&right).expect("oracle scalar must parse");
      let oracle_public = oracle_right.public_key();
      let oracle_shared = p256::ecdh::diffie_hellman(oracle_left.to_nonzero_scalar(), oracle_public.as_affine());
      let peer = secret(right).public_key();
      let ours = secret(left).diffie_hellman(&peer);
      assert_eq!(ours.as_bytes().as_slice(), oracle_shared.raw_secret_bytes().as_slice());
    }
  }
}

#[test]
fn complete_wycheproof_ecpoint_corpus_obeys_the_canonical_sec1_contract() {
  let document: serde_json::Value = serde_json::from_str(include_str!(
    "../testdata/auth/wycheproof/ecdh_secp256r1_ecpoint_test.json"
  ))
  .expect("pinned Wycheproof corpus must parse");
  assert_eq!(document["numberOfTests"], 355);

  for group in document["testGroups"].as_array().expect("test groups") {
    assert_eq!(group["curve"], "secp256r1");
    assert_eq!(group["encoding"], "ecpoint");
    for case in group["tests"].as_array().expect("test cases") {
      let id = case["tcId"].as_u64().expect("test case id");
      let public_bytes = decode(case["public"].as_str().expect("public encoding"));
      let oracle_public = p256::PublicKey::from_sec1_bytes(&public_bytes);
      let canonical = public_bytes.len() == 65 && public_bytes.first() == Some(&0x04) && oracle_public.is_ok();
      let ours = P256PublicKey::from_sec1_bytes(&public_bytes);
      assert_eq!(ours.is_ok(), canonical, "Wycheproof tcId {id}: parser/oracle mismatch");

      if case["result"] == "valid" {
        let peer = ours.expect("Wycheproof valid point must parse");
        let private = scalar_bytes(&decode(case["private"].as_str().expect("private scalar")));
        let expected = array::<32>(&decode(case["shared"].as_str().expect("shared secret")));
        let shared = secret(private).diffie_hellman(&peer);
        assert_eq!(
          shared.as_bytes(),
          &expected,
          "Wycheproof tcId {id}: shared secret mismatch"
        );

        let oracle_secret = p256::SecretKey::from_slice(&private).expect("valid Wycheproof scalar");
        let oracle_shared = p256::ecdh::diffie_hellman(
          oracle_secret.to_nonzero_scalar(),
          oracle_public.expect("valid Wycheproof oracle point").as_affine(),
        );
        assert_eq!(
          shared.as_bytes().as_slice(),
          oracle_shared.raw_secret_bytes().as_slice()
        );
      }
    }
  }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn ring_independent_implementation_agrees_with_rscrypto() {
  let ours = secret([0x42; 32]);
  let ours_public = ours.public_key();
  let rng = ring::rand::SystemRandom::new();
  let ring_secret = ring::agreement::EphemeralPrivateKey::generate(&ring::agreement::ECDH_P256, &rng)
    .expect("ring must generate a P-256 scalar");
  let ring_public = ring_secret
    .compute_public_key()
    .expect("ring must derive a P-256 public key");
  let peer = P256PublicKey::from_sec1_bytes(ring_public.as_ref()).expect("ring public key must parse");
  let ours_shared = ours.diffie_hellman(&peer);

  let ring_peer = ring::agreement::UnparsedPublicKey::new(&ring::agreement::ECDH_P256, ours_public.as_sec1_bytes());
  let ring_shared =
    ring::agreement::agree_ephemeral(ring_secret, &ring_peer, array).expect("ring must accept the rscrypto public key");
  assert_eq!(ours_shared.as_bytes(), &ring_shared);
}
