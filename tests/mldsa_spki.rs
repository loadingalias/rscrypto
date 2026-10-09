//! RFC 9881 SubjectPublicKeyInfo import and export of ML-DSA public keys.
#![cfg(feature = "ml-dsa")]

use rscrypto::{MlDsa44, MlDsa44PublicKey, MlDsa65, MlDsa65PublicKey, MlDsa87, MlDsa87PublicKey, MlDsaKeyError};

const RFC9881_44: &[u8] = include_bytes!("../testdata/mldsa/rfc9881/mldsa44_spki.der");
const RFC9881_65: &[u8] = include_bytes!("../testdata/mldsa/rfc9881/mldsa65_spki.der");
const RFC9881_87: &[u8] = include_bytes!("../testdata/mldsa/rfc9881/mldsa87_spki.der");

/// DER contents of `id-ml-dsa-44`, 2.16.840.1.101.3.4.3.17 (RFC 9881 section 2).
const ID_ML_DSA_44: [u8; 9] = [0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x03, 0x11];
/// `id-hash-ml-dsa-44-with-sha512`, 2.16.840.1.101.3.4.3.32: HashML-DSA, which RFC 9881
/// section 8.3 excludes.
const ID_HASH_ML_DSA_44: [u8; 9] = [0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x03, 0x20];

/// RFC 9881 Appendix C derives every example key from this seed.
fn rfc9881_seed() -> [u8; 32] {
  core::array::from_fn(|index| u8::try_from(index).expect("seed index fits a byte"))
}

macro_rules! rfc9881_example {
  ($name:ident, $profile:ident, $public:ident, $fixture:ident) => {
    #[test]
    fn $name() {
      let (public, _) = $profile::keypair_from_seed(&rfc9881_seed()).expect("key generation");
      assert_eq!($public::SPKI_DER_LENGTH, $fixture.len());
      assert_eq!(public.to_spki_der().as_slice(), $fixture);
      assert_eq!($public::from_spki_der($fixture), Ok(public));
    }
  };
}

rfc9881_example!(rfc9881_example_44, MlDsa44, MlDsa44PublicKey, RFC9881_44);
rfc9881_example!(rfc9881_example_65, MlDsa65, MlDsa65PublicKey, RFC9881_65);
rfc9881_example!(rfc9881_example_87, MlDsa87, MlDsa87PublicKey, RFC9881_87);

/// Minimal DER encoding of one element whose contents are shorter than 64 KiB.
fn tlv(tag: u8, contents: &[u8]) -> Vec<u8> {
  let mut out = vec![tag];
  match contents.len() {
    len @ 0..0x80 => out.push(u8::try_from(len).expect("short length")),
    len @ 0x80..0x100 => out.extend([0x81, u8::try_from(len).expect("one-octet length")]),
    len => out.extend(
      [0x82]
        .into_iter()
        .chain(u16::try_from(len).expect("two-octet length").to_be_bytes()),
    ),
  }
  out.extend_from_slice(contents);
  out
}

fn spki(algorithm: &[u8], unused_bits: u8, key: &[u8]) -> Vec<u8> {
  let mut bits = vec![unused_bits];
  bits.extend_from_slice(key);
  let mut contents = tlv(0x30, algorithm);
  contents.extend(tlv(0x03, &bits));
  tlv(0x30, &contents)
}

#[test]
fn rejections_name_the_failing_component() {
  let (public, _) = MlDsa44::keypair_from_seed(&rfc9881_seed()).expect("key generation");
  let key = public.as_bytes().as_slice();
  let oid = tlv(0x06, &ID_ML_DSA_44);
  // The builder reproduces the RFC example, anchoring the variants below.
  assert_eq!(spki(&oid, 0, key), RFC9881_44);

  // RFC 8410 section 10.1 Ed25519 (no parameters) and the RSA-3072 fixture (NULL parameters).
  let ed25519 = [
    0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00, 0x19, 0xbf, 0x44, 0x09, 0x69, 0x84, 0xcd,
    0xfe, 0x85, 0x41, 0xba, 0xc1, 0x67, 0xdc, 0x3b, 0x96, 0xc8, 0x50, 0x86, 0xaa, 0x30, 0xb6, 0xb6, 0xcb, 0x0c, 0x5c,
    0x38, 0xad, 0x70, 0x31, 0x66, 0xe1,
  ];
  let rsa = include_bytes!("../testdata/rsa/fixtures/rsa3072_spki.der");
  let hash_ml_dsa = spki(&tlv(0x06, &ID_HASH_ML_DSA_44), 0, key);
  for other in [RFC9881_65, RFC9881_87, &ed25519, rsa, &hash_ml_dsa] {
    assert_eq!(
      MlDsa44PublicKey::from_spki_der(other),
      Err(MlDsaKeyError::UnsupportedAlgorithm)
    );
  }

  let mut key_too_long = key.to_vec();
  key_too_long.push(0);
  for wrong_length in [&key[..key.len() - 1], &key_too_long] {
    assert_eq!(
      MlDsa44PublicKey::from_spki_der(&spki(&oid, 0, wrong_length)),
      Err(MlDsaKeyError::InvalidPublicKey)
    );
  }

  let mut null_parameters = oid.clone();
  null_parameters.extend([0x05, 0x00]);
  let mut trailing = RFC9881_44.to_vec();
  trailing.push(0);
  let mut long_form = vec![0x30, 0x83, 0x00];
  long_form.extend_from_slice(&RFC9881_44[2..]);
  let mut indefinite = vec![0x30, 0x80];
  indefinite.extend_from_slice(&RFC9881_44[4..]);
  indefinite.extend([0, 0]);
  let mut set_tag = RFC9881_44.to_vec();
  set_tag[0] = 0x31;
  for malformed in [
    spki(&null_parameters, 0, key),
    spki(&oid, 1, key),
    trailing,
    long_form,
    indefinite,
    set_tag,
  ] {
    assert_eq!(
      MlDsa44PublicKey::from_spki_der(&malformed),
      Err(MlDsaKeyError::MalformedDer)
    );
  }
  for len in 0..RFC9881_44.len() {
    assert_eq!(
      MlDsa44PublicKey::from_spki_der(&RFC9881_44[..len]),
      Err(MlDsaKeyError::MalformedDer),
      "truncated to {len} bytes"
    );
  }
}

#[test]
fn every_header_byte_is_required() {
  let header_len = RFC9881_44.len() - MlDsa44PublicKey::LENGTH;
  let mut input = RFC9881_44.to_vec();
  for position in 0..header_len {
    let original = input[position];
    for value in (0..=u8::MAX).filter(|&value| value != original) {
      input[position] = value;
      assert!(
        MlDsa44PublicKey::from_spki_der(&input).is_err(),
        "byte {position} = {value:#04x} was accepted"
      );
    }
    input[position] = original;
  }
}

#[cfg(any(
  all(
    any(unix, windows),
    not(target_arch = "wasm32"),
    not(any(target_arch = "s390x", target_arch = "powerpc64"))
  ),
  all(
    target_arch = "powerpc64",
    target_endian = "little",
    target_os = "linux",
    target_env = "gnu"
  )
))]
#[test]
fn spki_matches_aws_lc_rs() {
  use aws_lc_rs::{
    encoding::AsDer,
    signature::{self as aws, KeyPair, PqdsaKeyPair},
  };

  macro_rules! exchange {
    ($profile:ident, $public:ident, $signing:ident) => {
      for seed in [[0; 32], [0x5a; 32], [0xff; 32]] {
        let (public, _) = $profile::keypair_from_seed(&seed).expect("key generation");
        let theirs = PqdsaKeyPair::from_seed(&aws::$signing, &seed).expect("AWS-LC key generation");
        let encoded = theirs.public_key().as_der().expect("AWS-LC SPKI export");
        assert_eq!(public.to_spki_der().as_slice(), encoded.as_ref());
        assert_eq!($public::from_spki_der(encoded.as_ref()), Ok(public));
      }
    };
  }

  exchange!(MlDsa44, MlDsa44PublicKey, ML_DSA_44_SIGNING);
  exchange!(MlDsa65, MlDsa65PublicKey, ML_DSA_65_SIGNING);
  exchange!(MlDsa87, MlDsa87PublicKey, ML_DSA_87_SIGNING);
}
