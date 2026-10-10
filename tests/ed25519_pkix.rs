//! RFC 8410 SubjectPublicKeyInfo and RFC 5958 PKCS #8 import and export of
//! Ed25519 keys.
#![cfg(feature = "ed25519")]

use ring::{rand::SystemRandom, signature as ring_signature, signature::KeyPair as _};
use rscrypto::{Ed25519KeyError, Ed25519PublicKey, Ed25519SecretKey};

mod common;
use common::decode_hex_vec;

/// RFC 8410 section 10.1 public key.
const RFC8410_SPKI: &str = "302a300506032b657003210019bf44096984cdfe8541bac167dc3b96c85086aa30b6b6cb0c5c38ad703166e1";
/// RFC 8410 section 10.3 private key, version 1.
const RFC8410_PKCS8: &str =
  "302e020100300506032b657004220420d4ee72dbf913584ad5b6d8f1f769f8ad3afe7c28cbf1d4fbe097a88f44755842";
/// RFC 8410 section 10.3 private key, version 2 with an attribute and the
/// section 10.1 public key.
const RFC8410_PKCS8_ATTRIBUTES: &str = "3072020101300506032b657004220420d4ee72dbf913584ad5b6d8f1f769f8ad3afe7c28cbf1d4fbe097a88f44755842a01f301d060a2a864886f70d01090914310f0c0d437572646c652043686169727381210019bf44096984cdfe8541bac167dc3b96c85086aa30b6b6cb0c5c38ad703166e1";
/// DER contents of `id-Ed25519`, 1.3.101.112.
const ID_ED25519: &[u8] = &[0x2b, 0x65, 0x70];
/// `id-X25519`, 1.3.101.110.
const ID_X25519: &[u8] = &[0x2b, 0x65, 0x6e];

fn tlv(tag: u8, value: &[u8]) -> Vec<u8> {
  let [.., low] = value.len().to_be_bytes();
  assert!(value.len() < 0x80, "short-form test encodings");
  [&[tag, low], value].concat()
}

fn identifier(oid: &[u8], parameters: &[u8]) -> Vec<u8> {
  tlv(0x30, &[&tlv(0x06, oid), parameters].concat())
}

/// RFC 5958 OneAsymmetricKey: version, identifier, privateKey, then `rest`.
fn one_asymmetric_key(version: u8, identifier: &[u8], private_key: &[u8], rest: &[u8]) -> Vec<u8> {
  tlv(
    0x30,
    &[&tlv(0x02, &[version]), identifier, &tlv(0x04, private_key), rest].concat(),
  )
}

fn public_key_field(unused_bits: u8, key: &[u8]) -> Vec<u8> {
  tlv(0x81, &[&[unused_bits], key].concat())
}

fn spki(identifier: &[u8], unused_bits: u8, key: &[u8]) -> Vec<u8> {
  tlv(
    0x30,
    &[identifier, &tlv(0x03, &[&[unused_bits], key].concat())].concat(),
  )
}

#[test]
fn rfc8410_examples_round_trip() {
  let spki = decode_hex_vec(RFC8410_SPKI);
  let public = Ed25519PublicKey::from_spki_der(&spki).expect("RFC 8410 public key");
  assert_eq!(public.to_spki_der().as_slice(), spki.as_slice());
  assert_eq!(Ed25519PublicKey::SPKI_DER_LENGTH, spki.len());

  let pkcs8 = decode_hex_vec(RFC8410_PKCS8);
  let secret = Ed25519SecretKey::from_pkcs8_der(&pkcs8).expect("RFC 8410 private key");
  assert_eq!(secret.as_bytes().as_slice(), &pkcs8[16..]);
  // The section 10.3 version 2 example pairs this private key with the
  // section 10.1 public key.
  assert_eq!(secret.public_key().as_bytes(), public.as_bytes());
  let mut exported = [0; Ed25519SecretKey::PKCS8_DER_LENGTH];
  secret.to_pkcs8_der_into(&mut exported);
  assert_eq!(exported.as_slice(), pkcs8.as_slice());

  assert_eq!(
    Ed25519SecretKey::from_pkcs8_der(&decode_hex_vec(RFC8410_PKCS8_ATTRIBUTES)).err(),
    Some(Ed25519KeyError::UnsupportedEncoding)
  );
}

#[test]
fn keys_match_ring_in_both_directions() {
  let rng = SystemRandom::new();
  for _ in 0..8 {
    // ring writes version 2 with the public key.
    let generated = ring_signature::Ed25519KeyPair::generate_pkcs8(&rng).expect("ring key generation");
    let pair = ring_signature::Ed25519KeyPair::from_pkcs8(generated.as_ref()).expect("ring key");
    let secret = Ed25519SecretKey::from_pkcs8_der(generated.as_ref()).expect("ring PKCS #8 import");
    assert_eq!(secret.public_key().as_bytes().as_slice(), pair.public_key().as_ref());

    let mut exported = [0; Ed25519SecretKey::PKCS8_DER_LENGTH];
    secret.to_pkcs8_der_into(&mut exported);
    let reparsed =
      ring_signature::Ed25519KeyPair::from_pkcs8_maybe_unchecked(&exported).expect("ring parses the export");
    assert_eq!(reparsed.public_key().as_ref(), pair.public_key().as_ref());

    let message = b"rscrypto Ed25519 PKCS #8 exchange";
    assert_eq!(
      secret.sign(message).as_bytes().as_slice(),
      reparsed.sign(message).as_ref()
    );
  }
}

#[test]
fn pkcs8_rejects_each_violation() {
  let pkcs8 = decode_hex_vec(RFC8410_PKCS8);
  let key = &pkcs8[16..];
  let public = decode_hex_vec(RFC8410_SPKI).split_off(12);
  let foreign = Ed25519SecretKey::from_bytes([7; 32]).public_key();
  let ed25519 = identifier(ID_ED25519, &[]);
  let wrapped = tlv(0x04, key);
  let mut wrong_tag = pkcs8.clone();
  wrong_tag[0] = 0x31;
  let mut non_minimal_length = vec![0x30, 0x81];
  non_minimal_length.extend(&pkcs8[1..]);

  for (name, der, expected) in [
    ("outer tag", wrong_tag, Ed25519KeyError::MalformedDer),
    ("non-minimal length", non_minimal_length, Ed25519KeyError::MalformedDer),
    (
      "trailing data",
      [&pkcs8[..], &[0]].concat(),
      Ed25519KeyError::MalformedDer,
    ),
    (
      "unknown version",
      one_asymmetric_key(2, &ed25519, &wrapped, &[]),
      Ed25519KeyError::MalformedDer,
    ),
    (
      "version 2 without a public key",
      one_asymmetric_key(1, &ed25519, &wrapped, &[]),
      Ed25519KeyError::MalformedDer,
    ),
    (
      "version 1 with a public key",
      one_asymmetric_key(0, &ed25519, &wrapped, &public_key_field(0, &public)),
      Ed25519KeyError::MalformedDer,
    ),
    (
      "another algorithm",
      one_asymmetric_key(0, &identifier(ID_X25519, &[]), &wrapped, &[]),
      Ed25519KeyError::UnsupportedAlgorithm,
    ),
    (
      "parameters",
      one_asymmetric_key(0, &identifier(ID_ED25519, &[0x05, 0x00]), &wrapped, &[]),
      Ed25519KeyError::MalformedDer,
    ),
    (
      "attributes",
      one_asymmetric_key(0, &ed25519, &wrapped, &tlv(0xa0, &[])),
      Ed25519KeyError::UnsupportedEncoding,
    ),
    (
      "unwrapped private key",
      one_asymmetric_key(0, &ed25519, key, &[]),
      Ed25519KeyError::MalformedDer,
    ),
    (
      "short private key",
      one_asymmetric_key(0, &ed25519, &tlv(0x04, &key[1..]), &[]),
      Ed25519KeyError::InvalidSecretKey,
    ),
    (
      "long private key",
      one_asymmetric_key(0, &ed25519, &tlv(0x04, &[key, &[0]].concat()), &[]),
      Ed25519KeyError::InvalidSecretKey,
    ),
    (
      "data after the wrapped key",
      one_asymmetric_key(0, &ed25519, &[&wrapped[..], &[0x05, 0x00]].concat(), &[]),
      Ed25519KeyError::MalformedDer,
    ),
    (
      "foreign public key",
      one_asymmetric_key(1, &ed25519, &wrapped, &public_key_field(0, foreign.as_bytes())),
      Ed25519KeyError::InvalidSecretKey,
    ),
    (
      "short public key",
      one_asymmetric_key(1, &ed25519, &wrapped, &public_key_field(0, &public[1..])),
      Ed25519KeyError::InvalidPublicKey,
    ),
    (
      "public key with unused bits",
      one_asymmetric_key(1, &ed25519, &wrapped, &public_key_field(1, &public)),
      Ed25519KeyError::MalformedDer,
    ),
  ] {
    assert_eq!(Ed25519SecretKey::from_pkcs8_der(&der).err(), Some(expected), "{name}");
  }

  let version_2 = one_asymmetric_key(1, &ed25519, &wrapped, &public_key_field(0, &public));
  assert_eq!(
    Ed25519SecretKey::from_pkcs8_der(&version_2).map(|secret| *secret.as_bytes()),
    Ok(<[u8; 32]>::try_from(key).expect("key length"))
  );
}

#[test]
fn spki_rejects_each_violation() {
  let valid = decode_hex_vec(RFC8410_SPKI);
  let key = &valid[12..];
  let ed25519 = identifier(ID_ED25519, &[]);
  for (name, der, expected) in [
    (
      "trailing data",
      [&valid[..], &[0]].concat(),
      Ed25519KeyError::MalformedDer,
    ),
    (
      "truncated",
      valid[..valid.len().strict_sub(1)].to_vec(),
      Ed25519KeyError::MalformedDer,
    ),
    (
      "another algorithm",
      spki(&identifier(ID_X25519, &[]), 0, key),
      Ed25519KeyError::UnsupportedAlgorithm,
    ),
    (
      "parameters",
      spki(&identifier(ID_ED25519, &[0x05, 0x00]), 0, key),
      Ed25519KeyError::MalformedDer,
    ),
    ("unused bits", spki(&ed25519, 1, key), Ed25519KeyError::MalformedDer),
    (
      "short key",
      spki(&ed25519, 0, &key[1..]),
      Ed25519KeyError::InvalidPublicKey,
    ),
    (
      "long key",
      spki(&ed25519, 0, &[key, &[0]].concat()),
      Ed25519KeyError::InvalidPublicKey,
    ),
  ] {
    assert_eq!(Ed25519PublicKey::from_spki_der(&der).err(), Some(expected), "{name}");
  }
}

/// Exchange keys with the `openssl` CLI when it is installed; otherwise
/// report the skip.
#[test]
fn keys_match_openssl_cli_when_available() {
  use std::process::Command;

  fn openssl(args: &[&str]) -> Option<Vec<u8>> {
    let output = Command::new("openssl").args(args).output().ok()?;
    output.status.success().then_some(output.stdout)
  }

  if openssl(&["version"]).is_none() {
    eprintln!("skipping the OpenSSL Ed25519 key exchange because `openssl` is unavailable");
    return;
  }
  let directory = std::env::temp_dir().join(format!("rscrypto-ed25519-pkix-{}", std::process::id()));
  std::fs::create_dir_all(&directory).expect("temporary directory");
  let key = directory
    .join("key.der")
    .to_str()
    .expect("UTF-8 temporary path")
    .to_owned();

  openssl(&["genpkey", "-algorithm", "ED25519", "-outform", "DER", "-out", &key]).expect("OpenSSL key generation");
  let pkcs8 = std::fs::read(&key).expect("OpenSSL key file");
  let spki = openssl(&["pkey", "-in", &key, "-inform", "DER", "-pubout", "-outform", "DER"]).expect("OpenSSL SPKI");

  let secret = Ed25519SecretKey::from_pkcs8_der(&pkcs8).expect("OpenSSL PKCS #8 import");
  assert_eq!(secret.public_key().to_spki_der().as_slice(), spki.as_slice());
  assert_eq!(
    Ed25519PublicKey::from_spki_der(&spki).map(|public| *public.as_bytes()),
    Ok(*secret.public_key().as_bytes())
  );
  let mut exported = [0; Ed25519SecretKey::PKCS8_DER_LENGTH];
  secret.to_pkcs8_der_into(&mut exported);
  assert_eq!(exported.as_slice(), pkcs8.as_slice(), "OpenSSL writes the same form");

  std::fs::remove_dir_all(&directory).expect("remove temporary keys");
}
