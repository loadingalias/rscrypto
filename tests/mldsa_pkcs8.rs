//! RFC 9881 PKCS #8 import and export of ML-DSA private keys.
#![cfg(feature = "ml-dsa")]

use rscrypto::{
  MlDsa44, MlDsa44SecretKey, MlDsa44Seed, MlDsa65, MlDsa65SecretKey, MlDsa65Seed, MlDsa87, MlDsa87SecretKey,
  MlDsa87Seed, MlDsaKeyError,
};

macro_rules! fixture {
  ($name:literal) => {
    include_bytes!(concat!("../testdata/mldsa/rfc9881/", $name, ".der")).as_slice()
  };
}

const SEED_44: &[u8] = fixture!("mldsa44_pkcs8_seed");
const EXPANDED_44: &[u8] = fixture!("mldsa44_pkcs8_expanded");
const BOTH_44: &[u8] = fixture!("mldsa44_pkcs8_both");
const SEED_65: &[u8] = fixture!("mldsa65_pkcs8_seed");
const EXPANDED_65: &[u8] = fixture!("mldsa65_pkcs8_expanded");
const BOTH_65: &[u8] = fixture!("mldsa65_pkcs8_both");
const SEED_87: &[u8] = fixture!("mldsa87_pkcs8_seed");
const EXPANDED_87: &[u8] = fixture!("mldsa87_pkcs8_expanded");
const BOTH_87: &[u8] = fixture!("mldsa87_pkcs8_both");

/// DER contents of `id-ml-dsa-44`, 2.16.840.1.101.3.4.3.17 (RFC 9881 section 2).
const ID_ML_DSA_44: [u8; 9] = [0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x03, 0x11];
/// `id-hash-ml-dsa-44-with-sha512`, 2.16.840.1.101.3.4.3.32: HashML-DSA, which RFC 9881
/// section 8.3 excludes.
const ID_HASH_ML_DSA_44: [u8; 9] = [0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x03, 0x20];
/// `rsaEncryption`, 1.2.840.113549.1.1.1.
const RSA_ENCRYPTION: [u8; 9] = [0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01];
/// RFC 8410 section 10.3 Ed25519 private key.
const ED25519_PKCS8: [u8; 48] = [
  0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20, 0xd4, 0xee, 0x72,
  0xdb, 0xf9, 0x13, 0x58, 0x4a, 0xd5, 0xb6, 0xd8, 0xf1, 0xf7, 0x69, 0xf8, 0xad, 0x3a, 0xfe, 0x7c, 0x28, 0xcb, 0xf1,
  0xd4, 0xfb, 0xe0, 0x97, 0xa8, 0x8f, 0x44, 0x75, 0x58, 0x42,
];

/// RFC 9881 Appendix C derives every consistent example key from this seed.
fn rfc9881_seed() -> [u8; 32] {
  core::array::from_fn(|index| u8::try_from(index).expect("seed index fits a byte"))
}

macro_rules! rfc9881_examples {
  ($name:ident, $profile:ident, $secret:ident, $seed:ident, $seed_der:ident, $expanded_der:ident, $both_der:ident) => {
    #[test]
    fn $name() {
      let (public, secret) = $profile::keypair_from_seed(&rfc9881_seed()).expect("key generation");
      let expected = secret.expose_secret();

      for der in [$seed_der, $expanded_der, $both_der] {
        let imported = $secret::from_pkcs8_der(der).expect("RFC 9881 example import");
        assert_eq!(imported.expose_secret().as_bytes(), expected.as_bytes());
        assert_eq!(imported.public_key(), &public);
      }
      assert_eq!($seed::PKCS8_DER_LENGTH, $seed_der.len());
      assert_eq!($secret::PKCS8_DER_LENGTH, $expanded_der.len());

      let mut exported = [0; $secret::PKCS8_DER_LENGTH];
      secret.to_pkcs8_der_into(&mut exported);
      assert_eq!(exported.as_slice(), $expanded_der);

      for der in [$seed_der, $both_der] {
        let seed = $seed::from_pkcs8_der(der).expect("RFC 9881 seed import");
        assert_eq!(seed.expose_secret().as_bytes(), &rfc9881_seed());
        let mut exported = [0; $seed::PKCS8_DER_LENGTH];
        seed.to_pkcs8_der_into(&mut exported);
        assert_eq!(exported.as_slice(), $seed_der);
        let (seed_public, seed_secret) = seed.keypair().expect("seed expansion");
        assert_eq!(seed_public, public);
        assert_eq!(seed_secret.expose_secret().as_bytes(), expected.as_bytes());
      }
      // Expansion cannot recover the seed that the expanded form discarded.
      assert_eq!(
        $seed::from_pkcs8_der($expanded_der).err(),
        Some(MlDsaKeyError::UnsupportedEncoding)
      );
    }
  };
}

rfc9881_examples!(
  rfc9881_examples_44,
  MlDsa44,
  MlDsa44SecretKey,
  MlDsa44Seed,
  SEED_44,
  EXPANDED_44,
  BOTH_44
);
rfc9881_examples!(
  rfc9881_examples_65,
  MlDsa65,
  MlDsa65SecretKey,
  MlDsa65Seed,
  SEED_65,
  EXPANDED_65,
  BOTH_65
);
rfc9881_examples!(
  rfc9881_examples_87,
  MlDsa87,
  MlDsa87SecretKey,
  MlDsa87Seed,
  SEED_87,
  EXPANDED_87,
  BOTH_87
);

#[test]
fn rfc9881_inconsistent_examples_are_rejected() {
  for der in [
    fixture!("mldsa44_pkcs8_inconsistent_seed"),
    fixture!("mldsa44_pkcs8_inconsistent_tr"),
    fixture!("mldsa44_pkcs8_inconsistent_t0"),
  ] {
    assert_eq!(
      MlDsa44SecretKey::from_pkcs8_der(der).err(),
      Some(MlDsaKeyError::InvalidSecretKey)
    );
  }
  assert_eq!(
    MlDsa44Seed::from_pkcs8_der(fixture!("mldsa44_pkcs8_inconsistent_seed")).err(),
    Some(MlDsaKeyError::InvalidSecretKey)
  );
}

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

fn concat(parts: &[&[u8]]) -> Vec<u8> {
  parts.concat()
}

/// OneAsymmetricKey with the given version contents, AlgorithmIdentifier contents,
/// private-key CHOICE encoding, and trailing fields.
fn one_asymmetric_key(version: &[u8], algorithm: &[u8], private_key: &[u8], trailing: &[&[u8]]) -> Vec<u8> {
  let mut contents = concat(&[&tlv(0x02, version), &tlv(0x30, algorithm), &tlv(0x04, private_key)]);
  for field in trailing {
    contents.extend_from_slice(field);
  }
  tlv(0x30, &contents)
}

fn seed_choice(seed: &[u8]) -> Vec<u8> {
  tlv(0x80, seed)
}

fn expanded_choice(key: &[u8]) -> Vec<u8> {
  tlv(0x04, key)
}

fn both_choice(seed: &[u8], key: &[u8]) -> Vec<u8> {
  tlv(0x30, &concat(&[&tlv(0x04, seed), &tlv(0x04, key)]))
}

fn public_key_field(unused_bits: u8, key: &[u8]) -> Vec<u8> {
  tlv(0x81, &concat(&[&[unused_bits], key]))
}

#[test]
fn version_2_public_keys_must_belong_to_the_key() {
  let seed = rfc9881_seed();
  let (public, secret) = MlDsa44::keypair_from_seed(&seed).expect("key generation");
  let expanded = secret.expose_secret();
  let oid = tlv(0x06, &ID_ML_DSA_44);
  let (other_public, _) = MlDsa44::keypair_from_seed(&[0x5a; 32]).expect("key generation");

  for choice in [
    seed_choice(&seed),
    expanded_choice(expanded.as_bytes()),
    both_choice(&seed, expanded.as_bytes()),
  ] {
    let matching = one_asymmetric_key(&[1], &oid, &choice, &[&public_key_field(0, public.as_bytes())]);
    let imported = MlDsa44SecretKey::from_pkcs8_der(&matching).expect("version 2 import");
    assert_eq!(imported.public_key(), &public);

    let mismatched = one_asymmetric_key(&[1], &oid, &choice, &[&public_key_field(0, other_public.as_bytes())]);
    assert_eq!(
      MlDsa44SecretKey::from_pkcs8_der(&mismatched).err(),
      Some(MlDsaKeyError::InvalidSecretKey)
    );
    let short = one_asymmetric_key(&[1], &oid, &choice, &[&public_key_field(0, &public.as_bytes()[1..])]);
    assert_eq!(
      MlDsa44SecretKey::from_pkcs8_der(&short).err(),
      Some(MlDsaKeyError::InvalidPublicKey)
    );
    let unused_bits = one_asymmetric_key(&[1], &oid, &choice, &[&public_key_field(1, public.as_bytes())]);
    assert_eq!(
      MlDsa44SecretKey::from_pkcs8_der(&unused_bits).err(),
      Some(MlDsaKeyError::MalformedDer)
    );

    if choice[0] != 0x04 {
      assert_eq!(
        MlDsa44Seed::from_pkcs8_der(&matching)
          .expect("version 2 seed import")
          .expose_secret()
          .as_bytes(),
        &seed
      );
      assert_eq!(
        MlDsa44Seed::from_pkcs8_der(&mismatched).err(),
        Some(MlDsaKeyError::InvalidSecretKey)
      );
    }
  }
}

#[test]
fn rejections_name_the_failing_component() {
  let seed = rfc9881_seed();
  let (public, secret) = MlDsa44::keypair_from_seed(&seed).expect("key generation");
  let expanded = secret.expose_secret();
  let key = expanded.as_bytes().as_slice();
  let oid = tlv(0x06, &ID_ML_DSA_44);
  // The builders reproduce the RFC examples, anchoring the variants below.
  assert_eq!(one_asymmetric_key(&[0], &oid, &seed_choice(&seed), &[]), SEED_44);
  assert_eq!(one_asymmetric_key(&[0], &oid, &expanded_choice(key), &[]), EXPANDED_44);
  assert_eq!(one_asymmetric_key(&[0], &oid, &both_choice(&seed, key), &[]), BOTH_44);

  let mut rsa_algorithm = tlv(0x06, &RSA_ENCRYPTION);
  rsa_algorithm.extend([0x05, 0x00]);
  for other in [
    SEED_65.to_vec(),
    EXPANDED_87.to_vec(),
    ED25519_PKCS8.to_vec(),
    one_asymmetric_key(&[0], &tlv(0x06, &ID_HASH_ML_DSA_44), &seed_choice(&seed), &[]),
    one_asymmetric_key(&[0], &rsa_algorithm, &seed_choice(&seed), &[]),
  ] {
    assert_eq!(
      MlDsa44SecretKey::from_pkcs8_der(&other).err(),
      Some(MlDsaKeyError::UnsupportedAlgorithm)
    );
    assert_eq!(
      MlDsa44Seed::from_pkcs8_der(&other).err(),
      Some(MlDsaKeyError::UnsupportedAlgorithm)
    );
  }

  let attributes = tlv(0xa0, &tlv(0x30, &[]));
  for unsupported in [
    one_asymmetric_key(&[0], &oid, &seed_choice(&seed), &[&attributes]),
    one_asymmetric_key(
      &[1],
      &oid,
      &seed_choice(&seed),
      &[&attributes, &public_key_field(0, public.as_bytes())],
    ),
  ] {
    assert_eq!(
      MlDsa44SecretKey::from_pkcs8_der(&unsupported).err(),
      Some(MlDsaKeyError::UnsupportedEncoding)
    );
  }

  let mut key_too_long = key.to_vec();
  key_too_long.push(0);
  let mut invalid_noise = key.to_vec();
  // The first s1 coefficient's encoding moves outside the eta range.
  invalid_noise[128] = 0xff;
  for invalid in [
    seed_choice(&seed[1..]),
    seed_choice(&[seed.as_slice(), &[0]].concat()),
    expanded_choice(&key[1..]),
    expanded_choice(&key_too_long),
    expanded_choice(&invalid_noise),
    both_choice(&seed[1..], key),
    both_choice(&seed, &key[1..]),
    both_choice(&[0x5a; 32], key),
  ] {
    assert_eq!(
      MlDsa44SecretKey::from_pkcs8_der(&one_asymmetric_key(&[0], &oid, &invalid, &[])).err(),
      Some(MlDsaKeyError::InvalidSecretKey)
    );
  }

  let mut null_parameters = oid.clone();
  null_parameters.extend([0x05, 0x00]);
  let mut trailing = SEED_44.to_vec();
  trailing.push(0);
  let mut long_form = vec![0x30, 0x81, 0x34];
  long_form.extend_from_slice(&SEED_44[2..]);
  let mut indefinite = vec![0x30, 0x80];
  indefinite.extend_from_slice(&SEED_44[2..]);
  indefinite.extend([0, 0]);
  let mut set_tag = SEED_44.to_vec();
  set_tag[0] = 0x31;
  let seed_der = seed_choice(&seed);
  for malformed in [
    one_asymmetric_key(&[0], &null_parameters, &seed_der, &[]),
    one_asymmetric_key(&[2], &oid, &seed_der, &[]),
    one_asymmetric_key(&[0, 0], &oid, &seed_der, &[]),
    one_asymmetric_key(&[0xff], &oid, &seed_der, &[]),
    one_asymmetric_key(&[], &oid, &seed_der, &[]),
    // RFC 5958: version 1 has no public key, and version 2 requires one.
    one_asymmetric_key(&[0], &oid, &seed_der, &[&public_key_field(0, public.as_bytes())]),
    one_asymmetric_key(&[1], &oid, &seed_der, &[]),
    one_asymmetric_key(&[0], &oid, &seed_der, &[&tlv(0x82, &[])]),
    one_asymmetric_key(&[0], &oid, &[seed_der.as_slice(), &[0]].concat(), &[]),
    one_asymmetric_key(&[0], &oid, &tlv(0xa0, &tlv(0x04, &seed)), &[]),
    one_asymmetric_key(&[0], &oid, &tlv(0x02, &seed), &[]),
    one_asymmetric_key(&[0], &oid, &[], &[]),
    one_asymmetric_key(
      &[0],
      &oid,
      &tlv(0x30, &concat(&[&seed_der, &expanded_choice(key)])),
      &[],
    ),
    one_asymmetric_key(
      &[0],
      &oid,
      &tlv(
        0x30,
        &concat(&[&tlv(0x04, &seed), &expanded_choice(key), &[0x05, 0x00]]),
      ),
      &[],
    ),
    trailing,
    long_form,
    indefinite,
    set_tag,
  ] {
    assert_eq!(
      MlDsa44SecretKey::from_pkcs8_der(&malformed).err(),
      Some(MlDsaKeyError::MalformedDer)
    );
  }
  for der in [SEED_44, EXPANDED_44, BOTH_44] {
    for len in 0..der.len() {
      assert_eq!(
        MlDsa44SecretKey::from_pkcs8_der(&der[..len]).err(),
        Some(MlDsaKeyError::MalformedDer),
        "truncated to {len} bytes"
      );
    }
  }
}

/// Change every byte outside the `[start, end)` payloads to every other value; each change
/// must be rejected.
fn every_framing_byte_is_required(der: &[u8], payloads: &[(usize, usize)]) {
  let mut input = der.to_vec();
  for position in
    (0..der.len()).filter(|&position| !payloads.iter().any(|&(start, end)| (start..end).contains(&position)))
  {
    let original = input[position];
    for value in (0..=u8::MAX).filter(|&value| value != original) {
      input[position] = value;
      assert!(
        MlDsa44SecretKey::from_pkcs8_der(&input).is_err(),
        "byte {position} = {value:#04x} was accepted"
      );
    }
    input[position] = original;
  }
}

#[test]
fn every_seed_form_header_byte_is_required() {
  every_framing_byte_is_required(SEED_44, &[(22, 54)]);
}

#[test]
fn every_expanded_form_header_byte_is_required() {
  every_framing_byte_is_required(EXPANDED_44, &[(28, EXPANDED_44.len())]);
}

#[test]
fn every_both_form_framing_byte_is_required() {
  // Header and seed OCTET STRING header, the seed, the key's OCTET STRING header, then the key.
  every_framing_byte_is_required(BOTH_44, &[(30, 62), (66, BOTH_44.len())]);
}

#[test]
fn pkcs8_matches_rustcrypto_ml_dsa() {
  use rustcrypto_ml_dsa::{self as oracle, pkcs8::DecodePrivateKey, pkcs8::EncodePrivateKey};

  macro_rules! exchange {
    ($secret:ident, $seed:ident, $oracle:ident) => {
      for seed_bytes in [[0; 32], [0x5a; 32], [0xff; 32]] {
        let theirs = oracle::SigningKey::<oracle::$oracle>::from_seed(&seed_bytes.into());
        let encoded = theirs.to_pkcs8_der().expect("oracle PKCS #8 export");
        let seed = $seed::from_pkcs8_der(encoded.as_bytes()).expect("oracle seed import");
        assert_eq!(seed.expose_secret().as_bytes(), &seed_bytes);
        let mut ours = [0; $seed::PKCS8_DER_LENGTH];
        seed.to_pkcs8_der_into(&mut ours);
        assert_eq!(ours.as_slice(), encoded.as_bytes());

        let imported = oracle::SigningKey::<oracle::$oracle>::from_pkcs8_der(&ours).expect("oracle import");
        assert_eq!(imported.as_seed().as_slice(), &seed_bytes);
        let secret = $secret::from_pkcs8_der(encoded.as_bytes()).expect("oracle secret import");
        assert_eq!(
          secret.public_key().as_bytes().as_slice(),
          theirs.expanded_key().verifying_key().encode().as_slice()
        );
      }
    };
  }

  exchange!(MlDsa44SecretKey, MlDsa44Seed, MlDsa44);
  exchange!(MlDsa65SecretKey, MlDsa65Seed, MlDsa65);
  exchange!(MlDsa87SecretKey, MlDsa87Seed, MlDsa87);
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
fn pkcs8_matches_aws_lc_rs() {
  use aws_lc_rs::{
    encoding::AsDer,
    signature::{self as aws, KeyPair, PqdsaKeyPair},
  };

  macro_rules! exchange {
    ($secret:ident, $seed:ident, $signing:ident) => {
      for seed_bytes in [[0; 32], [0x5a; 32], [0xff; 32]] {
        let theirs = PqdsaKeyPair::from_seed(&aws::$signing, &seed_bytes).expect("AWS-LC key generation");
        let theirs_spki = theirs.public_key().as_der().expect("AWS-LC SPKI export");
        let encoded = theirs.to_pkcs8v1().expect("AWS-LC PKCS #8 export");
        let seed = $seed::from_pkcs8_der(encoded.as_ref()).expect("AWS-LC seed import");
        let mut ours_seed = [0; $seed::PKCS8_DER_LENGTH];
        seed.to_pkcs8_der_into(&mut ours_seed);
        assert_eq!(ours_seed.as_slice(), encoded.as_ref());

        let secret = $secret::from_pkcs8_der(encoded.as_ref()).expect("AWS-LC secret import");
        assert_eq!(secret.public_key().to_spki_der().as_slice(), theirs_spki.as_ref());
        let mut ours_expanded = [0; $secret::PKCS8_DER_LENGTH];
        secret.to_pkcs8_der_into(&mut ours_expanded);
        // AWS-LC accepts both forms and derives the same key pair from each.
        for ours in [ours_seed.as_slice(), ours_expanded.as_slice()] {
          let imported = PqdsaKeyPair::from_pkcs8(&aws::$signing, ours).expect("AWS-LC import");
          assert_eq!(
            imported.public_key().as_der().expect("AWS-LC SPKI export").as_ref(),
            theirs_spki.as_ref()
          );
        }
      }
    };
  }

  exchange!(MlDsa44SecretKey, MlDsa44Seed, ML_DSA_44_SIGNING);
  exchange!(MlDsa65SecretKey, MlDsa65Seed, ML_DSA_65_SIGNING);
  exchange!(MlDsa87SecretKey, MlDsa87Seed, ML_DSA_87_SIGNING);
}

#[test]
fn seed_owner_generation_matches_key_generation_and_redacts_debug() {
  let generated = MlDsa44Seed::generate(|out| {
    out.fill(0x5a);
    Ok(())
  })
  .expect("seed generation");
  assert_eq!(generated.expose_secret().as_bytes(), &[0x5a; 32]);
  let (public, secret) = generated.keypair().expect("seed expansion");
  let (expected_public, expected_secret) = MlDsa44::keypair_from_seed(&[0x5a; 32]).expect("key generation");
  assert_eq!(public, expected_public);
  assert_eq!(
    secret.expose_secret().as_bytes(),
    expected_secret.expose_secret().as_bytes()
  );

  let failed = MlDsa44Seed::generate(|out| {
    out[..13].fill(0xff);
    Err(rscrypto::MlDsaError::RandomGenerationFailed)
  });
  assert_eq!(failed.err(), Some(rscrypto::MlDsaError::RandomGenerationFailed));
  assert_eq!(format!("{generated:?}"), "MlDsa44Seed(****)");
}

/// OpenSSL 3.5 and later implement RFC 9881. Exchange every form with the
/// `openssl` CLI when it supports ML-DSA; otherwise report the skip.
#[test]
fn pkcs8_matches_openssl_cli_when_available() {
  use std::process::Command;

  fn openssl(args: &[&str]) -> Option<Vec<u8>> {
    let output = Command::new("openssl").args(args).output().ok()?;
    output.status.success().then_some(output.stdout)
  }

  let supported = openssl(&["list", "-signature-algorithms"])
    .is_some_and(|list| String::from_utf8_lossy(&list).contains("id-ml-dsa-87"));
  if !supported {
    eprintln!("skipping the OpenSSL ML-DSA PKCS #8 exchange because `openssl` lacks ML-DSA");
    return;
  }
  let directory = std::env::temp_dir().join(format!("rscrypto-mldsa-pkcs8-{}", std::process::id()));
  std::fs::create_dir_all(&directory).expect("temporary directory");
  let path = |name: &str| directory.join(name).to_str().expect("UTF-8 temporary path").to_owned();

  macro_rules! exchange {
    ($algorithm:literal, $secret:ident, $seed:ident) => {
      for format in ["seed-only", "priv-only", "seed-priv"] {
        let key = path(&format!("{}-{format}.der", $algorithm));
        let format_param = format!("ml-dsa.output_formats={format}");
        openssl(&[
          "genpkey",
          "-algorithm",
          $algorithm,
          "-provparam",
          &format_param,
          "-outform",
          "DER",
          "-out",
          &key,
        ])
        .expect("OpenSSL key generation");
        let spki =
          openssl(&["pkey", "-in", &key, "-inform", "DER", "-pubout", "-outform", "DER"]).expect("OpenSSL SPKI");
        let der = std::fs::read(&key).expect("OpenSSL key file");
        let secret = $secret::from_pkcs8_der(&der).expect("OpenSSL key import");
        assert_eq!(
          secret.public_key().to_spki_der().as_slice(),
          spki.as_slice(),
          "{format}"
        );
        assert_eq!(
          $seed::from_pkcs8_der(&der).err(),
          if format == "priv-only" {
            Some(MlDsaKeyError::UnsupportedEncoding)
          } else {
            None
          },
          "{format}"
        );

        let mut expanded = [0; $secret::PKCS8_DER_LENGTH];
        secret.to_pkcs8_der_into(&mut expanded);
        let exported = path(&format!("{}-{format}-rscrypto.der", $algorithm));
        std::fs::write(&exported, expanded).expect("export file");
        let parsed = openssl(&["pkey", "-in", &exported, "-inform", "DER", "-pubout", "-outform", "DER"])
          .expect("OpenSSL parses the expanded export");
        assert_eq!(parsed, spki, "{format}");
        if let Ok(seed) = $seed::from_pkcs8_der(&der) {
          let mut encoded = [0; $seed::PKCS8_DER_LENGTH];
          seed.to_pkcs8_der_into(&mut encoded);
          std::fs::write(&exported, encoded).expect("export file");
          let parsed = openssl(&["pkey", "-in", &exported, "-inform", "DER", "-pubout", "-outform", "DER"])
            .expect("OpenSSL parses the seed export");
          assert_eq!(parsed, spki, "{format}");
        }
      }
    };
  }

  exchange!("ML-DSA-44", MlDsa44SecretKey, MlDsa44Seed);
  exchange!("ML-DSA-65", MlDsa65SecretKey, MlDsa65Seed);
  exchange!("ML-DSA-87", MlDsa87SecretKey, MlDsa87Seed);
  std::fs::remove_dir_all(&directory).expect("remove temporary keys");
}
