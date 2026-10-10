//! RFC 9935 SubjectPublicKeyInfo and PKCS #8 import and export of ML-KEM keys.
#![cfg(feature = "ml-kem")]

use rscrypto::{
  Kem as _, MlKem512, MlKem512DecapsulationKey, MlKem512EncapsulationKey, MlKem512Seed, MlKem768,
  MlKem768DecapsulationKey, MlKem768EncapsulationKey, MlKem768Seed, MlKem1024, MlKem1024DecapsulationKey,
  MlKem1024EncapsulationKey, MlKem1024Seed, MlKemError, MlKemKeyError,
};

macro_rules! fixture {
  ($name:literal) => {
    include_bytes!(concat!("../testdata/mlkem/rfc9935/", $name, ".der")).as_slice()
  };
}

const SPKI_512: &[u8] = fixture!("mlkem512_spki");
const SEED_512: &[u8] = fixture!("mlkem512_pkcs8_seed");
const EXPANDED_512: &[u8] = fixture!("mlkem512_pkcs8_expanded");
const BOTH_512: &[u8] = fixture!("mlkem512_pkcs8_both");
const SPKI_768: &[u8] = fixture!("mlkem768_spki");
const SEED_768: &[u8] = fixture!("mlkem768_pkcs8_seed");
const EXPANDED_768: &[u8] = fixture!("mlkem768_pkcs8_expanded");
const BOTH_768: &[u8] = fixture!("mlkem768_pkcs8_both");
const SPKI_1024: &[u8] = fixture!("mlkem1024_spki");
const SEED_1024: &[u8] = fixture!("mlkem1024_pkcs8_seed");
const EXPANDED_1024: &[u8] = fixture!("mlkem1024_pkcs8_expanded");
const BOTH_1024: &[u8] = fixture!("mlkem1024_pkcs8_both");

/// DER contents of `id-alg-ml-kem-512`, 2.16.840.1.101.3.4.4.1 (RFC 9935 section 3).
const ID_ML_KEM_512: [u8; 9] = [0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x04, 0x01];
/// `rsaEncryption`, 1.2.840.113549.1.1.1.
const RSA_ENCRYPTION: [u8; 9] = [0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01];
/// RFC 8410 section 10.3 Ed25519 private key.
const ED25519_PKCS8: [u8; 48] = [
  0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20, 0xd4, 0xee, 0x72,
  0xdb, 0xf9, 0x13, 0x58, 0x4a, 0xd5, 0xb6, 0xd8, 0xf1, 0xf7, 0x69, 0xf8, 0xad, 0x3a, 0xfe, 0x7c, 0x28, 0xcb, 0xf1,
  0xd4, 0xfb, 0xe0, 0x97, 0xa8, 0x8f, 0x44, 0x75, 0x58, 0x42,
];
/// RFC 8410 section 10.1 Ed25519 public key.
const ED25519_SPKI: [u8; 44] = [
  0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00, 0x19, 0xbf, 0x44, 0x09, 0x69, 0x84, 0xcd,
  0xfe, 0x85, 0x41, 0xba, 0xc1, 0x67, 0xdc, 0x3b, 0x96, 0xc8, 0x50, 0x86, 0xaa, 0x30, 0xb6, 0xb6, 0xcb, 0x0c, 0x5c,
  0x38, 0xad, 0x70, 0x31, 0x66, 0xe1,
];

/// RFC 9935 Appendix C derives every consistent example key from this `d || z` seed.
fn rfc9935_seed() -> [u8; 64] {
  core::array::from_fn(|index| u8::try_from(index).expect("seed index fits a byte"))
}

macro_rules! rfc9935_examples {
  ($name:ident, $profile:ident, $ek:ident, $dk:ident, $seed:ident, $spki:ident, $seed_der:ident, $expanded_der:ident, $both_der:ident) => {
    #[test]
    fn $name() {
      let seed = $seed::from_bytes(rfc9935_seed());
      let (ek, dk) = seed.keypair();
      let (generated_ek, generated_dk) = $profile::generate_keypair(|out| {
        out.copy_from_slice(&rfc9935_seed());
        Ok::<(), MlKemError>(())
      })
      .expect("key generation");
      assert_eq!(ek.as_bytes(), generated_ek.as_bytes());
      assert_eq!(dk.as_bytes(), generated_dk.as_bytes());

      assert_eq!($ek::SPKI_DER_LENGTH, $spki.len());
      assert_eq!(ek.to_spki_der().as_slice(), $spki);
      assert_eq!(
        $ek::from_spki_der($spki).expect("RFC 9935 SPKI").as_bytes(),
        ek.as_bytes()
      );

      for der in [$seed_der, $expanded_der, $both_der] {
        let imported = $dk::from_pkcs8_der(der).expect("RFC 9935 private key");
        assert_eq!(imported.as_bytes(), dk.as_bytes());
      }
      assert_eq!($seed::PKCS8_DER_LENGTH, $seed_der.len());
      assert_eq!($dk::PKCS8_DER_LENGTH, $expanded_der.len());
      let mut exported = [0; $dk::PKCS8_DER_LENGTH];
      dk.to_pkcs8_der_into(&mut exported);
      assert_eq!(exported.as_slice(), $expanded_der);

      for der in [$seed_der, $both_der] {
        let imported = $seed::from_pkcs8_der(der).expect("RFC 9935 seed");
        assert_eq!(imported.expose_secret().as_bytes(), &rfc9935_seed());
        let mut exported = [0; $seed::PKCS8_DER_LENGTH];
        imported.to_pkcs8_der_into(&mut exported);
        assert_eq!(exported.as_slice(), $seed_der);
      }
      // Expansion cannot recover the seed that the expanded form discarded.
      assert_eq!(
        $seed::from_pkcs8_der($expanded_der).err(),
        Some(MlKemKeyError::UnsupportedEncoding)
      );
    }
  };
}

rfc9935_examples!(
  rfc9935_examples_512,
  MlKem512,
  MlKem512EncapsulationKey,
  MlKem512DecapsulationKey,
  MlKem512Seed,
  SPKI_512,
  SEED_512,
  EXPANDED_512,
  BOTH_512
);
rfc9935_examples!(
  rfc9935_examples_768,
  MlKem768,
  MlKem768EncapsulationKey,
  MlKem768DecapsulationKey,
  MlKem768Seed,
  SPKI_768,
  SEED_768,
  EXPANDED_768,
  BOTH_768
);
rfc9935_examples!(
  rfc9935_examples_1024,
  MlKem1024,
  MlKem1024EncapsulationKey,
  MlKem1024DecapsulationKey,
  MlKem1024Seed,
  SPKI_1024,
  SEED_1024,
  EXPANDED_1024,
  BOTH_1024
);

#[test]
fn rfc9935_inconsistent_examples_are_rejected() {
  for der in [
    fixture!("mlkem512_pkcs8_inconsistent_seed"),
    fixture!("mlkem512_pkcs8_inconsistent_s"),
    fixture!("mlkem512_pkcs8_inconsistent_hash"),
    fixture!("mlkem512_pkcs8_inconsistent_z"),
  ] {
    assert_eq!(
      MlKem512DecapsulationKey::from_pkcs8_der(der).err(),
      Some(MlKemKeyError::InvalidDecapsulationKey)
    );
  }
  for der in [
    fixture!("mlkem512_pkcs8_inconsistent_seed"),
    fixture!("mlkem512_pkcs8_inconsistent_z"),
  ] {
    assert_eq!(
      MlKem512Seed::from_pkcs8_der(der).err(),
      Some(MlKemKeyError::InvalidDecapsulationKey)
    );
  }
}

/// The FIPS 203 checks of raw import accept a mutated secret vector with a valid
/// `H(ek)`; PKCS #8 import adds the pairwise consistency check that rejects it.
#[test]
fn pkcs8_import_rejects_a_secret_vector_that_the_hash_check_accepts() {
  let der = fixture!("mlkem512_pkcs8_inconsistent_s");
  let expanded = &der[der.len() - MlKem512DecapsulationKey::LENGTH..];
  let raw = MlKem512DecapsulationKey::try_from_slice(expanded).expect("raw import runs only the FIPS 203 checks");
  assert_eq!(raw.validate(), Ok(()));
  assert_eq!(
    MlKem512DecapsulationKey::from_pkcs8_der(der).err(),
    Some(MlKemKeyError::InvalidDecapsulationKey)
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

fn spki(algorithm: &[u8], unused_bits: u8, key: &[u8]) -> Vec<u8> {
  let mut bits = vec![unused_bits];
  bits.extend_from_slice(key);
  let mut contents = tlv(0x30, algorithm);
  contents.extend(tlv(0x03, &bits));
  tlv(0x30, &contents)
}

#[test]
fn spki_rejections_name_the_failing_component() {
  let key = &SPKI_512[22..];
  let oid = tlv(0x06, &ID_ML_KEM_512);
  assert_eq!(spki(&oid, 0, key), SPKI_512);

  let mut rsa_algorithm = tlv(0x06, &RSA_ENCRYPTION);
  rsa_algorithm.extend([0x05, 0x00]);
  let mldsa_spki = include_bytes!("../testdata/mldsa/rfc9881/mldsa44_spki.der");
  for other in [
    SPKI_768.to_vec(),
    SPKI_1024.to_vec(),
    ED25519_SPKI.to_vec(),
    mldsa_spki.to_vec(),
    spki(&rsa_algorithm, 0, key),
  ] {
    assert_eq!(
      MlKem512EncapsulationKey::from_spki_der(&other).err(),
      Some(MlKemKeyError::UnsupportedAlgorithm)
    );
  }

  let mut key_too_long = key.to_vec();
  key_too_long.push(0);
  // The first 12-bit coefficient becomes 4095, at least the modulus 3329.
  let mut not_reduced = key.to_vec();
  not_reduced[0] = 0xff;
  not_reduced[1] |= 0x0f;
  for invalid in [&key[1..], key_too_long.as_slice(), not_reduced.as_slice()] {
    assert_eq!(
      MlKem512EncapsulationKey::from_spki_der(&spki(&oid, 0, invalid)).err(),
      Some(MlKemKeyError::InvalidEncapsulationKey)
    );
  }

  let mut null_parameters = oid.clone();
  null_parameters.extend([0x05, 0x00]);
  let mut trailing = SPKI_512.to_vec();
  trailing.push(0);
  let mut long_form = vec![0x30, 0x83, 0x00];
  long_form.extend_from_slice(&SPKI_512[2..]);
  let mut set_tag = SPKI_512.to_vec();
  set_tag[0] = 0x31;
  for malformed in [
    spki(&null_parameters, 0, key),
    spki(&oid, 1, key),
    trailing,
    long_form,
    set_tag,
  ] {
    assert_eq!(
      MlKem512EncapsulationKey::from_spki_der(&malformed).err(),
      Some(MlKemKeyError::MalformedDer)
    );
  }
  for len in 0..SPKI_512.len() {
    assert_eq!(
      MlKem512EncapsulationKey::from_spki_der(&SPKI_512[..len]).err(),
      Some(MlKemKeyError::MalformedDer),
      "truncated to {len} bytes"
    );
  }
}

/// OneAsymmetricKey with the given version contents, AlgorithmIdentifier contents,
/// private-key CHOICE encoding, and trailing fields.
fn one_asymmetric_key(version: &[u8], algorithm: &[u8], private_key: &[u8], trailing: &[&[u8]]) -> Vec<u8> {
  let mut contents = [tlv(0x02, version), tlv(0x30, algorithm), tlv(0x04, private_key)].concat();
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
  tlv(0x30, &[tlv(0x04, seed), tlv(0x04, key)].concat())
}

fn public_key_field(unused_bits: u8, key: &[u8]) -> Vec<u8> {
  tlv(0x81, &[&[unused_bits], key].concat())
}

#[test]
fn pkcs8_rejections_name_the_failing_component() {
  let seed = rfc9935_seed();
  let (ek, dk) = MlKem512Seed::from_bytes(seed).keypair();
  let key = dk.as_bytes().as_slice();
  let oid = tlv(0x06, &ID_ML_KEM_512);
  assert_eq!(one_asymmetric_key(&[0], &oid, &seed_choice(&seed), &[]), SEED_512);
  assert_eq!(one_asymmetric_key(&[0], &oid, &expanded_choice(key), &[]), EXPANDED_512);
  assert_eq!(one_asymmetric_key(&[0], &oid, &both_choice(&seed, key), &[]), BOTH_512);

  let (other_ek, _) = MlKem512Seed::from_bytes([0x5a; 64]).keypair();
  for choice in [seed_choice(&seed), expanded_choice(key), both_choice(&seed, key)] {
    let matching = one_asymmetric_key(&[1], &oid, &choice, &[&public_key_field(0, ek.as_bytes())]);
    assert_eq!(
      MlKem512DecapsulationKey::from_pkcs8_der(&matching)
        .expect("version 2 import")
        .as_bytes(),
      dk.as_bytes()
    );
    let mismatched = one_asymmetric_key(&[1], &oid, &choice, &[&public_key_field(0, other_ek.as_bytes())]);
    assert_eq!(
      MlKem512DecapsulationKey::from_pkcs8_der(&mismatched).err(),
      Some(MlKemKeyError::InvalidDecapsulationKey)
    );
    let short = one_asymmetric_key(&[1], &oid, &choice, &[&public_key_field(0, &ek.as_bytes()[1..])]);
    assert_eq!(
      MlKem512DecapsulationKey::from_pkcs8_der(&short).err(),
      Some(MlKemKeyError::InvalidEncapsulationKey)
    );
    if choice[0] != 0x04 {
      assert_eq!(
        MlKem512Seed::from_pkcs8_der(&matching)
          .expect("version 2 seed import")
          .expose_secret()
          .as_bytes(),
        &seed
      );
      assert_eq!(
        MlKem512Seed::from_pkcs8_der(&mismatched).err(),
        Some(MlKemKeyError::InvalidDecapsulationKey)
      );
    }
  }

  let mut rsa_algorithm = tlv(0x06, &RSA_ENCRYPTION);
  rsa_algorithm.extend([0x05, 0x00]);
  for other in [
    SEED_768.to_vec(),
    EXPANDED_1024.to_vec(),
    ED25519_PKCS8.to_vec(),
    include_bytes!("../testdata/mldsa/rfc9881/mldsa44_pkcs8_seed.der").to_vec(),
    one_asymmetric_key(&[0], &rsa_algorithm, &seed_choice(&seed), &[]),
  ] {
    assert_eq!(
      MlKem512DecapsulationKey::from_pkcs8_der(&other).err(),
      Some(MlKemKeyError::UnsupportedAlgorithm)
    );
  }

  let attributes = tlv(0xa0, &tlv(0x30, &[]));
  assert_eq!(
    MlKem512DecapsulationKey::from_pkcs8_der(&one_asymmetric_key(&[0], &oid, &seed_choice(&seed), &[&attributes]))
      .err(),
    Some(MlKemKeyError::UnsupportedEncoding)
  );

  let mut key_too_long = key.to_vec();
  key_too_long.push(0);
  let mut wrong_hash = key.to_vec();
  // H(ek) follows dk_pke (768 bytes) and ek (800 bytes).
  wrong_hash[768 + 800] ^= 1;
  for invalid in [
    seed_choice(&seed[1..]),
    seed_choice(&[seed.as_slice(), &[0]].concat()),
    expanded_choice(&key[1..]),
    expanded_choice(&key_too_long),
    expanded_choice(&wrong_hash),
    both_choice(&seed[1..], key),
    both_choice(&seed, &key[1..]),
    both_choice(&[0x5a; 64], key),
  ] {
    assert_eq!(
      MlKem512DecapsulationKey::from_pkcs8_der(&one_asymmetric_key(&[0], &oid, &invalid, &[])).err(),
      Some(MlKemKeyError::InvalidDecapsulationKey)
    );
  }

  let mut null_parameters = oid.clone();
  null_parameters.extend([0x05, 0x00]);
  let mut trailing = SEED_512.to_vec();
  trailing.push(0);
  let seed_der = seed_choice(&seed);
  for malformed in [
    one_asymmetric_key(&[0], &null_parameters, &seed_der, &[]),
    one_asymmetric_key(&[2], &oid, &seed_der, &[]),
    one_asymmetric_key(&[0, 0], &oid, &seed_der, &[]),
    one_asymmetric_key(&[0], &oid, &seed_der, &[&public_key_field(0, ek.as_bytes())]),
    one_asymmetric_key(&[1], &oid, &seed_der, &[]),
    one_asymmetric_key(&[1], &oid, &seed_der, &[&public_key_field(1, ek.as_bytes())]),
    one_asymmetric_key(&[0], &oid, &[seed_der.as_slice(), &[0]].concat(), &[]),
    one_asymmetric_key(&[0], &oid, &tlv(0xa0, &tlv(0x04, &seed)), &[]),
    one_asymmetric_key(&[0], &oid, &[], &[]),
    trailing,
  ] {
    assert_eq!(
      MlKem512DecapsulationKey::from_pkcs8_der(&malformed).err(),
      Some(MlKemKeyError::MalformedDer)
    );
  }
  for der in [SEED_512, EXPANDED_512, BOTH_512] {
    for len in 0..der.len() {
      assert_eq!(
        MlKem512DecapsulationKey::from_pkcs8_der(&der[..len]).err(),
        Some(MlKemKeyError::MalformedDer),
        "truncated to {len} bytes"
      );
    }
  }
}

/// Change every byte outside the `[start, end)` payloads to every other value; each
/// change must be rejected by `parse`.
fn every_framing_byte_is_required(der: &[u8], payloads: &[(usize, usize)], parse: impl Fn(&[u8]) -> bool) {
  let mut input = der.to_vec();
  for position in
    (0..der.len()).filter(|&position| !payloads.iter().any(|&(start, end)| (start..end).contains(&position)))
  {
    let original = input[position];
    for value in (0..=u8::MAX).filter(|&value| value != original) {
      input[position] = value;
      assert!(!parse(&input), "byte {position} = {value:#04x} was accepted");
    }
    input[position] = original;
  }
}

#[test]
fn every_framing_byte_is_required_for_each_encoding() {
  let spki = |der: &[u8]| MlKem512EncapsulationKey::from_spki_der(der).is_ok();
  let pkcs8 = |der: &[u8]| MlKem512DecapsulationKey::from_pkcs8_der(der).is_ok();
  every_framing_byte_is_required(SPKI_512, &[(22, SPKI_512.len())], spki);
  every_framing_byte_is_required(SEED_512, &[(22, 86)], pkcs8);
  every_framing_byte_is_required(EXPANDED_512, &[(28, EXPANDED_512.len())], pkcs8);
  // Header and seed OCTET STRING header, the seed, the key's OCTET STRING header, then the key.
  every_framing_byte_is_required(BOTH_512, &[(30, 94), (98, BOTH_512.len())], pkcs8);
}

#[test]
fn seed_owner_generation_matches_key_generation_and_redacts_debug() {
  let generated = MlKem768Seed::generate(|out| {
    out.fill(0x5a);
    Ok(())
  })
  .expect("seed generation");
  let (ek, dk) = generated.keypair();
  let (expected_ek, expected_dk) = MlKem768::generate_keypair(|out| {
    out.fill(0x5a);
    Ok::<(), MlKemError>(())
  })
  .expect("key generation");
  assert_eq!(ek.as_bytes(), expected_ek.as_bytes());
  assert_eq!(dk.as_bytes(), expected_dk.as_bytes());
  let failed = MlKem768Seed::generate(|out| {
    out[..13].fill(0xff);
    Err(MlKemError::RandomGenerationFailed)
  });
  assert_eq!(failed.err(), Some(MlKemError::RandomGenerationFailed));
  assert_eq!(format!("{generated:?}"), "MlKem768Seed(****)");
}

#[test]
fn pkix_matches_rustcrypto_ml_kem() {
  use rustcrypto_ml_kem::{
    self as oracle,
    pkcs8::{DecodePrivateKey, DecodePublicKey, EncodePrivateKey, EncodePublicKey},
  };

  macro_rules! exchange {
    ($ek:ident, $dk:ident, $seed:ident, $oracle:ident) => {
      for seed_bytes in [[0; 64], [0x5a; 64], [0xff; 64]] {
        let theirs = oracle::DecapsulationKey::<oracle::$oracle>::from_seed(seed_bytes.into());
        let their_spki = theirs
          .encapsulation_key()
          .to_public_key_der()
          .expect("oracle SPKI export");
        let their_pkcs8 = theirs.to_pkcs8_der().expect("oracle PKCS #8 export");

        let seed = $seed::from_pkcs8_der(their_pkcs8.as_bytes()).expect("oracle seed import");
        assert_eq!(seed.expose_secret().as_bytes(), &seed_bytes);
        let (ek, _) = seed.keypair();
        assert_eq!(ek.to_spki_der().as_slice(), their_spki.as_bytes());
        assert_eq!(
          $ek::from_spki_der(their_spki.as_bytes())
            .expect("oracle SPKI import")
            .as_bytes(),
          ek.as_bytes()
        );
        assert_eq!(
          $dk::from_pkcs8_der(their_pkcs8.as_bytes())
            .expect("oracle key import")
            .as_bytes(),
          seed.keypair().1.as_bytes()
        );

        let mut ours = [0; $seed::PKCS8_DER_LENGTH];
        seed.to_pkcs8_der_into(&mut ours);
        assert_eq!(ours.as_slice(), their_pkcs8.as_bytes());
        let imported = oracle::DecapsulationKey::<oracle::$oracle>::from_pkcs8_der(&ours).expect("oracle import");
        assert_eq!(
          imported
            .encapsulation_key()
            .to_public_key_der()
            .expect("oracle SPKI export")
            .as_bytes(),
          their_spki.as_bytes()
        );
        let their_ek = oracle::EncapsulationKey::<oracle::$oracle>::from_public_key_der(&ek.to_spki_der())
          .expect("oracle SPKI import");
        assert_eq!(
          their_ek.to_public_key_der().expect("oracle SPKI export").as_bytes(),
          ek.to_spki_der().as_slice()
        );
      }
    };
  }

  exchange!(
    MlKem512EncapsulationKey,
    MlKem512DecapsulationKey,
    MlKem512Seed,
    MlKem512
  );
  exchange!(
    MlKem768EncapsulationKey,
    MlKem768DecapsulationKey,
    MlKem768Seed,
    MlKem768
  );
  exchange!(
    MlKem1024EncapsulationKey,
    MlKem1024DecapsulationKey,
    MlKem1024Seed,
    MlKem1024
  );
}

/// OpenSSL 3.5 and later implement RFC 9935. Exchange every form with the
/// `openssl` CLI when it supports ML-KEM; otherwise report the skip.
#[test]
fn pkix_matches_openssl_cli_when_available() {
  use std::process::Command;

  fn openssl(args: &[&str]) -> Option<Vec<u8>> {
    let output = Command::new("openssl").args(args).output().ok()?;
    output.status.success().then_some(output.stdout)
  }

  let supported = openssl(&["list", "-kem-algorithms"])
    .is_some_and(|list| String::from_utf8_lossy(&list).contains("id-alg-ml-kem-1024"));
  if !supported {
    eprintln!("skipping the OpenSSL ML-KEM PKIX exchange because `openssl` lacks ML-KEM");
    return;
  }
  let directory = std::env::temp_dir().join(format!("rscrypto-mlkem-pkix-{}", std::process::id()));
  std::fs::create_dir_all(&directory).expect("temporary directory");
  let path = |name: &str| directory.join(name).to_str().expect("UTF-8 temporary path").to_owned();

  macro_rules! exchange {
    ($algorithm:literal, $ek:ident, $dk:ident, $seed:ident) => {
      for format in ["seed-only", "priv-only", "seed-priv"] {
        let key = path(&format!("{}-{format}.der", $algorithm));
        let format_param = format!("ml-kem.output_formats={format}");
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
        let dk = $dk::from_pkcs8_der(&der).expect("OpenSSL key import");
        let ek = $ek::from_spki_der(&spki).expect("OpenSSL SPKI import");
        assert_eq!(ek.to_spki_der().as_slice(), spki.as_slice(), "{format}");
        assert_eq!(
          &dk.as_bytes()[dk.as_bytes().len() - 64 - ek.as_bytes().len()..dk.as_bytes().len() - 64],
          ek.as_bytes().as_slice(),
          "{format}"
        );
        assert_eq!(
          $seed::from_pkcs8_der(&der).err(),
          if format == "priv-only" {
            Some(MlKemKeyError::UnsupportedEncoding)
          } else {
            None
          },
          "{format}"
        );

        let mut expanded = [0; $dk::PKCS8_DER_LENGTH];
        dk.to_pkcs8_der_into(&mut expanded);
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

  exchange!(
    "ML-KEM-512",
    MlKem512EncapsulationKey,
    MlKem512DecapsulationKey,
    MlKem512Seed
  );
  exchange!(
    "ML-KEM-768",
    MlKem768EncapsulationKey,
    MlKem768DecapsulationKey,
    MlKem768Seed
  );
  exchange!(
    "ML-KEM-1024",
    MlKem1024EncapsulationKey,
    MlKem1024DecapsulationKey,
    MlKem1024Seed
  );
  std::fs::remove_dir_all(&directory).expect("remove temporary keys");
}

#[cfg(feature = "alloc")]
#[test]
fn allocated_imports_match_by_value_imports() {
  extern crate alloc;
  use alloc::alloc::Global;

  macro_rules! allocated {
    ($dk:ident, $seed:ident, $seed_der:ident, $expanded_der:ident, $both_der:ident) => {
      let (ek, dk) = $seed::from_bytes(rfc9935_seed()).keypair();
      let (boxed_ek, boxed_dk) = $seed::from_bytes(rfc9935_seed()).keypair_in(Global);
      assert_eq!(boxed_ek.as_bytes(), ek.as_bytes());
      assert_eq!(boxed_dk.as_bytes(), dk.as_bytes());
      for der in [$seed_der, $expanded_der, $both_der] {
        let boxed = $dk::from_pkcs8_der_in(der, Global).expect("allocated PKCS #8 import");
        assert_eq!(boxed.as_bytes(), dk.as_bytes());
        assert_eq!(
          $dk::from_pkcs8_der_in(&der[..der.len() - 1], Global).err(),
          Some(MlKemKeyError::MalformedDer)
        );
      }
    };
  }

  allocated!(MlKem512DecapsulationKey, MlKem512Seed, SEED_512, EXPANDED_512, BOTH_512);
  allocated!(MlKem768DecapsulationKey, MlKem768Seed, SEED_768, EXPANDED_768, BOTH_768);
  allocated!(
    MlKem1024DecapsulationKey,
    MlKem1024Seed,
    SEED_1024,
    EXPANDED_1024,
    BOTH_1024
  );
  for der in [
    fixture!("mlkem512_pkcs8_inconsistent_seed"),
    fixture!("mlkem512_pkcs8_inconsistent_s"),
    fixture!("mlkem512_pkcs8_inconsistent_hash"),
    fixture!("mlkem512_pkcs8_inconsistent_z"),
  ] {
    assert_eq!(
      MlKem512DecapsulationKey::from_pkcs8_der_in(der, Global).err(),
      Some(MlKemKeyError::InvalidDecapsulationKey)
    );
  }
}
