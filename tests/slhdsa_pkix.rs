//! RFC 9909 SubjectPublicKeyInfo and PKCS #8 import and export of SLH-DSA keys.
#![cfg(feature = "slh-dsa")]

use rscrypto::*;

macro_rules! fixture {
  ($name:literal) => {
    include_bytes!(concat!("../testdata/slhdsa/rfc9909/", $name, ".der")).as_slice()
  };
}

const SPKI: &[u8] = fixture!("slhdsa_sha2_128s_spki");
const PKCS8: &[u8] = fixture!("slhdsa_sha2_128s_pkcs8");
const CERTIFICATE: &[u8] = fixture!("slhdsa_sha2_128s_certificate");

/// `rsaEncryption`, 1.2.840.113549.1.1.1.
const RSA_ENCRYPTION: [u8; 9] = [0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01];
/// RFC 8410 section 10.1 Ed25519 public key.
const ED25519_SPKI: [u8; 44] = [
  0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00, 0x19, 0xbf, 0x44, 0x09, 0x69, 0x84, 0xcd,
  0xfe, 0x85, 0x41, 0xba, 0xc1, 0x67, 0xdc, 0x3b, 0x96, 0xc8, 0x50, 0x86, 0xaa, 0x30, 0xb6, 0xb6, 0xcb, 0x0c, 0x5c,
  0x38, 0xad, 0x70, 0x31, 0x66, 0xe1,
];

/// DER contents of 2.16.840.1.101.3.4.3.`arc` (RFC 9909 section 3).
fn sig_algs_oid(arc: u8) -> Vec<u8> {
  vec![0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x03, arc]
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

/// OneAsymmetricKey with the given version contents, AlgorithmIdentifier
/// contents, `privateKey` contents, and trailing fields.
fn one_asymmetric_key(version: &[u8], algorithm: &[u8], private_key: &[u8], trailing: &[&[u8]]) -> Vec<u8> {
  let mut contents = [tlv(0x02, version), tlv(0x30, algorithm), tlv(0x04, private_key)].concat();
  for field in trailing {
    contents.extend_from_slice(field);
  }
  tlv(0x30, &contents)
}

fn public_key_field(unused_bits: u8, key: &[u8]) -> Vec<u8> {
  tlv(0x81, &[&[unused_bits], key].concat())
}

#[test]
fn rfc9909_examples_import_export_and_verify() {
  let public = SlhDsaSha2_128sPublicKey::from_spki_der(SPKI).expect("RFC 9909 SPKI");
  assert_eq!(SlhDsaSha2_128sPublicKey::SPKI_DER_LENGTH, SPKI.len());
  assert_eq!(public.to_spki_der().as_slice(), SPKI);

  let secret = SlhDsaSha2_128sSecretKey::from_pkcs8_der(PKCS8).expect("RFC 9909 private key");
  assert_eq!(secret.public_key(), &public);
  assert_eq!(SlhDsaSha2_128sSecretKey::PKCS8_DER_LENGTH, PKCS8.len());
  let mut exported = [0; SlhDsaSha2_128sSecretKey::PKCS8_DER_LENGTH];
  secret.to_pkcs8_der_into(&mut exported);
  assert_eq!(exported.as_slice(), PKCS8);

  // Certificate ::= SEQUENCE { tbsCertificate, signatureAlgorithm, signature }.
  // The TBSCertificate SEQUENCE is at offset 4 with a four-byte header and
  // 359 bytes of contents; the signature algorithm follows it.
  let tbs = &CERTIFICATE[4..367];
  let signature = &CERTIFICATE[CERTIFICATE.len().strict_sub(SlhDsaSha2_128s::SIGNATURE_LENGTH)..];
  assert_eq!(&CERTIFICATE[367..380], &tlv(0x30, &tlv(0x06, &sig_algs_oid(20)))[..]);
  public.verify(tbs, signature).expect("RFC 9909 certificate signature");
  let mut altered = tbs.to_vec();
  altered[20] ^= 1;
  assert!(public.verify(&altered, signature).is_err());
  let hash_public = HashSlhDsaSha2_128sWithSha256PublicKey::from_bytes(public.to_bytes());
  assert!(hash_public.verify(tbs, signature).is_err());
}

/// Every profile's export equals DER built from its RFC 9909 OID, and that
/// DER imports under this profile only.
macro_rules! profile_encodings {
  ($test:ident, $profile:ident, $public:ident, $secret:ident, $arc:literal, $sibling_arc:literal, $n:literal) => {
    #[test]
    fn $test() {
      let (public, secret) = $profile::generate_keypair(|out| {
        out.fill($arc);
        Ok(())
      })
      .expect("key generation");
      let algorithm = tlv(0x06, &sig_algs_oid($arc));
      let expected_spki = spki(&algorithm, 0, public.as_bytes());
      assert_eq!(public.to_spki_der().as_slice(), expected_spki);
      assert_eq!(&$public::from_spki_der(&expected_spki).expect("SPKI"), &public);

      let encoded = secret.expose_secret();
      let expected_pkcs8 = one_asymmetric_key(&[0], &algorithm, encoded.as_bytes(), &[]);
      let mut exported = [0; $secret::PKCS8_DER_LENGTH];
      secret.to_pkcs8_der_into(&mut exported);
      assert_eq!(exported.as_slice(), expected_pkcs8);
      assert_eq!(
        $secret::from_pkcs8_der(&expected_pkcs8)
          .expect("PKCS #8")
          .expose_secret()
          .as_bytes(),
        encoded.as_bytes()
      );
      let version_2 = one_asymmetric_key(
        &[1],
        &algorithm,
        encoded.as_bytes(),
        &[&public_key_field(0, public.as_bytes())],
      );
      assert_eq!(
        $secret::from_pkcs8_der(&version_2).expect("version 2").public_key(),
        &public
      );

      // The pure or HashSLH-DSA sibling and the other hash family's set.
      let other_family = match $arc {
        20..=25 | 35..=40 => $arc + 6,
        _ => $arc - 6,
      };
      for arc in [$sibling_arc, other_family] {
        let other = tlv(0x06, &sig_algs_oid(arc));
        assert_eq!(
          $public::from_spki_der(&spki(&other, 0, public.as_bytes())).err(),
          Some(SlhDsaKeyError::UnsupportedAlgorithm)
        );
        assert_eq!(
          $secret::from_pkcs8_der(&one_asymmetric_key(&[0], &other, encoded.as_bytes(), &[])).err(),
          Some(SlhDsaKeyError::UnsupportedAlgorithm)
        );
      }
      assert_eq!($public::LENGTH, 2 * $n);
      assert_eq!($secret::LENGTH, 4 * $n);
    }
  };
}

profile_encodings!(
  encodings_sha2_128s,
  SlhDsaSha2_128s,
  SlhDsaSha2_128sPublicKey,
  SlhDsaSha2_128sSecretKey,
  20,
  35,
  16
);
profile_encodings!(
  encodings_sha2_128f,
  SlhDsaSha2_128f,
  SlhDsaSha2_128fPublicKey,
  SlhDsaSha2_128fSecretKey,
  21,
  36,
  16
);
profile_encodings!(
  encodings_sha2_192s,
  SlhDsaSha2_192s,
  SlhDsaSha2_192sPublicKey,
  SlhDsaSha2_192sSecretKey,
  22,
  37,
  24
);
profile_encodings!(
  encodings_sha2_192f,
  SlhDsaSha2_192f,
  SlhDsaSha2_192fPublicKey,
  SlhDsaSha2_192fSecretKey,
  23,
  38,
  24
);
profile_encodings!(
  encodings_sha2_256s,
  SlhDsaSha2_256s,
  SlhDsaSha2_256sPublicKey,
  SlhDsaSha2_256sSecretKey,
  24,
  39,
  32
);
profile_encodings!(
  encodings_sha2_256f,
  SlhDsaSha2_256f,
  SlhDsaSha2_256fPublicKey,
  SlhDsaSha2_256fSecretKey,
  25,
  40,
  32
);
profile_encodings!(
  encodings_shake_128s,
  SlhDsaShake128s,
  SlhDsaShake128sPublicKey,
  SlhDsaShake128sSecretKey,
  26,
  41,
  16
);
profile_encodings!(
  encodings_shake_128f,
  SlhDsaShake128f,
  SlhDsaShake128fPublicKey,
  SlhDsaShake128fSecretKey,
  27,
  42,
  16
);
profile_encodings!(
  encodings_shake_192s,
  SlhDsaShake192s,
  SlhDsaShake192sPublicKey,
  SlhDsaShake192sSecretKey,
  28,
  43,
  24
);
profile_encodings!(
  encodings_shake_192f,
  SlhDsaShake192f,
  SlhDsaShake192fPublicKey,
  SlhDsaShake192fSecretKey,
  29,
  44,
  24
);
profile_encodings!(
  encodings_shake_256s,
  SlhDsaShake256s,
  SlhDsaShake256sPublicKey,
  SlhDsaShake256sSecretKey,
  30,
  45,
  32
);
profile_encodings!(
  encodings_shake_256f,
  SlhDsaShake256f,
  SlhDsaShake256fPublicKey,
  SlhDsaShake256fSecretKey,
  31,
  46,
  32
);
profile_encodings!(
  encodings_hash_sha2_128s,
  HashSlhDsaSha2_128sWithSha256,
  HashSlhDsaSha2_128sWithSha256PublicKey,
  HashSlhDsaSha2_128sWithSha256SecretKey,
  35,
  20,
  16
);
profile_encodings!(
  encodings_hash_sha2_128f,
  HashSlhDsaSha2_128fWithSha256,
  HashSlhDsaSha2_128fWithSha256PublicKey,
  HashSlhDsaSha2_128fWithSha256SecretKey,
  36,
  21,
  16
);
profile_encodings!(
  encodings_hash_sha2_192s,
  HashSlhDsaSha2_192sWithSha512,
  HashSlhDsaSha2_192sWithSha512PublicKey,
  HashSlhDsaSha2_192sWithSha512SecretKey,
  37,
  22,
  24
);
profile_encodings!(
  encodings_hash_sha2_192f,
  HashSlhDsaSha2_192fWithSha512,
  HashSlhDsaSha2_192fWithSha512PublicKey,
  HashSlhDsaSha2_192fWithSha512SecretKey,
  38,
  23,
  24
);
profile_encodings!(
  encodings_hash_sha2_256s,
  HashSlhDsaSha2_256sWithSha512,
  HashSlhDsaSha2_256sWithSha512PublicKey,
  HashSlhDsaSha2_256sWithSha512SecretKey,
  39,
  24,
  32
);
profile_encodings!(
  encodings_hash_sha2_256f,
  HashSlhDsaSha2_256fWithSha512,
  HashSlhDsaSha2_256fWithSha512PublicKey,
  HashSlhDsaSha2_256fWithSha512SecretKey,
  40,
  25,
  32
);
profile_encodings!(
  encodings_hash_shake_128s,
  HashSlhDsaShake128sWithShake128,
  HashSlhDsaShake128sWithShake128PublicKey,
  HashSlhDsaShake128sWithShake128SecretKey,
  41,
  26,
  16
);
profile_encodings!(
  encodings_hash_shake_128f,
  HashSlhDsaShake128fWithShake128,
  HashSlhDsaShake128fWithShake128PublicKey,
  HashSlhDsaShake128fWithShake128SecretKey,
  42,
  27,
  16
);
profile_encodings!(
  encodings_hash_shake_192s,
  HashSlhDsaShake192sWithShake256,
  HashSlhDsaShake192sWithShake256PublicKey,
  HashSlhDsaShake192sWithShake256SecretKey,
  43,
  28,
  24
);
profile_encodings!(
  encodings_hash_shake_192f,
  HashSlhDsaShake192fWithShake256,
  HashSlhDsaShake192fWithShake256PublicKey,
  HashSlhDsaShake192fWithShake256SecretKey,
  44,
  29,
  24
);
profile_encodings!(
  encodings_hash_shake_256s,
  HashSlhDsaShake256sWithShake256,
  HashSlhDsaShake256sWithShake256PublicKey,
  HashSlhDsaShake256sWithShake256SecretKey,
  45,
  30,
  32
);
profile_encodings!(
  encodings_hash_shake_256f,
  HashSlhDsaShake256fWithShake256,
  HashSlhDsaShake256fWithShake256PublicKey,
  HashSlhDsaShake256fWithShake256SecretKey,
  46,
  31,
  32
);

#[test]
fn spki_rejections_name_the_failing_component() {
  let key = &SPKI[18..];
  let oid = tlv(0x06, &sig_algs_oid(20));
  assert_eq!(spki(&oid, 0, key), SPKI);

  let mut rsa_algorithm = tlv(0x06, &RSA_ENCRYPTION);
  rsa_algorithm.extend([0x05, 0x00]);
  let mldsa_spki = include_bytes!("../testdata/mldsa/rfc9881/mldsa44_spki.der");
  for other in [ED25519_SPKI.to_vec(), mldsa_spki.to_vec(), spki(&rsa_algorithm, 0, key)] {
    assert_eq!(
      SlhDsaSha2_128sPublicKey::from_spki_der(&other).err(),
      Some(SlhDsaKeyError::UnsupportedAlgorithm)
    );
  }

  let mut key_too_long = key.to_vec();
  key_too_long.push(0);
  for invalid in [&key[1..], key_too_long.as_slice(), &[]] {
    assert_eq!(
      SlhDsaSha2_128sPublicKey::from_spki_der(&spki(&oid, 0, invalid)).err(),
      Some(SlhDsaKeyError::InvalidPublicKey)
    );
  }

  let mut null_parameters = oid.clone();
  null_parameters.extend([0x05, 0x00]);
  let mut trailing = SPKI.to_vec();
  trailing.push(0);
  let mut long_form = vec![0x30, 0x81, SPKI[1]];
  long_form.extend_from_slice(&SPKI[2..]);
  let mut set_tag = SPKI.to_vec();
  set_tag[0] = 0x31;
  for malformed in [
    spki(&null_parameters, 0, key),
    spki(&oid, 1, key),
    trailing,
    long_form,
    set_tag,
  ] {
    assert_eq!(
      SlhDsaSha2_128sPublicKey::from_spki_der(&malformed).err(),
      Some(SlhDsaKeyError::MalformedDer)
    );
  }
  for len in 0..SPKI.len() {
    assert_eq!(
      SlhDsaSha2_128sPublicKey::from_spki_der(&SPKI[..len]).err(),
      Some(SlhDsaKeyError::MalformedDer),
      "truncated to {len} bytes"
    );
  }
}

#[test]
fn pkcs8_rejections_name_the_failing_component() {
  let secret = SlhDsaSha2_128sSecretKey::from_pkcs8_der(PKCS8).expect("RFC 9909 private key");
  let encoded = secret.expose_secret();
  let key = encoded.as_bytes().as_slice();
  let public = secret.public_key().as_bytes().as_slice();
  let oid = tlv(0x06, &sig_algs_oid(20));
  assert_eq!(one_asymmetric_key(&[0], &oid, key, &[]), PKCS8);

  let (other_public, _) = SlhDsaSha2_128s::generate_keypair(|out| {
    out.fill(0x5a);
    Ok(())
  })
  .expect("key generation");
  let matching = one_asymmetric_key(&[1], &oid, key, &[&public_key_field(0, public)]);
  assert_eq!(
    SlhDsaSha2_128sSecretKey::from_pkcs8_der(&matching)
      .expect("version 2 import")
      .public_key()
      .as_bytes()
      .as_slice(),
    public
  );
  let mismatched = one_asymmetric_key(&[1], &oid, key, &[&public_key_field(0, other_public.as_bytes())]);
  assert_eq!(
    SlhDsaSha2_128sSecretKey::from_pkcs8_der(&mismatched).err(),
    Some(SlhDsaKeyError::InvalidSecretKey)
  );
  let short = one_asymmetric_key(&[1], &oid, key, &[&public_key_field(0, &public[1..])]);
  assert_eq!(
    SlhDsaSha2_128sSecretKey::from_pkcs8_der(&short).err(),
    Some(SlhDsaKeyError::InvalidPublicKey)
  );

  let mut rsa_algorithm = tlv(0x06, &RSA_ENCRYPTION);
  rsa_algorithm.extend([0x05, 0x00]);
  for other in [
    include_bytes!("../testdata/mldsa/rfc9881/mldsa44_pkcs8_seed.der").to_vec(),
    one_asymmetric_key(&[0], &rsa_algorithm, key, &[]),
  ] {
    assert_eq!(
      SlhDsaSha2_128sSecretKey::from_pkcs8_der(&other).err(),
      Some(SlhDsaKeyError::UnsupportedAlgorithm)
    );
  }

  let attributes = tlv(0xa0, &tlv(0x30, &[]));
  assert_eq!(
    SlhDsaSha2_128sSecretKey::from_pkcs8_der(&one_asymmetric_key(&[0], &oid, key, &[&attributes])).err(),
    Some(SlhDsaKeyError::UnsupportedEncoding)
  );

  let mut key_too_long = key.to_vec();
  key_too_long.push(0);
  let mut wrong_root = key.to_vec();
  wrong_root[63] ^= 1;
  for invalid in [
    &key[1..],
    key_too_long.as_slice(),
    wrong_root.as_slice(),
    // An RFC 8410-style inner OCTET STRING is not RFC 9909's form.
    &tlv(0x04, key),
  ] {
    assert_eq!(
      SlhDsaSha2_128sSecretKey::from_pkcs8_der(&one_asymmetric_key(&[0], &oid, invalid, &[])).err(),
      Some(SlhDsaKeyError::InvalidSecretKey)
    );
  }

  let mut null_parameters = oid.clone();
  null_parameters.extend([0x05, 0x00]);
  let mut trailing = PKCS8.to_vec();
  trailing.push(0);
  for malformed in [
    one_asymmetric_key(&[0], &null_parameters, key, &[]),
    one_asymmetric_key(&[2], &oid, key, &[]),
    one_asymmetric_key(&[0, 0], &oid, key, &[]),
    one_asymmetric_key(&[0], &oid, key, &[&public_key_field(0, public)]),
    one_asymmetric_key(&[1], &oid, key, &[]),
    one_asymmetric_key(&[1], &oid, key, &[&public_key_field(1, public)]),
    one_asymmetric_key(&[1], &oid, key, &[&public_key_field(0, public), &[0x05, 0x00]]),
    trailing,
  ] {
    assert_eq!(
      SlhDsaSha2_128sSecretKey::from_pkcs8_der(&malformed).err(),
      Some(SlhDsaKeyError::MalformedDer)
    );
  }
  for len in 0..PKCS8.len() {
    assert_eq!(
      SlhDsaSha2_128sSecretKey::from_pkcs8_der(&PKCS8[..len]).err(),
      Some(SlhDsaKeyError::MalformedDer),
      "truncated to {len} bytes"
    );
  }
}

/// Change every byte outside the `[start, end)` payload to every other value;
/// each change must be rejected by `parse`.
fn every_framing_byte_is_required(der: &[u8], payload: (usize, usize), parse: impl Fn(&[u8]) -> bool) {
  let mut input = der.to_vec();
  for position in (0..der.len()).filter(|&position| !(payload.0..payload.1).contains(&position)) {
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
  every_framing_byte_is_required(SPKI, (18, SPKI.len()), |der| {
    SlhDsaSha2_128sPublicKey::from_spki_der(der).is_ok()
  });
  // Import checks the PKCS #8 payload by regenerating its root; the sweeps
  // cover the framing. The 128-byte SHAKE-256f key uses long-form lengths.
  every_framing_byte_is_required(PKCS8, (20, PKCS8.len()), |der| {
    SlhDsaSha2_128sSecretKey::from_pkcs8_der(der).is_ok()
  });
  let (public, secret) = SlhDsaShake256f::generate_keypair(|out| {
    out.fill(0x31);
    Ok(())
  })
  .expect("key generation");
  let spki = public.to_spki_der();
  every_framing_byte_is_required(&spki, (18, spki.len()), |der| {
    SlhDsaShake256fPublicKey::from_spki_der(der).is_ok()
  });
  let mut pkcs8 = [0; SlhDsaShake256fSecretKey::PKCS8_DER_LENGTH];
  secret.to_pkcs8_der_into(&mut pkcs8);
  assert_eq!(&pkcs8[..3], &[0x30, 0x81, 0x93]);
  every_framing_byte_is_required(&pkcs8, (22, pkcs8.len()), |der| {
    SlhDsaShake256fSecretKey::from_pkcs8_der(der).is_ok()
  });
}

#[test]
fn pkix_matches_rustcrypto_slh_dsa() {
  use p256::pkcs8::{DecodePrivateKey, DecodePublicKey, EncodePrivateKey, EncodePublicKey};
  use rustcrypto_slh_dsa as oracle;

  macro_rules! exchange {
    ($profile:ident, $public:ident, $secret:ident, $oracle:ident, $n:literal) => {
      let seeds = [0x17; 3 * $n];
      let theirs =
        oracle::SigningKey::<oracle::$oracle>::slh_keygen_internal(&seeds[..$n], &seeds[$n..2 * $n], &seeds[2 * $n..]);
      let their_spki = theirs.as_ref().to_public_key_der().expect("oracle SPKI export");
      let their_pkcs8 = theirs.to_pkcs8_der().expect("oracle PKCS #8 export");

      let secret = $secret::from_pkcs8_der(their_pkcs8.as_bytes()).expect("oracle key import");
      let public = $public::from_spki_der(their_spki.as_bytes()).expect("oracle SPKI import");
      assert_eq!(secret.public_key(), &public);
      assert_eq!(public.to_spki_der().as_slice(), their_spki.as_bytes());
      let mut ours = [0; $secret::PKCS8_DER_LENGTH];
      secret.to_pkcs8_der_into(&mut ours);
      assert_eq!(ours.as_slice(), their_pkcs8.as_bytes());

      let imported = oracle::SigningKey::<oracle::$oracle>::from_pkcs8_der(&ours).expect("oracle import");
      assert_eq!(imported.to_bytes().as_slice(), secret.expose_secret().as_bytes());
      let imported =
        oracle::VerifyingKey::<oracle::$oracle>::from_public_key_der(&public.to_spki_der()).expect("oracle import");
      assert_eq!(imported.to_bytes().as_slice(), public.as_bytes());
    };
  }

  exchange!(
    SlhDsaSha2_128s,
    SlhDsaSha2_128sPublicKey,
    SlhDsaSha2_128sSecretKey,
    Sha2_128s,
    16
  );
  exchange!(
    SlhDsaSha2_128f,
    SlhDsaSha2_128fPublicKey,
    SlhDsaSha2_128fSecretKey,
    Sha2_128f,
    16
  );
  exchange!(
    SlhDsaSha2_192s,
    SlhDsaSha2_192sPublicKey,
    SlhDsaSha2_192sSecretKey,
    Sha2_192s,
    24
  );
  exchange!(
    SlhDsaSha2_192f,
    SlhDsaSha2_192fPublicKey,
    SlhDsaSha2_192fSecretKey,
    Sha2_192f,
    24
  );
  exchange!(
    SlhDsaSha2_256s,
    SlhDsaSha2_256sPublicKey,
    SlhDsaSha2_256sSecretKey,
    Sha2_256s,
    32
  );
  exchange!(
    SlhDsaSha2_256f,
    SlhDsaSha2_256fPublicKey,
    SlhDsaSha2_256fSecretKey,
    Sha2_256f,
    32
  );
  exchange!(
    SlhDsaShake128s,
    SlhDsaShake128sPublicKey,
    SlhDsaShake128sSecretKey,
    Shake128s,
    16
  );
  exchange!(
    SlhDsaShake128f,
    SlhDsaShake128fPublicKey,
    SlhDsaShake128fSecretKey,
    Shake128f,
    16
  );
  exchange!(
    SlhDsaShake192s,
    SlhDsaShake192sPublicKey,
    SlhDsaShake192sSecretKey,
    Shake192s,
    24
  );
  exchange!(
    SlhDsaShake192f,
    SlhDsaShake192fPublicKey,
    SlhDsaShake192fSecretKey,
    Shake192f,
    24
  );
  exchange!(
    SlhDsaShake256s,
    SlhDsaShake256sPublicKey,
    SlhDsaShake256sSecretKey,
    Shake256s,
    32
  );
  exchange!(
    SlhDsaShake256f,
    SlhDsaShake256fPublicKey,
    SlhDsaShake256fSecretKey,
    Shake256f,
    32
  );
}

/// OpenSSL 3.5 and later implement the pure RFC 9909 sets. Exchange keys and
/// signatures with the `openssl` CLI when it supports SLH-DSA; otherwise
/// report the skip. OpenSSL has no HashSLH-DSA.
#[test]
fn pkix_and_signatures_match_openssl_cli_when_available() {
  use std::process::Command;

  fn openssl(args: &[&str]) -> Option<Vec<u8>> {
    let output = Command::new("openssl").args(args).output().ok()?;
    output.status.success().then_some(output.stdout)
  }

  let supported = openssl(&["list", "-signature-algorithms"])
    .is_some_and(|list| String::from_utf8_lossy(&list).contains("id-slh-dsa-shake-256f"));
  if !supported {
    eprintln!("skipping the OpenSSL SLH-DSA exchange because `openssl` lacks SLH-DSA");
    return;
  }
  let directory = std::env::temp_dir().join(format!("rscrypto-slhdsa-pkix-{}", std::process::id()));
  std::fs::create_dir_all(&directory).expect("temporary directory");
  let path = |name: &str| directory.join(name).to_str().expect("UTF-8 temporary path").to_owned();
  let message = path("message.bin");
  std::fs::write(&message, b"rscrypto and OpenSSL exchange").expect("message file");

  macro_rules! exchange {
    ($algorithm:literal, $public:ident, $secret:ident, $profile:ident) => {
      let key = path(concat!($algorithm, ".der"));
      openssl(&["genpkey", "-algorithm", $algorithm, "-outform", "DER", "-out", &key]).expect("OpenSSL key generation");
      let spki = openssl(&["pkey", "-in", &key, "-inform", "DER", "-pubout", "-outform", "DER"]).expect("OpenSSL SPKI");
      let der = std::fs::read(&key).expect("OpenSSL key file");
      let secret = $secret::from_pkcs8_der(&der).expect("OpenSSL key import");
      let public = $public::from_spki_der(&spki).expect("OpenSSL SPKI import");
      assert_eq!(secret.public_key(), &public);
      assert_eq!(public.to_spki_der().as_slice(), spki.as_slice());

      let mut exported = [0; $secret::PKCS8_DER_LENGTH];
      secret.to_pkcs8_der_into(&mut exported);
      assert_eq!(exported.as_slice(), der.as_slice());
      let exported_path = path(concat!($algorithm, "-rscrypto.der"));
      std::fs::write(&exported_path, exported).expect("export file");
      let parsed = openssl(&[
        "pkey",
        "-in",
        &exported_path,
        "-inform",
        "DER",
        "-pubout",
        "-outform",
        "DER",
      ])
      .expect("OpenSSL parses the export");
      assert_eq!(parsed, spki);

      // OpenSSL signs with a context; rscrypto verifies. Then the reverse.
      let theirs = openssl(&[
        "pkeyutl",
        "-sign",
        "-rawin",
        "-inkey",
        &key,
        "-keyform",
        "DER",
        "-in",
        &message,
        "-pkeyopt",
        "context-string:openssl",
      ])
      .expect("OpenSSL signing");
      let text = std::fs::read(&message).expect("message");
      public
        .verify_with_context(&text, b"openssl", &theirs)
        .expect("OpenSSL signature verifies");
      let mut ours = [0; $profile::SIGNATURE_LENGTH];
      secret.sign_deterministic(&text, b"", &mut ours).expect("signing");
      let signature = path(concat!($algorithm, ".sig"));
      std::fs::write(&signature, ours).expect("signature file");
      let deterministic = openssl(&[
        "pkeyutl",
        "-sign",
        "-rawin",
        "-inkey",
        &key,
        "-keyform",
        "DER",
        "-in",
        &message,
        "-pkeyopt",
        "deterministic:1",
      ])
      .expect("OpenSSL deterministic signing");
      assert_eq!(deterministic, ours.to_vec());
      let spki_path = path(concat!($algorithm, "-spki.der"));
      std::fs::write(&spki_path, &spki).expect("SPKI file");
      openssl(&[
        "pkeyutl", "-verify", "-rawin", "-pubin", "-inkey", &spki_path, "-keyform", "DER", "-in", &message, "-sigfile",
        &signature,
      ])
      .expect("OpenSSL verifies the rscrypto signature");
    };
  }

  exchange!(
    "SLH-DSA-SHA2-128s",
    SlhDsaSha2_128sPublicKey,
    SlhDsaSha2_128sSecretKey,
    SlhDsaSha2_128s
  );
  exchange!(
    "SLH-DSA-SHA2-128f",
    SlhDsaSha2_128fPublicKey,
    SlhDsaSha2_128fSecretKey,
    SlhDsaSha2_128f
  );
  exchange!(
    "SLH-DSA-SHA2-192s",
    SlhDsaSha2_192sPublicKey,
    SlhDsaSha2_192sSecretKey,
    SlhDsaSha2_192s
  );
  exchange!(
    "SLH-DSA-SHA2-192f",
    SlhDsaSha2_192fPublicKey,
    SlhDsaSha2_192fSecretKey,
    SlhDsaSha2_192f
  );
  exchange!(
    "SLH-DSA-SHA2-256s",
    SlhDsaSha2_256sPublicKey,
    SlhDsaSha2_256sSecretKey,
    SlhDsaSha2_256s
  );
  exchange!(
    "SLH-DSA-SHA2-256f",
    SlhDsaSha2_256fPublicKey,
    SlhDsaSha2_256fSecretKey,
    SlhDsaSha2_256f
  );
  exchange!(
    "SLH-DSA-SHAKE-128s",
    SlhDsaShake128sPublicKey,
    SlhDsaShake128sSecretKey,
    SlhDsaShake128s
  );
  exchange!(
    "SLH-DSA-SHAKE-128f",
    SlhDsaShake128fPublicKey,
    SlhDsaShake128fSecretKey,
    SlhDsaShake128f
  );
  exchange!(
    "SLH-DSA-SHAKE-192s",
    SlhDsaShake192sPublicKey,
    SlhDsaShake192sSecretKey,
    SlhDsaShake192s
  );
  exchange!(
    "SLH-DSA-SHAKE-192f",
    SlhDsaShake192fPublicKey,
    SlhDsaShake192fSecretKey,
    SlhDsaShake192f
  );
  exchange!(
    "SLH-DSA-SHAKE-256s",
    SlhDsaShake256sPublicKey,
    SlhDsaShake256sSecretKey,
    SlhDsaShake256s
  );
  exchange!(
    "SLH-DSA-SHAKE-256f",
    SlhDsaShake256fPublicKey,
    SlhDsaShake256fSecretKey,
    SlhDsaShake256f
  );
  std::fs::remove_dir_all(&directory).expect("remove temporary keys");
}
