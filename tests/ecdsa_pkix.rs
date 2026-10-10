//! RFC 5958 PKCS #8 and RFC 5915 SEC1 private keys, RFC 5480 SubjectPublicKeyInfo
//! export, and DER signatures for ECDSA P-256 and P-384.
#![cfg(any(feature = "ecdsa-p256", feature = "ecdsa-p384"))]

use rscrypto::EcdsaError;

mod common;
use common::decode_hex_vec;

/// DER contents of `id-ecPublicKey`, 1.2.840.10045.2.1 (RFC 5480 section 2.1.1).
const ID_EC_PUBLIC_KEY: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01];
/// `prime256v1`, 1.2.840.10045.3.1.7.
const SECP256R1: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07];
/// `secp384r1`, 1.3.132.0.34.
const SECP384R1: &[u8] = &[0x2b, 0x81, 0x04, 0x00, 0x22];
/// `rsaEncryption`, 1.2.840.113549.1.1.1.
const RSA_ENCRYPTION: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01];

fn tlv(tag: u8, value: &[u8]) -> Vec<u8> {
  let [.., high, low] = value.len().to_be_bytes();
  let mut out = vec![tag];
  match value.len() {
    0..=0x7f => out.push(low),
    0x80..=0xff => out.extend([0x81, low]),
    _ => out.extend([0x82, high, low]),
  }
  out.extend_from_slice(value);
  out
}

fn concat(parts: &[&[u8]]) -> Vec<u8> {
  parts.concat()
}

fn bit_string(key: &[u8]) -> Vec<u8> {
  tlv(0x03, &concat(&[&[0], key]))
}

/// AlgorithmIdentifier `id-ecPublicKey` with `namedCurve` parameters.
fn ec_identifier(curve: &[u8]) -> Vec<u8> {
  tlv(0x30, &concat(&[&tlv(0x06, ID_EC_PUBLIC_KEY), &tlv(0x06, curve)]))
}

/// RFC 5915 ECPrivateKey with optional `[0]` parameters and `[1]` public key.
fn ec_private_key(version: u8, scalar: &[u8], parameters: Option<&[u8]>, public: Option<&[u8]>) -> Vec<u8> {
  let mut fields = concat(&[&tlv(0x02, &[version]), &tlv(0x04, scalar)]);
  if let Some(curve) = parameters {
    fields.extend(tlv(0xa0, &tlv(0x06, curve)));
  }
  if let Some(public) = public {
    fields.extend(tlv(0xa1, &bit_string(public)));
  }
  tlv(0x30, &fields)
}

/// RFC 5958 OneAsymmetricKey: version, identifier, privateKey, then `rest`.
fn one_asymmetric_key(version: u8, identifier: &[u8], private_key: &[u8], rest: &[u8]) -> Vec<u8> {
  tlv(
    0x30,
    &concat(&[&tlv(0x02, &[version]), identifier, &tlv(0x04, private_key), rest]),
  )
}

/// Scalars below each group order: small, high-bit, and leading-zero values.
fn scalar_samples<const N: usize>() -> Vec<[u8; N]> {
  let last = N.strict_sub(1);
  let mut samples = vec![[0; N]; 4];
  samples[0][last] = 1;
  samples[1][0] = 0x7f;
  samples[1][last] = 0x5a;
  samples[2][1] = 0xc3;
  samples[2][last] = 0x11;
  for (index, byte) in samples[3].iter_mut().enumerate() {
    *byte = u8::try_from(index)
      .expect("index fits a byte")
      .wrapping_mul(37)
      .wrapping_add(1)
      & 0x7f;
  }
  samples
}

macro_rules! curve_tests {
  (
    $module:ident,
    $feature:literal,
    $secret:ident,
    $public:ident,
    $signature:ident,
    $oracle:ident,
    $curve:expr,
    $other_curve:expr,
    $scalar_len:literal,
    $openssl_curve:literal,
    $order:literal,
    $ring_signing:ident,
    $ring_verify:ident,
    $digest:ident
  ) => {
    #[cfg(feature = $feature)]
    mod $module {
      use ring::{rand::SystemRandom, signature as ring_signature, signature::KeyPair as _};
      use rscrypto::{$public, $secret, $signature};
      use sha2::Digest as _;
      use $oracle::{
        ecdsa::Signature as OracleSignature,
        pkcs8::{DecodePrivateKey as _, EncodePrivateKey as _, EncodePublicKey as _},
      };

      use super::*;

      fn oracle(scalar: &[u8; $scalar_len]) -> $oracle::SecretKey {
        $oracle::SecretKey::from_slice(scalar).expect("oracle scalar")
      }

      fn oracle_public(scalar: &[u8; $scalar_len]) -> Vec<u8> {
        oracle(scalar).public_key().to_sec1_bytes().to_vec()
      }

      #[test]
      fn keys_match_rustcrypto_in_both_directions() {
        for scalar in scalar_samples::<$scalar_len>() {
          let oracle = oracle(&scalar);
          let oracle_pkcs8 = oracle.to_pkcs8_der().expect("oracle PKCS #8");
          let imported = $secret::from_pkcs8_der(oracle_pkcs8.as_bytes()).expect("RustCrypto PKCS #8 import");
          assert_eq!(imported.expose_secret().as_bytes(), &scalar);
          let oracle_sec1 = oracle.to_sec1_der().expect("oracle SEC1");
          let imported = $secret::from_sec1_der(&oracle_sec1).expect("RustCrypto SEC1 import");
          assert_eq!(imported.expose_secret().as_bytes(), &scalar);

          let mut exported = [0; $secret::PKCS8_DER_LENGTH];
          imported.to_pkcs8_der_into(&mut exported);
          let parsed = $oracle::SecretKey::from_pkcs8_der(&exported).expect("RustCrypto parses the export");
          assert_eq!(parsed.to_bytes().as_slice(), scalar.as_slice());
          assert_eq!(
            exported.as_slice(),
            oracle_pkcs8.as_bytes(),
            "RustCrypto writes the same form"
          );

          let spki = oracle.public_key().to_public_key_der().expect("oracle SPKI");
          assert_eq!(imported.public_key().to_spki_der().as_slice(), spki.as_bytes());
          assert_eq!($public::SPKI_DER_LENGTH, spki.as_bytes().len());
        }
      }

      #[test]
      fn keys_and_der_signatures_match_ring_in_both_directions() {
        let rng = SystemRandom::new();
        for _ in 0..8 {
          let generated = ring_signature::EcdsaKeyPair::generate_pkcs8(&ring_signature::$ring_signing, &rng)
            .expect("ring key generation");
          let pair = ring_signature::EcdsaKeyPair::from_pkcs8(&ring_signature::$ring_signing, generated.as_ref(), &rng)
            .expect("ring key");
          let secret = $secret::from_pkcs8_der(generated.as_ref()).expect("ring PKCS #8 import");
          let public = secret.public_key();
          assert_eq!(public.to_sec1_bytes().as_slice(), pair.public_key().as_ref());

          let mut exported = [0; $secret::PKCS8_DER_LENGTH];
          secret.to_pkcs8_der_into(&mut exported);
          let reparsed = ring_signature::EcdsaKeyPair::from_pkcs8(&ring_signature::$ring_signing, &exported, &rng)
            .expect("ring parses the export");
          assert_eq!(reparsed.public_key().as_ref(), pair.public_key().as_ref());

          let message = b"rscrypto ECDSA DER signature exchange";
          let signature = secret.try_sign(message).expect("signing");
          let mut buffer = [0; $signature::DER_MAX_LENGTH];
          let der = signature.to_der_into(&mut buffer);
          ring_signature::UnparsedPublicKey::new(&ring_signature::$ring_verify, pair.public_key().as_ref())
            .verify(message, der)
            .expect("ring verifies the DER signature");
          let ring_der = pair.sign(&rng, message).expect("ring signing");
          let ring_signature = $signature::from_der(ring_der.as_ref()).expect("ring DER signature");
          public
            .verify(message, &ring_signature)
            .expect("verifies the ring signature");
        }
      }

      #[test]
      fn der_signatures_match_rustcrypto() {
        let half: usize = $scalar_len;
        let mut cases: Vec<Vec<u8>> = Vec::new();
        for r in scalar_samples::<$scalar_len>() {
          for s in scalar_samples::<$scalar_len>() {
            cases.push(concat(&[&r, &s]));
          }
        }
        // Both scalars at full width with the sign bit set: the longest encoding.
        let mut longest = vec![0xc0; half.strict_mul(2)];
        longest[half.strict_sub(1)] = 0x01;
        longest[half.strict_mul(2).strict_sub(1)] = 0x01;
        cases.push(longest);
        let secret = $secret::from_bytes(scalar_samples::<$scalar_len>()[3]).expect("secret");
        for index in 0..32_u8 {
          cases.push(secret.try_sign(&[index]).expect("signing").to_bytes().to_vec());
        }

        let mut longest_seen = 0;
        for raw in cases {
          let signature = $signature::from_bytes(raw.as_slice().try_into().expect("raw length")).expect("signature");
          let mut buffer = [0; $signature::DER_MAX_LENGTH];
          let der = signature.to_der_into(&mut buffer).to_vec();
          let expected = OracleSignature::from_slice(&raw).expect("oracle signature").to_der();
          assert_eq!(der.as_slice(), expected.as_bytes(), "{raw:02x?}");
          assert_eq!($signature::from_der(&der), Ok(signature));
          longest_seen = longest_seen.max(der.len());
        }
        assert_eq!(longest_seen, $signature::DER_MAX_LENGTH);
      }

      #[test]
      fn pkcs8_rejects_each_violation() {
        let scalar = scalar_samples::<$scalar_len>()[3];
        let public = oracle_public(&scalar);
        let foreign = oracle_public(&scalar_samples::<$scalar_len>()[2]);
        let identifier = ec_identifier($curve);
        let inner = ec_private_key(1, &scalar, None, Some(&public));
        let valid = one_asymmetric_key(0, &identifier, &inner, &[]);
        assert_eq!(
          $secret::from_pkcs8_der(&valid).map(|key| *key.expose_secret().as_bytes()),
          Ok(scalar)
        );

        let order = decode_hex_vec($order);
        let mut long_scalar = scalar.to_vec();
        long_scalar.push(0);
        let compressed = &public[..=$scalar_len];
        let mut wrong_tag = valid.clone();
        wrong_tag[0] = 0x31;
        let mut non_minimal_length = vec![0x30, 0x81];
        non_minimal_length.extend(&valid[1..]);

        let cases: Vec<(&str, Vec<u8>, EcdsaError)> = vec![
          ("outer tag", wrong_tag, EcdsaError::MalformedDer),
          ("non-minimal length", non_minimal_length, EcdsaError::MalformedDer),
          ("trailing data", concat(&[&valid, &[0]]), EcdsaError::MalformedDer),
          (
            "unknown version",
            one_asymmetric_key(2, &identifier, &inner, &[]),
            EcdsaError::MalformedDer,
          ),
          (
            "version 2 without a public key",
            one_asymmetric_key(1, &identifier, &inner, &[]),
            EcdsaError::MalformedDer,
          ),
          (
            "version 1 with a public key",
            one_asymmetric_key(0, &identifier, &inner, &tlv(0x81, &concat(&[&[0], &public]))),
            EcdsaError::MalformedDer,
          ),
          (
            "another algorithm",
            one_asymmetric_key(
              0,
              &tlv(0x30, &concat(&[&tlv(0x06, RSA_ENCRYPTION), &[0x05, 0x00]])),
              &inner,
              &[],
            ),
            EcdsaError::UnsupportedAlgorithm,
          ),
          (
            "another curve",
            one_asymmetric_key(0, &ec_identifier($other_curve), &inner, &[]),
            EcdsaError::UnsupportedAlgorithm,
          ),
          (
            "absent curve parameters",
            one_asymmetric_key(0, &tlv(0x30, &tlv(0x06, ID_EC_PUBLIC_KEY)), &inner, &[]),
            EcdsaError::MalformedDer,
          ),
          (
            "implicit curve",
            one_asymmetric_key(
              0,
              &tlv(0x30, &concat(&[&tlv(0x06, ID_EC_PUBLIC_KEY), &[0x05, 0x00]])),
              &inner,
              &[],
            ),
            EcdsaError::MalformedDer,
          ),
          (
            "explicit curve parameters",
            one_asymmetric_key(
              0,
              &tlv(
                0x30,
                &concat(&[&tlv(0x06, ID_EC_PUBLIC_KEY), &tlv(0x30, &tlv(0x02, &[1]))]),
              ),
              &inner,
              &[],
            ),
            EcdsaError::MalformedDer,
          ),
          (
            "attributes",
            one_asymmetric_key(0, &identifier, &inner, &tlv(0xa0, &[])),
            EcdsaError::UnsupportedEncoding,
          ),
          (
            "ECPrivateKey version 0",
            one_asymmetric_key(0, &identifier, &ec_private_key(0, &scalar, None, Some(&public)), &[]),
            EcdsaError::MalformedDer,
          ),
          (
            "short scalar",
            one_asymmetric_key(0, &identifier, &ec_private_key(1, &scalar[1..], None, None), &[]),
            EcdsaError::InvalidSecretKey,
          ),
          (
            "long scalar",
            one_asymmetric_key(0, &identifier, &ec_private_key(1, &long_scalar, None, None), &[]),
            EcdsaError::InvalidSecretKey,
          ),
          (
            "zero scalar",
            one_asymmetric_key(0, &identifier, &ec_private_key(1, &[0; $scalar_len], None, None), &[]),
            EcdsaError::InvalidSecretKey,
          ),
          (
            "scalar equal to the group order",
            one_asymmetric_key(0, &identifier, &ec_private_key(1, &order, None, None), &[]),
            EcdsaError::InvalidSecretKey,
          ),
          (
            "inner parameters for another curve",
            one_asymmetric_key(
              0,
              &identifier,
              &ec_private_key(1, &scalar, Some($other_curve), Some(&public)),
              &[],
            ),
            EcdsaError::UnsupportedAlgorithm,
          ),
          (
            "foreign inner public key",
            one_asymmetric_key(0, &identifier, &ec_private_key(1, &scalar, None, Some(&foreign)), &[]),
            EcdsaError::InvalidSecretKey,
          ),
          (
            "compressed inner public key",
            one_asymmetric_key(0, &identifier, &ec_private_key(1, &scalar, None, Some(compressed)), &[]),
            EcdsaError::InvalidPublicKey,
          ),
          (
            "inner public key with unused bits",
            one_asymmetric_key(
              0,
              &identifier,
              &tlv(
                0x30,
                &concat(&[
                  &tlv(0x02, &[1]),
                  &tlv(0x04, &scalar),
                  &tlv(0xa1, &tlv(0x03, &concat(&[&[1], &public]))),
                ]),
              ),
              &[],
            ),
            EcdsaError::MalformedDer,
          ),
          (
            "inner fields out of order",
            one_asymmetric_key(
              0,
              &identifier,
              &tlv(
                0x30,
                &concat(&[
                  &tlv(0x02, &[1]),
                  &tlv(0x04, &scalar),
                  &tlv(0xa1, &bit_string(&public)),
                  &tlv(0xa0, &tlv(0x06, $curve)),
                ]),
              ),
              &[],
            ),
            EcdsaError::MalformedDer,
          ),
          (
            "foreign container public key",
            one_asymmetric_key(1, &identifier, &inner, &tlv(0x81, &concat(&[&[0], &foreign]))),
            EcdsaError::InvalidSecretKey,
          ),
        ];
        for (name, der, expected) in cases {
          assert_eq!($secret::from_pkcs8_der(&der).err(), Some(expected), "{name}");
        }

        for (name, der) in [
          (
            "matching inner parameters",
            one_asymmetric_key(
              0,
              &identifier,
              &ec_private_key(1, &scalar, Some($curve), Some(&public)),
              &[],
            ),
          ),
          (
            "no inner public key",
            one_asymmetric_key(0, &identifier, &ec_private_key(1, &scalar, None, None), &[]),
          ),
          (
            "matching container public key",
            one_asymmetric_key(1, &identifier, &inner, &tlv(0x81, &concat(&[&[0], &public]))),
          ),
        ] {
          assert_eq!(
            $secret::from_pkcs8_der(&der).map(|key| *key.expose_secret().as_bytes()),
            Ok(scalar),
            "{name}"
          );
        }
      }

      #[test]
      fn sec1_accepts_optional_fields_and_rejects_violations() {
        let scalar = scalar_samples::<$scalar_len>()[1];
        let public = oracle_public(&scalar);
        let foreign = oracle_public(&scalar_samples::<$scalar_len>()[0]);
        for (parameters, carried) in [
          (Some($curve), Some(public.as_slice())),
          (None, Some(public.as_slice())),
          (Some($curve), None),
          (None, None),
        ] {
          let der = ec_private_key(1, &scalar, parameters, carried);
          assert_eq!(
            $secret::from_sec1_der(&der).map(|key| *key.expose_secret().as_bytes()),
            Ok(scalar)
          );
        }

        let valid = ec_private_key(1, &scalar, Some($curve), Some(&public));
        for (name, der, expected) in [
          ("trailing data", concat(&[&valid, &[0]]), EcdsaError::MalformedDer),
          (
            "version 2",
            ec_private_key(2, &scalar, Some($curve), None),
            EcdsaError::MalformedDer,
          ),
          (
            "another curve",
            ec_private_key(1, &scalar, Some($other_curve), None),
            EcdsaError::UnsupportedAlgorithm,
          ),
          (
            "foreign public key",
            ec_private_key(1, &scalar, Some($curve), Some(&foreign)),
            EcdsaError::InvalidSecretKey,
          ),
          (
            "a PKCS #8 container",
            one_asymmetric_key(0, &ec_identifier($curve), &valid, &[]),
            EcdsaError::MalformedDer,
          ),
        ] {
          assert_eq!($secret::from_sec1_der(&der).err(), Some(expected), "{name}");
        }
      }

      /// Exchange keys and signatures with the `openssl` CLI when it is
      /// installed; otherwise report the skip.
      #[test]
      fn keys_and_signatures_match_openssl_cli_when_available() {
        use std::process::Command;

        fn openssl(args: &[&str]) -> Option<Vec<u8>> {
          let output = Command::new("openssl").args(args).output().ok()?;
          output.status.success().then_some(output.stdout)
        }

        if openssl(&["version"]).is_none() {
          eprintln!("skipping the OpenSSL ECDSA key exchange because `openssl` is unavailable");
          return;
        }
        let directory =
          std::env::temp_dir().join(format!("rscrypto-{}-pkix-{}", stringify!($module), std::process::id()));
        std::fs::create_dir_all(&directory).expect("temporary directory");
        let path = |name: &str| directory.join(name).to_str().expect("UTF-8 temporary path").to_owned();
        let curve_name = concat!("ec_paramgen_curve:", $openssl_curve);

        let key = path("key.der");
        openssl(&[
          "genpkey",
          "-algorithm",
          "EC",
          "-pkeyopt",
          curve_name,
          "-outform",
          "DER",
          "-out",
          &key,
        ])
        .expect("OpenSSL key generation");
        // OpenSSL 3 writes DER EC keys as SEC1 and OpenSSL 4 as PKCS #8, so request PKCS #8.
        let pkcs8 = openssl(&[
          "pkcs8", "-topk8", "-nocrypt", "-inform", "DER", "-in", &key, "-outform", "DER",
        ])
        .expect("OpenSSL PKCS #8");
        let spki =
          openssl(&["pkey", "-in", &key, "-inform", "DER", "-pubout", "-outform", "DER"]).expect("OpenSSL SPKI");
        let sec1 = openssl(&["ec", "-in", &key, "-inform", "DER", "-outform", "DER"]).expect("OpenSSL SEC1");

        let secret = $secret::from_pkcs8_der(&pkcs8).expect("OpenSSL PKCS #8 import");
        let from_sec1 = $secret::from_sec1_der(&sec1).expect("OpenSSL SEC1 import");
        assert_eq!(from_sec1.expose_secret().as_bytes(), secret.expose_secret().as_bytes());
        assert_eq!(secret.public_key().to_spki_der().as_slice(), spki.as_slice());
        assert_eq!(
          $public::from_spki_der(&spki).map(|key| key.to_sec1_bytes()),
          Ok(secret.public_key().to_sec1_bytes())
        );

        let mut exported = [0; $secret::PKCS8_DER_LENGTH];
        secret.to_pkcs8_der_into(&mut exported);
        assert_eq!(exported.as_slice(), pkcs8.as_slice(), "OpenSSL writes the same form");
        let exported_path = path("rscrypto.der");
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

        let message = b"rscrypto ECDSA OpenSSL exchange";
        let digest = sha2::$digest::digest(message);
        let mut buffer = [0; $signature::DER_MAX_LENGTH];
        let der = secret
          .try_sign(message)
          .expect("signing")
          .to_der_into(&mut buffer)
          .to_vec();
        let (digest_path, signature_path, public_path) =
          (path("digest.bin"), path("signature.der"), path("public.der"));
        std::fs::write(&digest_path, digest).expect("digest file");
        std::fs::write(&signature_path, &der).expect("signature file");
        std::fs::write(&public_path, &spki).expect("public key file");
        openssl(&[
          "pkeyutl",
          "-verify",
          "-pubin",
          "-inkey",
          &public_path,
          "-keyform",
          "DER",
          "-in",
          &digest_path,
          "-sigfile",
          &signature_path,
        ])
        .expect("OpenSSL verifies the DER signature");

        std::fs::remove_dir_all(&directory).expect("remove temporary keys");
      }
    }
  };
}

curve_tests!(
  p256_keys,
  "ecdsa-p256",
  EcdsaP256SecretKey,
  EcdsaP256PublicKey,
  EcdsaP256Signature,
  p256,
  SECP256R1,
  SECP384R1,
  32,
  "P-256",
  "ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551",
  ECDSA_P256_SHA256_ASN1_SIGNING,
  ECDSA_P256_SHA256_ASN1,
  Sha256
);
curve_tests!(
  p384_keys,
  "ecdsa-p384",
  EcdsaP384SecretKey,
  EcdsaP384PublicKey,
  EcdsaP384Signature,
  p384,
  SECP384R1,
  SECP256R1,
  48,
  "P-384",
  "ffffffffffffffffffffffffffffffffffffffffffffffffc7634d81f4372ddf581a0db248b0a77aecec196accc52973",
  ECDSA_P384_SHA384_ASN1_SIGNING,
  ECDSA_P384_SHA384_ASN1,
  Sha384
);
