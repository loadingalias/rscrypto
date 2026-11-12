//! Composite ML-DSA (draft-ietf-lamps-pq-composite-sigs-19) against the
//! draft's Appendix E vectors, with hostile encodings and API equivalences.
#![cfg(all(feature = "mldsa65-ed25519", feature = "mldsa65-ecdsa-p256", feature = "alloc"))]

extern crate alloc;

use alloc::alloc::Global;
use core::alloc::{AllocError, Allocator, Layout};
use core::cell::Cell;
use core::ptr::NonNull;

use rscrypto::*;
use serde_json::Value;

/// Counts allocations and deallocations that pass through to `Global`.
#[derive(Default)]
struct Live {
  allocations: Cell<usize>,
  deallocations: Cell<usize>,
}

impl Live {
  fn outstanding(&self) -> usize {
    self.allocations.get().strict_sub(self.deallocations.get())
  }
}

// SAFETY: every allocation and deallocation, with its original layout, passes
// through to Global unchanged; this allocator only counts them.
unsafe impl Allocator for &Live {
  fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    let allocation = Global.allocate(layout)?;
    self.allocations.set(self.allocations.get().strict_add(1));
    Ok(allocation)
  }

  unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) {
    self.deallocations.set(self.deallocations.get().strict_add(1));
    // SAFETY: Global made this allocation with this layout in `allocate`.
    unsafe { Global.deallocate(pointer, layout) };
  }
}

const VECTORS: &str = include_str!("../testdata/composite_mldsa/draft19/vectors.json");

fn base64(input: &str) -> Vec<u8> {
  const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  let digits: Vec<u8> = input
    .bytes()
    .filter(|&byte| byte != b'=')
    .map(|byte| u8::try_from(ALPHABET.iter().position(|&c| c == byte).expect("base64 digit")).expect("digit"))
    .collect();
  let mut out = Vec::new();
  for chunk in digits.chunks(4) {
    let mut word = 0u32;
    for (i, &digit) in chunk.iter().enumerate() {
      word |= u32::from(digit) << 18usize.strict_sub(i.strict_mul(6));
    }
    let bytes = word.to_be_bytes();
    out.extend_from_slice(&bytes[1..chunk.len()]);
  }
  out
}

struct Case {
  pk: Vec<u8>,
  sk: Vec<u8>,
  sk_pkcs8: Vec<u8>,
  x5c: Vec<u8>,
  s: Vec<u8>,
  s_with_context: Vec<u8>,
}

struct Vectors {
  message: Vec<u8>,
  context: Vec<u8>,
  ed25519: Case,
  ecdsa_p256: Case,
}

fn vectors() -> Vectors {
  let root: Value = serde_json::from_str(VECTORS).expect("vector JSON");
  let field = |value: &Value, name: &str| base64(value[name].as_str().expect("base64 field"));
  let case = |id: &str| {
    let test = root["tests"]
      .as_array()
      .expect("tests")
      .iter()
      .find(|test| test["tcId"] == id)
      .expect("test case");
    Case {
      pk: field(test, "pk"),
      sk: field(test, "sk"),
      sk_pkcs8: field(test, "sk_pkcs8"),
      x5c: field(test, "x5c"),
      s: field(test, "s"),
      s_with_context: field(test, "sWithContext"),
    }
  };
  Vectors {
    message: field(&root, "m"),
    context: field(&root, "ctx"),
    ed25519: case("id-MLDSA65-Ed25519-SHA512"),
    ecdsa_p256: case("id-MLDSA65-ECDSA-P256-SHA512"),
  }
}

/// One DER element: its tag, its contents, and the input after it.
fn element(der: &[u8]) -> (u8, &[u8], &[u8]) {
  let tag = der[0];
  let (len, header) = if der[1] < 0x80 {
    (usize::from(der[1]), 2usize)
  } else if der[1] == 0x81 {
    (usize::from(der[2]), 3)
  } else {
    assert_eq!(der[1], 0x82, "a DER length of at most two octets");
    (usize::from(u16::from_be_bytes([der[2], der[3]])), 4)
  };
  let end = header.strict_add(len);
  (tag, &der[header..end], &der[end..])
}

/// The full encoding of the first element of `der`.
fn element_bytes(der: &[u8]) -> &[u8] {
  let (_, _, rest) = element(der);
  &der[..der.len().strict_sub(rest.len())]
}

/// The tbsCertificate encoding, the signatureAlgorithm contents, the
/// signature bits, and the subjectPublicKeyInfo encoding of a certificate.
fn certificate(der: &[u8]) -> (&[u8], &[u8], &[u8], &[u8]) {
  let (_, certificate, _) = element(der);
  let tbs = element_bytes(certificate);
  let (_, algorithm, rest) = element(&certificate[tbs.len()..]);
  let (tag, bits, _) = element(rest);
  assert_eq!(tag, 0x03);
  assert_eq!(bits[0], 0, "no unused bits");
  // version [0], serialNumber, signature, issuer, validity, subject, then SPKI.
  let (_, mut fields, _) = element(tbs);
  for _ in 0..6 {
    fields = element(fields).2;
  }
  (tbs, algorithm, &bits[1..], element_bytes(fields))
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

/// DER contents of 1.3.6.1.5.5.7.6.`arc`.
fn composite_oid(arc: u8) -> Vec<u8> {
  vec![0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x06, arc]
}

fn spki(oid: &[u8], parameters: &[u8], unused_bits: u8, key: &[u8]) -> Vec<u8> {
  let mut bits = vec![unused_bits];
  bits.extend_from_slice(key);
  let algorithm = [tlv(0x06, oid), parameters.to_vec()].concat();
  tlv(0x30, &[tlv(0x30, &algorithm), tlv(0x03, &bits)].concat())
}

fn pkcs8(version: u8, oid: &[u8], private_key: &[u8], trailing: &[&[u8]]) -> Vec<u8> {
  let mut contents = [
    tlv(0x02, &[version]),
    tlv(0x30, &tlv(0x06, oid)),
    tlv(0x04, private_key),
  ]
  .concat();
  for field in trailing {
    contents.extend_from_slice(field);
  }
  tlv(0x30, &contents)
}

fn fixed_fill(byte: u8) -> impl FnMut(&mut [u8]) -> Result<(), CompositeMlDsaError> {
  move |out| {
    out.fill(byte);
    Ok(())
  }
}

/// Both composites share these checks; the macro instantiates them with
/// each algorithm's types and vectors.
macro_rules! composite_tests {
  ($module:ident, $profile:ident, $public:ident, $secret:ident, $signature:ident, $case:ident, $arc:literal) => {
    mod $module {
      use super::*;

      fn case() -> (Vectors, $public, $secret) {
        let vectors = vectors();
        let public = $public::try_from_slice(&vectors.$case.pk).expect("vector public key");
        let secret = $secret::try_from_slice(&vectors.$case.sk).expect("vector secret key");
        (vectors, public, secret)
      }

      #[test]
      fn identifiers_match_the_draft() {
        assert_eq!($profile::OID, concat!("1.3.6.1.5.5.7.6.", stringify!($arc)));
        let label = core::str::from_utf8($profile::LABEL).expect("ASCII label");
        let name = stringify!($profile);
        assert!(
          label.starts_with("COMPSIG-MLDSA65-") && label.ends_with("-SHA512"),
          "{name}: {label}"
        );
      }

      #[test]
      fn draft_signatures_verify() {
        let (vectors, public, _) = case();
        let case = &vectors.$case;
        let plain = $signature::try_from_slice(&case.s).expect("vector signature");
        let with_context = $signature::try_from_slice(&case.s_with_context).expect("vector signature");
        public.verify(&vectors.message, &plain).expect("empty-context vector");
        public
          .verify_with_context(&vectors.message, &vectors.context, &with_context)
          .expect("context vector");
        // Each signature binds its own context.
        assert!(
          public
            .verify_with_context(&vectors.message, &vectors.context, &plain)
            .is_err()
        );
        assert!(public.verify(&vectors.message, &with_context).is_err());
        assert!(public.verify(b"another message", &plain).is_err());
        assert_eq!(plain.as_ref(), case.s.as_slice());
      }

      #[test]
      fn every_component_must_verify() {
        let (vectors, public, _) = case();
        let encoded = &vectors.$case.s;
        // A changed byte in the ML-DSA challenge, or in the traditional
        // signature's last scalar byte, fails the whole signature.
        for index in [0, encoded.len() - 1] {
          let mut altered = encoded.clone();
          altered[index] ^= 1;
          if let Ok(signature) = $signature::try_from_slice(&altered) {
            assert!(public.verify(&vectors.message, &signature).is_err(), "byte {index}");
          }
        }
      }

      #[test]
      fn components_are_not_standalone_signatures() {
        // Stripping: neither component verifies as an ordinary signature
        // over the message, because both sign the representative M'.
        let (vectors, _, _) = case();
        let case = &vectors.$case;
        let ml_dsa_public = MlDsa65PublicKey::try_from_slice(&case.pk[..MlDsa65PublicKey::LENGTH]).expect("ML-DSA key");
        let ml_dsa = MlDsa65Signature::try_from_slice(&case.s[..MlDsa65Signature::LENGTH]).expect("ML-DSA signature");
        assert!(ml_dsa_public.verify(&vectors.message, &ml_dsa).is_err());
        assert!(
          ml_dsa_public
            .verify_with_context(&vectors.message, $profile::LABEL, &ml_dsa)
            .is_err()
        );
      }

      #[test]
      fn draft_keys_import_and_export_byte_for_byte() {
        let (vectors, public, secret) = case();
        let case = &vectors.$case;
        assert_eq!(public.to_bytes().as_slice(), case.pk.as_slice());
        assert_eq!(secret.public_key(), public);
        assert_eq!(secret.expose_secret().as_bytes().as_slice(), case.sk.as_slice());

        let imported = $secret::from_pkcs8_der(&case.sk_pkcs8).expect("vector PKCS #8");
        assert_eq!(imported.expose_secret().as_bytes(), secret.expose_secret().as_bytes());
        let mut der = [0; $secret::PKCS8_DER_LENGTH];
        imported.to_pkcs8_der_into(&mut der);
        assert_eq!(der.as_slice(), case.sk_pkcs8.as_slice());
      }

      #[test]
      fn draft_certificate_spki_and_signature() {
        let (vectors, public, _) = case();
        let (tbs, algorithm, signature, spki) = certificate(&vectors.$case.x5c);
        assert_eq!(
          algorithm,
          tlv(0x06, &composite_oid($arc)).as_slice(),
          "signatureAlgorithm"
        );
        assert_eq!($public::from_spki_der(spki).expect("certificate SPKI"), public);
        assert_eq!(public.to_spki_der().as_slice(), spki);
        let signature = $signature::try_from_slice(signature).expect("certificate signature");
        public.verify(tbs, &signature).expect("self-signed certificate");
        let mut altered = tbs.to_vec();
        *altered.last_mut().expect("TBS") ^= 1;
        assert!(public.verify(&altered, &signature).is_err());
      }

      #[test]
      fn imported_key_signs_for_the_vector_public_key() {
        let (vectors, public, secret) = case();
        for context in [&[][..], &vectors.context, &[0x5a; 255][..]] {
          let deterministic = secret.sign_deterministic(&vectors.message, context).expect("sign");
          public
            .verify_with_context(&vectors.message, context, &deterministic)
            .expect("deterministic");
          assert_eq!(
            secret.sign_deterministic(&vectors.message, context).expect("sign"),
            deterministic,
            "deterministic signing is repeatable"
          );
          let hedged = secret
            .sign_with(&vectors.message, context, fixed_fill(0x33))
            .expect("sign");
          public
            .verify_with_context(&vectors.message, context, &hedged)
            .expect("hedged");
          assert_ne!(hedged, deterministic, "hedging changes the ML-DSA component");
          let reparsed = $signature::try_from_slice(hedged.as_ref()).expect("own encoding parses");
          assert_eq!(reparsed, hedged);
        }
      }

      #[test]
      fn contexts_longer_than_255_bytes_are_rejected() {
        let (vectors, public, secret) = case();
        let long = [0; 256];
        assert_eq!(
          secret
            .sign_deterministic(&vectors.message, &long)
            .expect_err("rejected"),
          CompositeMlDsaError::ContextTooLong
        );
        let mut requested = false;
        let result = secret.sign_with(&vectors.message, &long, |_| {
          requested = true;
          Ok(())
        });
        assert_eq!(result.expect_err("rejected"), CompositeMlDsaError::ContextTooLong);
        assert!(!requested, "the context is checked before randomness is requested");
        let signature = $signature::try_from_slice(&vectors.$case.s).expect("vector signature");
        assert!(public.verify_with_context(&vectors.message, &long, &signature).is_err());
      }

      #[test]
      fn random_failure_is_returned_and_the_key_stays_usable() {
        let (vectors, public, secret) = case();
        let result = secret.sign_with(&vectors.message, b"", |out| {
          let half = out.len() / 2;
          out[..half].fill(0x42);
          Err(CompositeMlDsaError::RandomGenerationFailed)
        });
        assert_eq!(
          result.expect_err("rejected"),
          CompositeMlDsaError::RandomGenerationFailed
        );
        let signature = secret.sign_with(&vectors.message, b"", fixed_fill(1)).expect("sign");
        public
          .verify(&vectors.message, &signature)
          .expect("verify after failure");
      }

      #[test]
      fn prehash_and_prepared_signing_match_ordinary_signing() {
        let (vectors, public, secret) = case();
        let digest = Sha512::digest(&vectors.message);
        let expected = secret
          .sign_deterministic(&vectors.message, &vectors.context)
          .expect("sign");
        assert_eq!(
          secret
            .sign_prehash_deterministic(&digest, &vectors.context)
            .expect("sign"),
          expected
        );
        public
          .verify_prehash(&digest, &vectors.context, &expected)
          .expect("prehash verification");
        let hedged = secret
          .sign_with(&vectors.message, &vectors.context, fixed_fill(9))
          .expect("sign");
        assert_eq!(
          secret
            .sign_prehash_with(&digest, &vectors.context, fixed_fill(9))
            .expect("sign"),
          hedged
        );

        let mut storage = MlDsa65PreparedSecretKeyStorage::new();
        let prepared = secret.prepare(&mut storage).expect("prepare");
        assert_eq!(
          prepared
            .sign_deterministic(&vectors.message, &vectors.context)
            .expect("sign"),
          expected
        );
        assert_eq!(
          prepared
            .sign_with(&vectors.message, &vectors.context, fixed_fill(9))
            .expect("sign"),
          hedged
        );
        drop(prepared);
        let retained = secret.prepare_in(alloc::alloc::Global).expect("prepare");
        drop(secret);
        assert_eq!(
          retained
            .sign_deterministic(&vectors.message, &vectors.context)
            .expect("sign"),
          expected
        );
      }

      #[test]
      fn generated_keys_round_trip() {
        let (public, secret) = $profile::generate_keypair(fixed_fill(0x07)).expect("generate");
        let (again, _) = $profile::generate_keypair(fixed_fill(0x07)).expect("generate");
        assert_eq!(public, again, "generation is a function of the randomness");
        assert_eq!($public::from_bytes(&public.to_bytes()).expect("own encoding"), public);
        assert_eq!($public::from_spki_der(&public.to_spki_der()).expect("own SPKI"), public);
        let raw = secret.expose_secret();
        assert_eq!(
          $secret::from_bytes(raw.as_bytes()).expect("own encoding").public_key(),
          public
        );
        let mut der = [0; $secret::PKCS8_DER_LENGTH];
        secret.to_pkcs8_der_into(&mut der);
        assert_eq!($secret::from_pkcs8_der(&der).expect("own PKCS #8").public_key(), public);
        let signature = secret.sign_deterministic(b"message", b"").expect("sign");
        public.verify(b"message", &signature).expect("verify");
        assert!(format!("{secret:?}").ends_with("(****)"));

        let mut calls = 0;
        let result = $profile::generate_keypair(|out| {
          calls += 1;
          out.fill(0x55);
          if calls == 2 {
            Err(CompositeMlDsaError::RandomGenerationFailed)
          } else {
            Ok(())
          }
        });
        assert_eq!(
          result.expect_err("rejected"),
          CompositeMlDsaError::RandomGenerationFailed
        );
      }

      #[test]
      fn hostile_spki_is_rejected() {
        let (vectors, _, _) = case();
        let key = &vectors.$case.pk;
        let oid = composite_oid($arc);
        $public::from_spki_der(&spki(&oid, &[], 0, key)).expect("canonical SPKI");
        let cases: [(Vec<u8>, CompositeMlDsaKeyError); 7] = [
          (
            spki(&composite_oid(39), &[], 0, key),
            CompositeMlDsaKeyError::UnsupportedAlgorithm,
          ),
          (
            spki(&[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x03, 0x12], &[], 0, key),
            CompositeMlDsaKeyError::UnsupportedAlgorithm,
          ),
          (spki(&oid, &[0x05, 0x00], 0, key), CompositeMlDsaKeyError::MalformedDer),
          (spki(&oid, &[], 1, key), CompositeMlDsaKeyError::MalformedDer),
          (spki(&oid, &[], 0, &key[1..]), CompositeMlDsaKeyError::InvalidPublicKey),
          (
            spki(&oid, &[], 0, &[key.as_slice(), &[0]].concat()),
            CompositeMlDsaKeyError::InvalidPublicKey,
          ),
          (
            [spki(&oid, &[], 0, key), vec![0]].concat(),
            CompositeMlDsaKeyError::MalformedDer,
          ),
        ];
        for (index, (der, expected)) in cases.iter().enumerate() {
          assert_eq!(
            $public::from_spki_der(der).expect_err("rejected"),
            *expected,
            "case {index}"
          );
        }
        assert_eq!(
          $public::try_from_slice(&key[1..]).expect_err("rejected"),
          CompositeMlDsaKeyError::InvalidPublicKey
        );
      }

      /// Hostile PKCS #8 encodings of the vector key, with the error each must return.
      fn hostile_pkcs8(vectors: &Vectors, public: &$public) -> [(Vec<u8>, CompositeMlDsaKeyError); 7] {
        let sk = &vectors.$case.sk;
        let oid = composite_oid($arc);
        let public_field = tlv(0x81, &[&[0][..], &public.to_bytes()].concat());
        let (other, _) = $profile::generate_keypair(fixed_fill(0x11)).expect("generate");
        let wrong_public = tlv(0x81, &[&[0][..], &other.to_bytes()].concat());
        [
          (
            pkcs8(0, &composite_oid(39), sk, &[]),
            CompositeMlDsaKeyError::UnsupportedAlgorithm,
          ),
          (
            pkcs8(1, &oid, sk, &[&wrong_public]),
            CompositeMlDsaKeyError::InvalidSecretKey,
          ),
          (pkcs8(1, &oid, sk, &[]), CompositeMlDsaKeyError::MalformedDer),
          (
            pkcs8(0, &oid, sk, &[&public_field]),
            CompositeMlDsaKeyError::MalformedDer,
          ),
          (
            pkcs8(0, &oid, sk, &[&tlv(0xa0, &[])]),
            CompositeMlDsaKeyError::UnsupportedEncoding,
          ),
          (pkcs8(0, &oid, &sk[1..], &[]), CompositeMlDsaKeyError::InvalidSecretKey),
          (
            [pkcs8(0, &oid, sk, &[]), vec![0]].concat(),
            CompositeMlDsaKeyError::MalformedDer,
          ),
        ]
      }

      #[test]
      fn hostile_pkcs8_is_rejected() {
        let (vectors, public, _) = case();
        let sk = &vectors.$case.sk;
        let oid = composite_oid($arc);
        assert_eq!(pkcs8(0, &oid, sk, &[]), vectors.$case.sk_pkcs8);
        let public_field = tlv(0x81, &[&[0][..], &public.to_bytes()].concat());
        $secret::from_pkcs8_der(&pkcs8(1, &oid, sk, &[&public_field])).expect("version 2 with its own public key");
        for (index, (der, expected)) in hostile_pkcs8(&vectors, &public).iter().enumerate() {
          assert_eq!(
            $secret::from_pkcs8_der(der).expect_err("rejected"),
            *expected,
            "case {index}"
          );
        }
        assert_eq!(
          $secret::try_from_slice(&sk[1..]).expect_err("rejected"),
          CompositeMlDsaKeyError::InvalidSecretKey
        );
      }

      #[test]
      fn allocated_constructors_match_by_value_keys() {
        // A boxed key must hold the same key, and sign byte for byte like the
        // by-value key, for the draft's key in both of its encodings.
        let (vectors, public, secret) = case();
        let case = &vectors.$case;
        let expected = secret
          .sign_deterministic(&vectors.message, &vectors.context)
          .expect("sign");
        let hedged = secret
          .sign_with(&vectors.message, &vectors.context, fixed_fill(9))
          .expect("sign");
        let raw = $secret::try_from_slice_in(&case.sk, Global).expect("allocated raw import");
        let der = $secret::from_pkcs8_der_in(&case.sk_pkcs8, Global).expect("allocated PKCS #8 import");
        for boxed in [raw, der] {
          assert_eq!(boxed.expose_secret().as_bytes(), secret.expose_secret().as_bytes());
          assert_eq!(boxed.public_key(), public);
          assert_eq!(
            boxed
              .sign_deterministic(&vectors.message, &vectors.context)
              .expect("sign"),
            expected
          );
          assert_eq!(
            boxed
              .sign_with(&vectors.message, &vectors.context, fixed_fill(9))
              .expect("sign"),
            hedged
          );
        }
        let (generated_public, generated) = $profile::generate_keypair(fixed_fill(0x07)).expect("generate");
        let (boxed_public, boxed) =
          $profile::generate_keypair_in(fixed_fill(0x07), Global).expect("allocated generate");
        assert_eq!(boxed_public, generated_public);
        assert_eq!(boxed.expose_secret().as_bytes(), generated.expose_secret().as_bytes());
      }

      #[test]
      fn rejected_allocated_imports_match_by_value_errors_and_free_their_box() {
        // A failed `*_in` call must return the by-value error and leave no
        // allocation behind, including after it has filled its box.
        let (vectors, public, _) = case();
        let live = Live::default();
        for (index, (der, expected)) in hostile_pkcs8(&vectors, &public).iter().enumerate() {
          assert_eq!(
            $secret::from_pkcs8_der_in(der, &live).expect_err("rejected"),
            *expected,
            "case {index}"
          );
          assert_eq!(live.outstanding(), 0, "case {index}");
        }
        assert!(
          live.allocations.get() > 0,
          "a public-key mismatch fails after allocating"
        );
        assert_eq!(
          $secret::try_from_slice_in(&vectors.$case.sk[1..], &live).expect_err("rejected"),
          CompositeMlDsaKeyError::InvalidSecretKey
        );
        let failed = $profile::generate_keypair_in(|_| Err(CompositeMlDsaError::RandomGenerationFailed), &live);
        assert_eq!(failed.err(), Some(CompositeMlDsaError::RandomGenerationFailed));
        assert_eq!(live.outstanding(), 0);
      }

      #[test]
      fn hostile_signature_lengths_are_rejected() {
        let (vectors, _, _) = case();
        let s = &vectors.$case.s;
        for bad in [
          &s[..MlDsa65Signature::LENGTH],
          &s[..s.len() - 1],
          &[s.as_slice(), &[0]].concat()[..],
          &[],
        ] {
          assert_eq!(
            $signature::try_from_slice(bad).expect_err("rejected"),
            CompositeMlDsaError::InvalidSignature,
            "length {}",
            bad.len()
          );
        }
        // A non-canonical ML-DSA hint section is structurally invalid.
        let mut hints = s.clone();
        hints[MlDsa65Signature::LENGTH - 1] = 0xff;
        assert_eq!(
          $signature::try_from_slice(&hints).expect_err("rejected"),
          CompositeMlDsaError::InvalidSignature
        );
      }
    }
  };
}

composite_tests!(
  ed25519,
  MlDsa65Ed25519,
  MlDsa65Ed25519PublicKey,
  MlDsa65Ed25519SecretKey,
  MlDsa65Ed25519Signature,
  ed25519,
  48
);
composite_tests!(
  ecdsa_p256,
  MlDsa65EcdsaP256,
  MlDsa65EcdsaP256PublicKey,
  MlDsa65EcdsaP256SecretKey,
  MlDsa65EcdsaP256Signature,
  ecdsa_p256,
  45
);

#[test]
fn ed25519_signature_is_fixed_length() {
  assert_eq!(MlDsa65Ed25519PublicKey::LENGTH, 1984);
  assert_eq!(MlDsa65Ed25519SecretKey::LENGTH, 64);
  assert_eq!(MlDsa65Ed25519Signature::LENGTH, 3373);
  assert_eq!(MlDsa65EcdsaP256PublicKey::LENGTH, 2017);
  assert_eq!(MlDsa65EcdsaP256SecretKey::LENGTH, 83);
  assert_eq!(MlDsa65EcdsaP256Signature::MAX_LENGTH, 3381);
}

#[test]
fn ecdsa_public_point_must_be_on_the_curve() {
  let vectors = vectors();
  let pk = &vectors.ecdsa_p256.pk;
  let point = MlDsa65PublicKey::LENGTH;
  for (index, value) in [(point, 0x02), (point, 0x00), (pk.len() - 1, pk[pk.len() - 1] ^ 1)] {
    let mut altered = pk.clone();
    altered[index] = value;
    assert_eq!(
      MlDsa65EcdsaP256PublicKey::try_from_slice(&altered).expect_err("rejected"),
      CompositeMlDsaKeyError::InvalidPublicKey,
      "byte {index}"
    );
  }
}

#[test]
fn ecdsa_private_key_must_use_the_draft_encoding() {
  let vectors = vectors();
  let sk = &vectors.ecdsa_p256.sk;
  // seed (32) || 30 31 02 01 01 04 20 <scalar> a0 0a 06 08 <prime256v1>.
  let with = |index: usize, value: u8| {
    let mut altered = sk.clone();
    altered[index] = value;
    MlDsa65EcdsaP256SecretKey::try_from_slice(&altered)
      .map(|_| ())
      .expect_err("rejected")
  };
  assert_eq!(
    with(36, 0x02),
    CompositeMlDsaKeyError::MalformedDer,
    "ECPrivateKey version 2"
  );
  assert_eq!(
    with(82, 0x08),
    CompositeMlDsaKeyError::UnsupportedAlgorithm,
    "another curve"
  );
  let scalar = 32 + 7;
  let mut zero = sk.clone();
  zero[scalar..scalar + 32].fill(0);
  let mut order = sk.clone();
  order[scalar..scalar + 32].fill(0xff);
  let live = Live::default();
  for altered in [zero, order] {
    assert_eq!(
      MlDsa65EcdsaP256SecretKey::try_from_slice(&altered).expect_err("rejected"),
      CompositeMlDsaKeyError::InvalidSecretKey
    );
    // The scalar is checked inside the box, which must then be freed.
    assert_eq!(
      MlDsa65EcdsaP256SecretKey::try_from_slice_in(&altered, &live).expect_err("rejected"),
      CompositeMlDsaKeyError::InvalidSecretKey
    );
    assert_eq!(live.outstanding(), 0);
  }
  assert_eq!(
    live.allocations.get(),
    2,
    "each rejected scalar was checked in an allocation"
  );
  // The general RFC 5915 form with a public key is a valid key in another
  // encoding, and the composite length rejects it first.
  let secret = MlDsa65EcdsaP256SecretKey::try_from_slice(sk).expect("vector key");
  let public = secret.public_key().to_bytes();
  let ec_private_key = tlv(
    0x30,
    &[
      tlv(0x02, &[1]),
      tlv(0x04, &sk[scalar..scalar + 32]),
      tlv(
        0xa1,
        &tlv(0x03, &[&[0][..], &public[MlDsa65PublicKey::LENGTH..]].concat()),
      ),
    ]
    .concat(),
  );
  assert_eq!(
    MlDsa65EcdsaP256SecretKey::try_from_slice(&[&sk[..32], &ec_private_key[..]].concat()).expect_err("rejected"),
    CompositeMlDsaKeyError::InvalidSecretKey
  );
}

#[test]
fn ecdsa_signature_must_be_canonical_der() {
  let vectors = vectors();
  let s = &vectors.ecdsa_p256.s;
  let (ml_dsa, der) = s.split_at(MlDsa65Signature::LENGTH);
  let (_, sequence, _) = element(der);
  let (_, r, rest) = element(sequence);
  let (_, s_scalar, _) = element(rest);
  let rebuild = |r: &[u8], s: &[u8], long_form: bool| {
    let contents = [tlv(0x02, r), tlv(0x02, s)].concat();
    let sequence = if long_form {
      [vec![0x30, 0x81, u8::try_from(contents.len()).expect("short")], contents].concat()
    } else {
      tlv(0x30, &contents)
    };
    [ml_dsa, &sequence[..]].concat()
  };
  MlDsa65EcdsaP256Signature::try_from_slice(&rebuild(r, s_scalar, false)).expect("canonical DER");
  let padded = [&[0][..], r].concat();
  for bad in [
    rebuild(&padded, s_scalar, false),
    rebuild(r, s_scalar, true),
    [&rebuild(r, s_scalar, false)[..], &[0]].concat(),
    rebuild(&[0], s_scalar, false),
  ] {
    assert_eq!(
      MlDsa65EcdsaP256Signature::try_from_slice(&bad).expect_err("rejected"),
      CompositeMlDsaError::InvalidSignature
    );
  }
}

#[test]
fn composite_keys_are_not_interchangeable() {
  let vectors = vectors();
  assert_eq!(
    MlDsa65Ed25519PublicKey::from_spki_der(
      &MlDsa65EcdsaP256PublicKey::try_from_slice(&vectors.ecdsa_p256.pk)
        .expect("key")
        .to_spki_der()
    )
    .expect_err("rejected"),
    CompositeMlDsaKeyError::UnsupportedAlgorithm
  );
  assert_eq!(
    MlDsa65EcdsaP256SecretKey::from_pkcs8_der(&vectors.ed25519.sk_pkcs8).expect_err("rejected"),
    CompositeMlDsaKeyError::UnsupportedAlgorithm
  );
  // An ML-DSA-65 key is not a composite key.
  let (ml_dsa, _) = MlDsa65::keypair_from_seed(&[3; 32]).expect("ML-DSA key");
  assert_eq!(
    MlDsa65Ed25519PublicKey::from_spki_der(&ml_dsa.to_spki_der()).expect_err("rejected"),
    CompositeMlDsaKeyError::UnsupportedAlgorithm
  );
}
