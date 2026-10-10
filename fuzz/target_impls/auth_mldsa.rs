//! The named parser seeds prepend 32 key-seed bytes, 32 randomness bytes, and
//! nine mutation-control bytes to the pinned ACVP runtime key/signature bytes.
//! They enter exact-length parsers without waiting for fuzz-input length growth.

use rscrypto::*;
use rscrypto_fuzz::{FuzzInput, some_or_return};

pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let seed: [u8; 32] = some_or_return!(input.bytes());
  let random: [u8; 32] = some_or_return!(input.bytes());
  let mutation = some_or_return!(input.bit_mutation());
  let message = input.rest();
  macro_rules! exercise {
    ($profile:ident, $public:ident, $secret:ident, $seed:ident, $signature:ident) => {{
      let (public, secret) = $profile::keypair_from_seed(&seed).expect("fuzz key generation");
      let signature = secret
        .sign_with(message, b"fuzz", |out| {
          out.copy_from_slice(&random);
          Ok(())
        })
        .expect("fuzz signing");
      public
        .verify_with_context(message, b"fuzz", &signature)
        .expect("valid signature");
      let mut storage = Default::default();
      let prepared = secret.prepare(&mut storage).expect("prepare secret");
      assert_eq!(
        signature,
        prepared
          .sign_with(message, b"fuzz", |out| {
            out.copy_from_slice(&random);
            Ok(())
          })
          .expect("prepared signature")
      );
      let mut changed = signature.to_bytes();
      mutation.apply(&mut changed);
      if let Ok(changed) = $signature::try_from_slice(&changed) {
        public
          .verify_with_context(message, b"fuzz", &changed)
          .expect_err("single-bit signature forgery");
      }
      // An accepted SPKI is the unique encoding of its key: a one-bit change
      // either yields another key's encoding or is rejected.
      let mut encoded = public.to_spki_der();
      assert_eq!($public::from_spki_der(&encoded), Ok(public.clone()));
      mutation.apply(&mut encoded);
      for candidate in [encoded.as_slice(), message] {
        if let Ok(parsed) = $public::from_spki_der(candidate) {
          assert_eq!(
            parsed.to_spki_der().as_slice(),
            candidate,
            "accepted SPKI must be canonical"
          );
        }
      }
      // Each fixed-length PKCS #8 form has one encoding per key, so an
      // accepted input of that length must re-encode to itself.
      let seed_owner = $seed::from_bytes(seed);
      let mut seed_der = [0; $seed::PKCS8_DER_LENGTH];
      seed_owner.to_pkcs8_der_into(&mut seed_der);
      let mut expanded_der = [0; $secret::PKCS8_DER_LENGTH];
      secret.to_pkcs8_der_into(&mut expanded_der);
      let imported = $secret::from_pkcs8_der(&seed_der).expect("seed-form import");
      assert_eq!(imported.public_key(), &public);
      let imported = $secret::from_pkcs8_der(&expanded_der).expect("expanded-form import");
      assert_eq!(imported.public_key(), &public);
      mutation.apply(&mut seed_der);
      mutation.apply(&mut expanded_der);
      for candidate in [seed_der.as_slice(), expanded_der.as_slice(), message] {
        let parsed_seed = $seed::from_pkcs8_der(candidate);
        if let Ok(parsed_seed) = &parsed_seed
          && candidate.len() == $seed::PKCS8_DER_LENGTH
        {
          let mut encoded = [0; $seed::PKCS8_DER_LENGTH];
          parsed_seed.to_pkcs8_der_into(&mut encoded);
          assert_eq!(encoded.as_slice(), candidate, "accepted seed form must be canonical");
        }
        if let Ok(parsed) = $secret::from_pkcs8_der(candidate) {
          if candidate.len() == $secret::PKCS8_DER_LENGTH {
            let mut encoded = [0; $secret::PKCS8_DER_LENGTH];
            parsed.to_pkcs8_der_into(&mut encoded);
            assert_eq!(encoded.as_slice(), candidate, "accepted expanded form must be canonical");
          }
          if let Ok(parsed_seed) = parsed_seed {
            let (seed_public, _) = parsed_seed.keypair().expect("accepted seed expansion");
            assert_eq!(&seed_public, parsed.public_key(), "both imports name one key");
          }
        }
      }
      // Arbitrary byte lengths and contents must not panic at a public parser.
      let _public = $public::try_from_slice(message);
      let _secret = $secret::try_from_slice(message);
      let _signature = $signature::try_from_slice(message);
    }};
  }
  match seed[0] % 3 {
    0 => exercise!(MlDsa44, MlDsa44PublicKey, MlDsa44SecretKey, MlDsa44Seed, MlDsa44Signature),
    1 => exercise!(MlDsa65, MlDsa65PublicKey, MlDsa65SecretKey, MlDsa65Seed, MlDsa65Signature),
    _ => exercise!(MlDsa87, MlDsa87PublicKey, MlDsa87SecretKey, MlDsa87Seed, MlDsa87Signature),
  }
}
