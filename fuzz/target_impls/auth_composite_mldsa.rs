//! Composite ML-DSA (draft-ietf-lamps-pq-composite-sigs-19) signing, import,
//! and parsing invariants for both composites.
//!
//! Input: one selector byte, 32 key-randomness bytes, 96 signing-randomness
//! bytes, nine mutation-control bytes, then the message. The named parser
//! seeds put the draft's vector keys and signatures in the message, so they
//! reach the exact-length parsers directly.

use rscrypto::*;
use rscrypto_fuzz::{FuzzInput, some_or_return};

pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let selector = some_or_return!(input.byte());
  let key_random: [u8; 32] = some_or_return!(input.bytes());
  let sign_random: [u8; 96] = some_or_return!(input.bytes());
  let mutation = some_or_return!(input.bit_mutation());
  let message = input.rest();
  let context: &[u8] = &message[..message.len().min(usize::from(selector >> 1))];
  macro_rules! exercise {
    ($profile:ident, $public:ident, $secret:ident, $signature:ident) => {{
      // Key generation draws the ML-DSA seed, then the traditional key.
      let mut offset = 0usize;
      let fill = |out: &mut [u8]| {
        for byte in out.iter_mut() {
          let round = u8::try_from(offset.strict_div(32).strict_rem(256)).expect("byte");
          *byte = key_random[offset.strict_rem(32)] ^ round;
          offset = offset.strict_add(1);
        }
        Ok(())
      };
      let (public, secret) = $profile::generate_keypair(fill).expect("fuzz key generation");
      let hedged = |out: &mut [u8]| {
        out.copy_from_slice(&sign_random[..out.len()]);
        Ok(())
      };
      let signature = secret.sign_with(message, context, hedged).expect("fuzz signing");
      public
        .verify_with_context(message, context, &signature)
        .expect("valid signature");
      let deterministic = secret.sign_deterministic(message, context).expect("fuzz signing");
      public
        .verify_with_context(message, context, &deterministic)
        .expect("valid deterministic signature");
      let mut storage = Default::default();
      let prepared = secret.prepare(&mut storage).expect("prepare secret");
      assert_eq!(signature, prepared.sign_with(message, context, hedged).expect("prepared"));
      drop(prepared);

      let mut changed = signature.as_ref().to_vec();
      mutation.apply(&mut changed);
      if let Ok(changed) = $signature::try_from_slice(&changed) {
        public
          .verify_with_context(message, context, &changed)
          .expect_err("single-bit signature forgery");
      }

      // Accepted SPKI and PKCS #8 encodings are the unique encodings of
      // their keys.
      let mut spki = public.to_spki_der();
      assert_eq!($public::from_spki_der(&spki), Ok(public.clone()));
      mutation.apply(&mut spki);
      for candidate in [spki.as_slice(), message] {
        if let Ok(parsed) = $public::from_spki_der(candidate) {
          assert_eq!(parsed.to_spki_der().as_slice(), candidate, "accepted SPKI must be canonical");
        }
      }
      let mut pkcs8 = [0; $secret::PKCS8_DER_LENGTH];
      secret.to_pkcs8_der_into(&mut pkcs8);
      let imported = $secret::from_pkcs8_der(&pkcs8).expect("own PKCS #8");
      assert_eq!(imported.public_key(), public);
      mutation.apply(&mut pkcs8);
      for candidate in [pkcs8.as_slice(), message] {
        if let Ok(parsed) = $secret::from_pkcs8_der(candidate)
          && candidate.len() == $secret::PKCS8_DER_LENGTH
        {
          let mut encoded = [0; $secret::PKCS8_DER_LENGTH];
          parsed.to_pkcs8_der_into(&mut encoded);
          assert_eq!(encoded.as_slice(), candidate, "accepted PKCS #8 must be canonical");
        }
      }

      // Raw parsers: arbitrary input must not panic, and an accepted input
      // re-encodes to itself.
      if let Ok(parsed) = $public::try_from_slice(message) {
        assert_eq!(parsed.to_bytes().as_slice(), message);
      }
      if let Ok(parsed) = $secret::try_from_slice(message) {
        assert_eq!(parsed.expose_secret().as_bytes().as_slice(), message);
      }
      if let Ok(parsed) = $signature::try_from_slice(message) {
        assert_eq!(parsed.as_ref(), message);
        let _rejected = public.verify(message, &parsed);
      }
    }};
  }
  if selector & 1 == 0 {
    exercise!(MlDsa65Ed25519, MlDsa65Ed25519PublicKey, MlDsa65Ed25519SecretKey, MlDsa65Ed25519Signature);
  } else {
    exercise!(MlDsa65EcdsaP256, MlDsa65EcdsaP256PublicKey, MlDsa65EcdsaP256SecretKey, MlDsa65EcdsaP256Signature);
  }
}
