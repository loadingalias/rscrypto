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
    ($profile:ident, $public:ident, $secret:ident, $signature:ident) => {{
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
      let prepared = secret.prepare().expect("prepare secret");
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
      // Arbitrary byte lengths and contents must not panic at a public parser.
      let _public = $public::try_from_slice(message);
      let _secret = $secret::try_from_slice(message);
      let _signature = $signature::try_from_slice(message);
    }};
  }
  match seed[0] % 3 {
    0 => exercise!(MlDsa44, MlDsa44PublicKey, MlDsa44SecretKey, MlDsa44Signature),
    1 => exercise!(MlDsa65, MlDsa65PublicKey, MlDsa65SecretKey, MlDsa65Signature),
    _ => exercise!(MlDsa87, MlDsa87PublicKey, MlDsa87SecretKey, MlDsa87Signature),
  }
}
