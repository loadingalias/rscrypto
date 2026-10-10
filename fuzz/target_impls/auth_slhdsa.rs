//! SLH-DSA hostile verification, forgery, and key-encoding canonicality for
//! all 24 pure and HashSLH-DSA profiles.
//!
//! Every profile verifies a fuzz-chosen signature and parses fuzz-chosen
//! SPKI, PKCS #8, and raw keys. For the fast sets, a key pair and one anchor
//! signature are made once per process; each input flips one bit of the
//! anchor and of the key encodings, and a fuzz-chosen share of inputs signs
//! the fuzz message. The small sets use a fixed public key instead: their key
//! generation and signing cost seconds, and the ACVP sigVer vectors already
//! cover modified full-length signatures for every set.
//!
//! Input: profile selector, nine mutation-control bytes, 32 randomness bytes,
//! a split ratio, then the candidate encoding and the message.

use std::sync::OnceLock;

use rscrypto::*;
use rscrypto_fuzz::{FuzzInput, some_or_return};

const ANCHOR_MESSAGE: &[u8] = b"rscrypto SLH-DSA fuzz anchor";

pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let selector = some_or_return!(input.byte());
  let mutation = some_or_return!(input.bit_mutation());
  let random: [u8; 32] = some_or_return!(input.bytes());
  let (candidate, message) = some_or_return!(input.split_rest());
  // One fuzz-chosen input in 256 runs the checks that cost a signature or a
  // key generation; ACVP and the differential tests carry signing itself.
  let expensive = random[0] == 0;

  // Checks every profile shares: an arbitrary signature never verifies, and
  // every accepted encoding is canonical.
  macro_rules! common {
    ($public_type:ident, $secret_type:ident, $public:expr) => {{
      let public: &$public_type = $public;
      public
        .verify(message, candidate)
        .expect_err("fuzz-chosen signature forgery");
      let mut spki = public.to_spki_der();
      assert_eq!($public_type::from_spki_der(&spki).ok().as_ref(), Some(public));
      mutation.apply(&mut spki);
      for der in [spki.as_slice(), candidate] {
        if let Ok(parsed) = $public_type::from_spki_der(der) {
          assert_eq!(parsed.to_spki_der().as_slice(), der, "accepted SPKI must be canonical");
        }
      }
      if let Ok(parsed) = $secret_type::from_pkcs8_der(candidate)
        && candidate.len() == $secret_type::PKCS8_DER_LENGTH
      {
        let mut encoded = [0; $secret_type::PKCS8_DER_LENGTH];
        parsed.to_pkcs8_der_into(&mut encoded);
        assert_eq!(encoded.as_slice(), candidate, "accepted PKCS #8 must be canonical");
      }
      // Raw parsers accept any length or content without panicking; an
      // accepted secret key re-encodes to its input.
      let _public = $public_type::try_from_slice(candidate);
      if let Ok(parsed) = $secret_type::try_from_slice(candidate) {
        assert_eq!(parsed.expose_secret().as_bytes().as_slice(), candidate);
      }
    }};
  }

  macro_rules! small {
    ($public_type:ident, $secret_type:ident, $n:literal) => {{
      let public = $public_type::from_bytes([$n; 2 * $n]);
      common!($public_type, $secret_type, &public);
    }};
  }

  macro_rules! fast {
    ($profile:ident, $public_type:ident, $secret_type:ident, $n:literal) => {{
      static KEYS: OnceLock<($public_type, $secret_type, Vec<u8>)> = OnceLock::new();
      let (public, secret, anchor) = KEYS.get_or_init(|| {
        let (public, secret) = $profile::generate_keypair(|seeds| {
          seeds.fill($n);
          Ok(())
        })
        .expect("fuzz key generation");
        let mut signature = vec![0; $profile::SIGNATURE_LENGTH];
        let out = <&mut [u8; $profile::SIGNATURE_LENGTH]>::try_from(signature.as_mut_slice()).expect("buffer");
        secret
          .sign_deterministic(ANCHOR_MESSAGE, b"", out)
          .expect("anchor signature");
        public.verify(ANCHOR_MESSAGE, &signature).expect("anchor verifies");
        let mut pkcs8 = [0; $secret_type::PKCS8_DER_LENGTH];
        secret.to_pkcs8_der_into(&mut pkcs8);
        assert_eq!(
          $secret_type::from_pkcs8_der(&pkcs8).expect("own PKCS #8").public_key(),
          &public
        );
        (public, secret, signature)
      });
      common!($public_type, $secret_type, public);

      // No one-bit change to the anchor verifies.
      let mut changed = anchor.clone();
      mutation.apply(&mut changed);
      public
        .verify(ANCHOR_MESSAGE, &changed)
        .expect_err("single-bit signature forgery");

      // The version 1 PKCS #8 form is unique per key: a changed SK.prf still
      // imports, and a changed seed or root fails the root check, which costs
      // a key generation. Other inputs change only the DER header.
      let mut pkcs8 = [0; $secret_type::PKCS8_DER_LENGTH];
      secret.to_pkcs8_der_into(&mut pkcs8);
      let header = $secret_type::PKCS8_DER_LENGTH.strict_sub($secret_type::LENGTH);
      mutation.apply(if expensive {
        &mut pkcs8[..]
      } else {
        &mut pkcs8[..header]
      });
      if let Ok(parsed) = $secret_type::from_pkcs8_der(&pkcs8) {
        let mut encoded = [0; $secret_type::PKCS8_DER_LENGTH];
        parsed.to_pkcs8_der_into(&mut encoded);
        assert_eq!(encoded, pkcs8, "accepted PKCS #8 must be canonical");
      }

      if expensive {
        let mut signature = vec![0; $profile::SIGNATURE_LENGTH];
        let out = <&mut [u8; $profile::SIGNATURE_LENGTH]>::try_from(signature.as_mut_slice()).expect("buffer");
        secret
          .sign_with(
            message,
            b"fuzz",
            |out| {
              out.copy_from_slice(&random[..$n]);
              Ok(())
            },
            out,
          )
          .expect("fuzz signing");
        public
          .verify_with_context(message, b"fuzz", &signature)
          .expect("fresh signature");
        public.verify(message, &signature).expect_err("context is bound");
        mutation.apply(&mut signature);
        public
          .verify_with_context(message, b"fuzz", &signature)
          .expect_err("single-bit signature forgery");
      }
    }};
  }

  match selector % 24 {
    0 => small!(SlhDsaSha2_128sPublicKey, SlhDsaSha2_128sSecretKey, 16),
    1 => fast!(SlhDsaSha2_128f, SlhDsaSha2_128fPublicKey, SlhDsaSha2_128fSecretKey, 16),
    2 => small!(SlhDsaSha2_192sPublicKey, SlhDsaSha2_192sSecretKey, 24),
    3 => fast!(SlhDsaSha2_192f, SlhDsaSha2_192fPublicKey, SlhDsaSha2_192fSecretKey, 24),
    4 => small!(SlhDsaSha2_256sPublicKey, SlhDsaSha2_256sSecretKey, 32),
    5 => fast!(SlhDsaSha2_256f, SlhDsaSha2_256fPublicKey, SlhDsaSha2_256fSecretKey, 32),
    6 => small!(SlhDsaShake128sPublicKey, SlhDsaShake128sSecretKey, 16),
    7 => fast!(SlhDsaShake128f, SlhDsaShake128fPublicKey, SlhDsaShake128fSecretKey, 16),
    8 => small!(SlhDsaShake192sPublicKey, SlhDsaShake192sSecretKey, 24),
    9 => fast!(SlhDsaShake192f, SlhDsaShake192fPublicKey, SlhDsaShake192fSecretKey, 24),
    10 => small!(SlhDsaShake256sPublicKey, SlhDsaShake256sSecretKey, 32),
    11 => fast!(SlhDsaShake256f, SlhDsaShake256fPublicKey, SlhDsaShake256fSecretKey, 32),
    12 => small!(
      HashSlhDsaSha2_128sWithSha256PublicKey,
      HashSlhDsaSha2_128sWithSha256SecretKey,
      16
    ),
    13 => fast!(
      HashSlhDsaSha2_128fWithSha256,
      HashSlhDsaSha2_128fWithSha256PublicKey,
      HashSlhDsaSha2_128fWithSha256SecretKey,
      16
    ),
    14 => small!(
      HashSlhDsaSha2_192sWithSha512PublicKey,
      HashSlhDsaSha2_192sWithSha512SecretKey,
      24
    ),
    15 => fast!(
      HashSlhDsaSha2_192fWithSha512,
      HashSlhDsaSha2_192fWithSha512PublicKey,
      HashSlhDsaSha2_192fWithSha512SecretKey,
      24
    ),
    16 => small!(
      HashSlhDsaSha2_256sWithSha512PublicKey,
      HashSlhDsaSha2_256sWithSha512SecretKey,
      32
    ),
    17 => fast!(
      HashSlhDsaSha2_256fWithSha512,
      HashSlhDsaSha2_256fWithSha512PublicKey,
      HashSlhDsaSha2_256fWithSha512SecretKey,
      32
    ),
    18 => small!(
      HashSlhDsaShake128sWithShake128PublicKey,
      HashSlhDsaShake128sWithShake128SecretKey,
      16
    ),
    19 => fast!(
      HashSlhDsaShake128fWithShake128,
      HashSlhDsaShake128fWithShake128PublicKey,
      HashSlhDsaShake128fWithShake128SecretKey,
      16
    ),
    20 => small!(
      HashSlhDsaShake192sWithShake256PublicKey,
      HashSlhDsaShake192sWithShake256SecretKey,
      24
    ),
    21 => fast!(
      HashSlhDsaShake192fWithShake256,
      HashSlhDsaShake192fWithShake256PublicKey,
      HashSlhDsaShake192fWithShake256SecretKey,
      24
    ),
    22 => small!(
      HashSlhDsaShake256sWithShake256PublicKey,
      HashSlhDsaShake256sWithShake256SecretKey,
      32
    ),
    _ => fast!(
      HashSlhDsaShake256fWithShake256,
      HashSlhDsaShake256fWithShake256PublicKey,
      HashSlhDsaShake256fWithShake256SecretKey,
      32
    ),
  }
}
