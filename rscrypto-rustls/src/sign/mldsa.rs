//! Pure, hedged ML-DSA with the empty context required by TLS.

use alloc::{alloc::Global, boxed::Box, sync::Arc, vec::Vec};
use rscrypto::{
  MlDsa44PreparedSecretKey, MlDsa44PreparedSecretKeyStorage, MlDsa44SecretKey, MlDsa65PreparedSecretKey,
  MlDsa65PreparedSecretKeyStorage, MlDsa65SecretKey, MlDsa87PreparedSecretKey, MlDsa87PreparedSecretKeyStorage,
  MlDsa87SecretKey, MlDsaError,
};
use rustls::Error;
use rustls::crypto::SignatureScheme;
use rustls::crypto::{Signer, SigningKey};
use rustls::pki_types::{PrivateKeyDer, SubjectPublicKeyInfoDer};

/// Load a key, prepared for repeated signing unless `compact`.
pub(super) fn load(
  der: &PrivateKeyDer<'_>,
  random: &'static dyn rustls::crypto::SecureRandom,
  compact: bool,
) -> Option<Key> {
  let PrivateKeyDer::Pkcs8(der) = der else {
    return None;
  };
  let der = der.secret_pkcs8_der();
  macro_rules! load {
    ($key:ident, $variant:ident) => {
      if let Ok(key) = $key::from_pkcs8_der_in(der, Global) {
        let spki = key.public_key().to_spki_der().to_vec();
        // A validated key always prepares: its matrix seed already expanded
        // during import. The compact key is cleared when it drops.
        let form = if compact {
          Form::Compact(key)
        } else {
          Form::Prepared(key.prepare_in(Global).ok()?)
        };
        return Some(Key {
          secret: Arc::new(Secret::$variant(form)),
          spki,
          random,
        });
      }
    };
  }
  load!(MlDsa44SecretKey, Dsa44);
  load!(MlDsa65SecretKey, Dsa65);
  load!(MlDsa87SecretKey, Dsa87);
  None
}

/// A loaded key: the compact key, or its prepared form, which owns its
/// storage and keeps no compact copy.
#[derive(Debug)]
enum Form<C, P> {
  Compact(Box<C>),
  Prepared(P),
}

#[derive(Debug)]
enum Secret {
  Dsa44(Form<MlDsa44SecretKey, MlDsa44PreparedSecretKey<Box<MlDsa44PreparedSecretKeyStorage>>>),
  Dsa65(Form<MlDsa65SecretKey, MlDsa65PreparedSecretKey<Box<MlDsa65PreparedSecretKeyStorage>>>),
  Dsa87(Form<MlDsa87SecretKey, MlDsa87PreparedSecretKey<Box<MlDsa87PreparedSecretKeyStorage>>>),
}

impl Secret {
  fn scheme(&self) -> SignatureScheme {
    match self {
      Self::Dsa44(_) => SignatureScheme::ML_DSA_44,
      Self::Dsa65(_) => SignatureScheme::ML_DSA_65,
      Self::Dsa87(_) => SignatureScheme::ML_DSA_87,
    }
  }

  fn sign_with(&self, message: &[u8], fill: impl FnMut(&mut [u8]) -> Result<(), MlDsaError>) -> Result<Vec<u8>, Error> {
    // Rustls supplies the TLS CertificateVerify context and transcript. The
    // distinct FIPS 204 context must stay empty, without HashML-DSA prehashing.
    // Compact signing expands the public matrix in every rejection attempt, so
    // a compact key prepares heap storage for each signature instead: the
    // matrix is expanded once and the signature bytes are the same. That
    // prepared key clears its secrets before its storage is freed.
    macro_rules! sign {
      ($form:expr) => {
        match $form {
          Form::Prepared(prepared) => prepared.sign_with(message, b"", fill),
          Form::Compact(key) => key
            .prepare_in(Global)
            .and_then(|prepared| prepared.sign_with(message, b"", fill)),
        }
        .map(|signature| signature.as_bytes().to_vec())
      };
    }
    match self {
      Self::Dsa44(form) => sign!(form),
      Self::Dsa65(form) => sign!(form),
      Self::Dsa87(form) => sign!(form),
    }
    .map_err(|_| super::signing_failed())
  }
}

#[derive(Debug)]
pub(super) struct Key {
  secret: Arc<Secret>,
  spki: Vec<u8>,
  random: &'static dyn rustls::crypto::SecureRandom,
}

impl SigningKey for Key {
  fn choose_scheme(&self, offered: &[SignatureScheme]) -> Option<Box<dyn Signer>> {
    offered.contains(&self.secret.scheme()).then(|| {
      Box::new(Sign {
        secret: Arc::clone(&self.secret),
        random: self.random,
      }) as Box<dyn Signer>
    })
  }

  fn public_key(&self) -> Option<SubjectPublicKeyInfoDer<'_>> {
    Some(SubjectPublicKeyInfoDer::from(self.spki.as_slice()))
  }
}

#[derive(Debug)]
struct Sign {
  secret: Arc<Secret>,
  random: &'static dyn rustls::crypto::SecureRandom,
}

impl Signer for Sign {
  fn sign(self: Box<Self>, message: &[u8]) -> Result<Vec<u8>, Error> {
    self.secret.sign_with(message, |out| {
      self.random.fill(out).map_err(|_| MlDsaError::RandomGenerationFailed)
    })
  }

  fn scheme(&self) -> SignatureScheme {
    self.secret.scheme()
  }
}

#[cfg(test)]
#[test]
fn partial_entropy_failure_is_opaque_and_key_stays_usable() {
  for der in [
    &include_bytes!("../../tests/vectors/credentials/mldsa44-leaf-key.der")[..],
    &include_bytes!("../../tests/vectors/credentials/mldsa65-leaf-key.der")[..],
    &include_bytes!("../../tests/vectors/credentials/mldsa87-leaf-key.der")[..],
  ] {
    for compact in [false, true] {
      let key = load(
        &PrivateKeyDer::try_from(der).expect("fixture container"),
        &crate::tests::RANDOM,
        compact,
      )
      .expect("fixture key");
      let result = key.secret.sign_with(b"message", |out| {
        out[..17].fill(0x42);
        Err(MlDsaError::RandomGenerationFailed)
      });
      assert!(matches!(result, Err(Error::General(message)) if message == "TLS signing failed"));
      let signer = key.choose_scheme(&[key.secret.scheme()]).expect("scheme");
      let scheme = signer.scheme();
      let signature = signer.sign(b"message").expect("sign after failure");
      let peer = rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER.signature_verification_algorithms;
      let verifier = peer
        .mapping()
        .iter()
        .find(|(candidate, _)| *candidate == scheme)
        .expect("peer scheme")
        .1[0];
      // RFC 9881 SPKI has a 22-byte header for all three parameter sets.
      verifier
        .verify_signature(&key.spki[22..], b"message", &signature)
        .expect("independent signature check");
    }
  }
}

#[cfg(test)]
#[test]
fn prepared_and_compact_keys_sign_the_compact_signature() {
  macro_rules! check {
    ($secret:ident, $variant:ident, $file:literal) => {{
      let der = include_bytes!(concat!("../../tests/vectors/credentials/", $file, "-leaf-key.der"));
      let fill = |out: &mut [u8]| {
        out.fill(0x5a);
        Ok(())
      };
      // Core compact signing with the same randomness is the authority.
      let expected = $secret::from_pkcs8_der(der)
        .expect("fixture key")
        .sign_with(b"message", b"", fill)
        .expect("compact signature");
      for compact in [false, true] {
        let key = load(
          &PrivateKeyDer::try_from(&der[..]).expect("fixture container"),
          &crate::tests::RANDOM,
          compact,
        )
        .expect("fixture key");
        assert_eq!(
          matches!(&*key.secret, Secret::$variant(Form::Compact(_))),
          compact,
          "the loader keeps the selected form"
        );
        for _ in 0..2 {
          let signature = key.secret.sign_with(b"message", fill).expect("signature");
          assert_eq!(signature, expected.as_bytes(), "{} compact={compact}", $file);
        }
      }
    }};
  }
  check!(MlDsa44SecretKey, Dsa44, "mldsa44");
  check!(MlDsa65SecretKey, Dsa65, "mldsa65");
  check!(MlDsa87SecretKey, Dsa87, "mldsa87");
}
