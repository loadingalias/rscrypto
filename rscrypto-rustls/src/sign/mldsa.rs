//! Pure, hedged ML-DSA with the empty context required by TLS.

use alloc::{alloc::Global, boxed::Box, sync::Arc, vec::Vec};
use rscrypto::{MlDsa44SecretKey, MlDsa65SecretKey, MlDsa87SecretKey, MlDsaError};
use rustls::pki_types::{PrivateKeyDer, SubjectPublicKeyInfoDer};
use rustls::sign::{Signer, SigningKey};
use rustls::{Error, SignatureAlgorithm, SignatureScheme};

pub(super) fn load(der: &PrivateKeyDer<'_>, random: &'static dyn rustls::crypto::SecureRandom) -> Option<Key> {
  let PrivateKeyDer::Pkcs8(der) = der else {
    return None;
  };
  let der = der.secret_pkcs8_der();
  macro_rules! load {
    ($key:ident, $variant:ident) => {
      if let Ok(key) = $key::from_pkcs8_der_in(der, Global) {
        let spki = key.public_key().to_spki_der().to_vec();
        return Some(Key {
          secret: Arc::new(Secret::$variant(key)),
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

#[derive(Debug)]
enum Secret {
  Dsa44(Box<MlDsa44SecretKey>),
  Dsa65(Box<MlDsa65SecretKey>),
  Dsa87(Box<MlDsa87SecretKey>),
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
    match self {
      Self::Dsa44(key) => key
        .sign_with(message, b"", fill)
        .map(|signature| signature.as_bytes().to_vec()),
      Self::Dsa65(key) => key
        .sign_with(message, b"", fill)
        .map(|signature| signature.as_bytes().to_vec()),
      Self::Dsa87(key) => key
        .sign_with(message, b"", fill)
        .map(|signature| signature.as_bytes().to_vec()),
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

  fn algorithm(&self) -> SignatureAlgorithm {
    // This legacy enum describes TLS 1.2 algorithms. ML-DSA is TLS 1.3 only;
    // rustls selects it through choose_scheme, never through this enum.
    SignatureAlgorithm::Unknown(0)
  }
}

#[derive(Debug)]
struct Sign {
  secret: Arc<Secret>,
  random: &'static dyn rustls::crypto::SecureRandom,
}

impl Signer for Sign {
  fn sign(&self, message: &[u8]) -> Result<Vec<u8>, Error> {
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
    let key = load(
      &PrivateKeyDer::try_from(der).expect("fixture container"),
      crate::random::SECURE_RANDOM,
    )
    .expect("fixture key");
    let result = key.secret.sign_with(b"message", |out| {
      out[..17].fill(0x42);
      Err(MlDsaError::RandomGenerationFailed)
    });
    assert!(matches!(result, Err(Error::General(message)) if message == "TLS signing failed"));
    let signer = key.choose_scheme(&[key.secret.scheme()]).expect("scheme");
    let signature = signer.sign(b"message").expect("sign after failure");
    let peer = rustls::crypto::aws_lc_rs::default_provider().signature_verification_algorithms;
    let verifier = peer
      .mapping
      .iter()
      .find(|(scheme, _)| *scheme == signer.scheme())
      .expect("peer scheme")
      .1[0];
    // RFC 9881 SPKI has a 22-byte header for all three parameter sets.
    verifier
      .verify_signature(&key.spki[22..], b"message", &signature)
      .expect("independent signature check");
  }
}
