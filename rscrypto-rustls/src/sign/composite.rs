//! Composite ML-DSA signing (draft-ietf-tls-composite-mldsa-00) with the
//! empty composite context that TLS requires and hedged ML-DSA.

use alloc::{alloc::Global, boxed::Box, sync::Arc, vec::Vec};
use rscrypto::{CompositeMlDsaError, MlDsa65PreparedSecretKeyStorage};
#[cfg(feature = "mldsa65-ecdsa-p256")]
use rscrypto::{MlDsa65EcdsaP256PreparedSecretKey, MlDsa65EcdsaP256SecretKey};
#[cfg(feature = "mldsa65-ed25519")]
use rscrypto::{MlDsa65Ed25519PreparedSecretKey, MlDsa65Ed25519SecretKey};
use rustls::Error;
use rustls::crypto::SignatureScheme;
use rustls::crypto::{Signer, SigningKey};
use rustls::pki_types::{PrivateKeyDer, SubjectPublicKeyInfoDer};

/// Load a composite key, prepared for repeated signing unless `compact`.
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
      if let Ok(key) = $key::from_pkcs8_der(der) {
        let spki = key.public_key().to_spki_der().to_vec();
        let form = if compact {
          Form::Compact(Box::new(key))
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
  #[cfg(feature = "mldsa65-ed25519")]
  load!(MlDsa65Ed25519SecretKey, Ed25519);
  #[cfg(feature = "mldsa65-ecdsa-p256")]
  load!(MlDsa65EcdsaP256SecretKey, EcdsaP256);
  None
}

/// A loaded key: the compact key, or its prepared form, which owns its
/// ML-DSA storage and a copy of the traditional key.
#[derive(Debug)]
enum Form<C, P> {
  Compact(Box<C>),
  Prepared(P),
}

#[derive(Debug)]
enum Secret {
  #[cfg(feature = "mldsa65-ed25519")]
  Ed25519(Form<MlDsa65Ed25519SecretKey, MlDsa65Ed25519PreparedSecretKey<Box<MlDsa65PreparedSecretKeyStorage>>>),
  #[cfg(feature = "mldsa65-ecdsa-p256")]
  EcdsaP256(Form<MlDsa65EcdsaP256SecretKey, MlDsa65EcdsaP256PreparedSecretKey<Box<MlDsa65PreparedSecretKeyStorage>>>),
}

impl Secret {
  fn scheme(&self) -> SignatureScheme {
    match self {
      #[cfg(feature = "mldsa65-ed25519")]
      Self::Ed25519(_) => crate::verify::MLDSA65_ED25519_SHA512,
      #[cfg(feature = "mldsa65-ecdsa-p256")]
      Self::EcdsaP256(_) => crate::verify::MLDSA65_ECDSA_SECP256R1_SHA512,
    }
  }

  fn sign_with(
    &self,
    message: &[u8],
    fill: impl FnMut(&mut [u8]) -> Result<(), CompositeMlDsaError>,
  ) -> Result<Vec<u8>, Error> {
    // rustls supplies the TLS CertificateVerify context and transcript. The
    // composite application context stays empty, as the TLS draft requires.
    // A compact key prepares heap storage for each signature, so ML-DSA
    // expands its matrix once; the signature bytes are the same.
    macro_rules! sign {
      ($form:expr) => {
        match $form {
          Form::Prepared(prepared) => prepared.sign_with(message, b"", fill),
          Form::Compact(key) => key
            .prepare_in(Global)
            .and_then(|prepared| prepared.sign_with(message, b"", fill)),
        }
        .map(|signature| signature.as_ref().to_vec())
      };
    }
    match self {
      #[cfg(feature = "mldsa65-ed25519")]
      Self::Ed25519(form) => sign!(form),
      #[cfg(feature = "mldsa65-ecdsa-p256")]
      Self::EcdsaP256(form) => sign!(form),
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
      self
        .random
        .fill(out)
        .map_err(|_| CompositeMlDsaError::RandomGenerationFailed)
    })
  }

  fn scheme(&self) -> SignatureScheme {
    self.secret.scheme()
  }
}

#[cfg(test)]
#[test]
fn partial_entropy_failure_is_opaque_and_key_stays_usable() {
  let fill = |out: &mut [u8]| {
    out.fill(0x5a);
    Ok(())
  };
  type Check = Box<dyn Fn(&[u8]) -> bool>;
  let mut keys: Vec<(Vec<u8>, Check)> = Vec::new();
  #[cfg(feature = "mldsa65-ed25519")]
  {
    let (public, secret) = rscrypto::MlDsa65Ed25519::generate_keypair(fill).expect("key");
    let mut der = vec![0; MlDsa65Ed25519SecretKey::PKCS8_DER_LENGTH];
    secret.to_pkcs8_der_into(der.as_mut_slice().try_into().expect("length"));
    keys.push((
      der,
      Box::new(move |signature| {
        rscrypto::MlDsa65Ed25519Signature::try_from_slice(signature)
          .is_ok_and(|signature| public.verify(b"message", &signature).is_ok())
      }),
    ));
  }
  #[cfg(feature = "mldsa65-ecdsa-p256")]
  {
    let (public, secret) = rscrypto::MlDsa65EcdsaP256::generate_keypair(fill).expect("key");
    let mut der = vec![0; MlDsa65EcdsaP256SecretKey::PKCS8_DER_LENGTH];
    secret.to_pkcs8_der_into(der.as_mut_slice().try_into().expect("length"));
    keys.push((
      der,
      Box::new(move |signature| {
        rscrypto::MlDsa65EcdsaP256Signature::try_from_slice(signature)
          .is_ok_and(|signature| public.verify(b"message", &signature).is_ok())
      }),
    ));
  }
  for (der, verify) in keys {
    for compact in [false, true] {
      let key = load(
        &PrivateKeyDer::try_from(der.as_slice()).expect("container"),
        &crate::tests::RANDOM,
        compact,
      )
      .expect("composite key");
      let result = key.secret.sign_with(b"message", |out| {
        out[..17].fill(0x42);
        Err(CompositeMlDsaError::RandomGenerationFailed)
      });
      assert!(matches!(result, Err(Error::General(message)) if message == "TLS signing failed"));
      let signer = key.choose_scheme(&[key.secret.scheme()]).expect("scheme");
      assert!(verify(&signer.sign(b"message").expect("sign after failure")));
    }
  }
}
