//! Private-key loading and TLS 1.3 signing through rscrypto.
//!
//! PKCS#8 supports each enabled signing algorithm; SEC1 supports ECDSA, and
//! PKCS#1 supports RSA. ECDSA and RSA signing require `getrandom`. RSA signing
//! keys must have 3,072–8,192-bit moduli and public exponent 65,537.

use alloc::boxed::Box;
use rustls::Error;
use rustls::crypto::KeyProvider;
use rustls::crypto::SigningKey;
use rustls::pki_types::PrivateKeyDer;

#[cfg(all(feature = "ml-dsa", feature = "getrandom"))]
mod mldsa;

/// Load enabled signing keys through rscrypto's validated key codecs.
///
/// Containers larger than 16 KiB, unsupported formats, and invalid keys are
/// rejected. Successful keys return SPKI for `rustls::crypto::Credentials::new`.
/// Owned input DER is cleared on every return path. This provider uses OS entropy
/// and prepares ML-DSA keys for repeated signing; see [`Provider::compact`].
pub static KEY_PROVIDER: &dyn KeyProvider = &Provider {
  #[cfg(feature = "getrandom")]
  random: crate::random::SECURE_RANDOM,
  #[cfg(all(feature = "ml-dsa", feature = "getrandom"))]
  compact: false,
};

/// Private-key loader with a caller-selected entropy source.
///
/// Loaded keys retain that source for signing. Ed25519 does not request entropy.
#[derive(Debug)]
pub struct Provider {
  #[cfg(feature = "getrandom")]
  random: &'static dyn rustls::crypto::SecureRandom,
  #[cfg(all(feature = "ml-dsa", feature = "getrandom"))]
  compact: bool,
}

#[cfg(feature = "getrandom")]
impl Provider {
  /// Use `random` for key validation, blinding, and randomized signing.
  ///
  /// The source must provide cryptographically secure random bytes. Its failures
  /// propagate without fallback to the OS. Key formats, bounds, and owned DER
  /// cleanup match [`KEY_PROVIDER`].
  #[must_use]
  pub const fn new(random: &'static dyn rustls::crypto::SecureRandom) -> Self {
    Self {
      random,
      #[cfg(all(feature = "ml-dsa", feature = "getrandom"))]
      compact: false,
    }
  }

  /// Keep loaded ML-DSA keys compact instead of prepared.
  ///
  /// By default each loaded ML-DSA key is prepared once and keeps its
  /// transformed secrets and public matrix: 28,800, 48,256, or 81,024 bytes
  /// for ML-DSA-44, 65, or 87, in place of the 2,560, 4,032, or 4,896-byte
  /// compact key. A compact key prepares that storage again for every
  /// signature, which is slower and produces the same signatures. Other
  /// algorithms are unaffected.
  #[cfg(feature = "ml-dsa")]
  #[must_use]
  pub const fn compact(self) -> Self {
    Self { compact: true, ..self }
  }
}

impl KeyProvider for Provider {
  fn load_private_key(&self, key_der: PrivateKeyDer<'static>) -> Result<Box<dyn SigningKey>, Error> {
    // pki-types exposes Zeroize but does not clear owned DER automatically.
    let key_der = zeroize::Zeroizing::new(key_der);
    if key_der.secret_der().len() > 16_384 {
      return Err(key_rejected());
    }
    #[cfg(feature = "getrandom")]
    let _random = self.random;
    #[cfg(all(feature = "ml-dsa", feature = "getrandom"))]
    if let Some(key) = mldsa::load(&key_der, _random, self.compact) {
      return Ok(Box::new(key));
    }
    #[cfg(all(feature = "rsa", feature = "getrandom"))]
    if let Some(key) = rsa::load(&key_der, _random) {
      return Ok(Box::new(key));
    }
    #[cfg(any(
      feature = "ed25519",
      all(feature = "getrandom", any(feature = "ecdsa-p256", feature = "ecdsa-p384"))
    ))]
    if let Some(key) = curve::load(
      &key_der,
      #[cfg(feature = "getrandom")]
      _random,
    )? {
      return Ok(Box::new(key));
    }
    Err(key_rejected())
  }
}

fn key_rejected() -> Error {
  Error::General("invalid or unsupported TLS private key".into())
}

#[cfg(all(
  feature = "getrandom",
  any(feature = "ecdsa-p256", feature = "ecdsa-p384", feature = "rsa", feature = "ml-dsa")
))]
fn signing_failed() -> Error {
  Error::General("TLS signing failed".into())
}

#[cfg(any(
  feature = "ed25519",
  all(feature = "getrandom", any(feature = "ecdsa-p256", feature = "ecdsa-p384"))
))]
mod curve {
  use alloc::{boxed::Box, sync::Arc, vec::Vec};
  use rustls::Error;
  use rustls::crypto::SignatureScheme;
  use rustls::crypto::{Signer, SigningKey};
  use rustls::pki_types::{PrivateKeyDer, SubjectPublicKeyInfoDer};

  #[cfg(all(feature = "ecdsa-p256", feature = "getrandom"))]
  use rscrypto::EcdsaP256SecretKey;
  #[cfg(all(feature = "ecdsa-p384", feature = "getrandom"))]
  use rscrypto::EcdsaP384SecretKey;
  #[cfg(feature = "ed25519")]
  use rscrypto::Ed25519SecretKey;

  pub(super) fn load(
    der: &PrivateKeyDer<'_>,
    #[cfg(feature = "getrandom")] _random: &'static dyn rustls::crypto::SecureRandom,
  ) -> Result<Option<Key>, Error> {
    #[cfg(all(feature = "ecdsa-p256", feature = "getrandom"))]
    {
      let key = match der {
        PrivateKeyDer::Pkcs8(value) => EcdsaP256SecretKey::from_pkcs8_der(value.secret_pkcs8_der()),
        PrivateKeyDer::Sec1(value) => EcdsaP256SecretKey::from_sec1_der(value.secret_sec1_der()),
        _ => Err(rscrypto::EcdsaError::UnsupportedEncoding),
      };
      if let Ok(key) = key {
        let public = key.try_public_key_blinded_with(|out| _random.fill(out))?;
        return Ok(Some(Key {
          secret: Arc::new(Secret::P256(key, _random)),
          spki: public.to_spki_der().to_vec(),
        }));
      }
    }
    #[cfg(all(feature = "ecdsa-p384", feature = "getrandom"))]
    {
      let key = match der {
        PrivateKeyDer::Pkcs8(value) => EcdsaP384SecretKey::from_pkcs8_der(value.secret_pkcs8_der()),
        PrivateKeyDer::Sec1(value) => EcdsaP384SecretKey::from_sec1_der(value.secret_sec1_der()),
        _ => Err(rscrypto::EcdsaError::UnsupportedEncoding),
      };
      if let Ok(key) = key {
        let public = key.try_public_key_blinded_with(|out| _random.fill(out))?;
        return Ok(Some(Key {
          secret: Arc::new(Secret::P384(key, _random)),
          spki: public.to_spki_der().to_vec(),
        }));
      }
    }
    #[cfg(feature = "ed25519")]
    if let PrivateKeyDer::Pkcs8(value) = der
      && let Ok(key) = Ed25519SecretKey::from_pkcs8_der(value.secret_pkcs8_der())
    {
      let spki = key.public_key().to_spki_der().to_vec();
      return Ok(Some(Key {
        secret: Arc::new(Secret::Ed25519(key)),
        spki,
      }));
    }
    Ok(None)
  }

  #[derive(Debug)]
  enum Secret {
    #[cfg(all(feature = "ecdsa-p256", feature = "getrandom"))]
    P256(EcdsaP256SecretKey, &'static dyn rustls::crypto::SecureRandom),
    #[cfg(all(feature = "ecdsa-p384", feature = "getrandom"))]
    P384(EcdsaP384SecretKey, &'static dyn rustls::crypto::SecureRandom),
    #[cfg(feature = "ed25519")]
    Ed25519(Ed25519SecretKey),
  }

  impl Secret {
    fn scheme(&self) -> SignatureScheme {
      match self {
        #[cfg(all(feature = "ecdsa-p256", feature = "getrandom"))]
        Self::P256(..) => SignatureScheme::ECDSA_NISTP256_SHA256,
        #[cfg(all(feature = "ecdsa-p384", feature = "getrandom"))]
        Self::P384(..) => SignatureScheme::ECDSA_NISTP384_SHA384,
        #[cfg(feature = "ed25519")]
        Self::Ed25519(_) => SignatureScheme::ED25519,
      }
    }
  }

  #[derive(Debug)]
  pub(super) struct Key {
    secret: Arc<Secret>,
    spki: Vec<u8>,
  }

  impl SigningKey for Key {
    fn choose_scheme(&self, offered: &[SignatureScheme]) -> Option<Box<dyn Signer>> {
      offered.contains(&self.secret.scheme()).then(|| {
        Box::new(Sign {
          secret: Arc::clone(&self.secret),
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
  }

  #[cfg(all(feature = "ecdsa-p256", feature = "getrandom"))]
  fn sign_p256<E>(
    key: &EcdsaP256SecretKey,
    message: &[u8],
    fill: impl FnOnce(&mut [u8; 64]) -> Result<(), E>,
  ) -> Result<Vec<u8>, Error> {
    let signature = key
      .try_sign_blinded_with(message, fill)
      .map_err(|_| super::signing_failed())?;
    Ok(
      signature
        .to_der_into(&mut [0; rscrypto::EcdsaP256Signature::DER_MAX_LENGTH])
        .to_vec(),
    )
  }

  #[cfg(all(feature = "ecdsa-p384", feature = "getrandom"))]
  fn sign_p384<E>(
    key: &EcdsaP384SecretKey,
    message: &[u8],
    fill: impl FnOnce(&mut [u8; 96]) -> Result<(), E>,
  ) -> Result<Vec<u8>, Error> {
    let signature = key
      .try_sign_blinded_with(message, fill)
      .map_err(|_| super::signing_failed())?;
    Ok(
      signature
        .to_der_into(&mut [0; rscrypto::EcdsaP384Signature::DER_MAX_LENGTH])
        .to_vec(),
    )
  }

  impl Signer for Sign {
    fn sign(self: Box<Self>, message: &[u8]) -> Result<Vec<u8>, Error> {
      match self.secret.as_ref() {
        #[cfg(all(feature = "ecdsa-p256", feature = "getrandom"))]
        Secret::P256(key, random) => sign_p256(key, message, |out| random.fill(out)),
        #[cfg(all(feature = "ecdsa-p384", feature = "getrandom"))]
        Secret::P384(key, random) => sign_p384(key, message, |out| random.fill(out)),
        #[cfg(feature = "ed25519")]
        Secret::Ed25519(key) => Ok(key.sign(message).as_bytes().to_vec()),
      }
    }

    fn scheme(&self) -> SignatureScheme {
      self.secret.scheme()
    }
  }

  #[cfg(all(test, feature = "getrandom", feature = "ecdsa-p256", feature = "ecdsa-p384"))]
  #[test]
  fn ecdsa_partial_entropy_failure_is_opaque() {
    let p256 = EcdsaP256SecretKey::from_pkcs8_der(include_bytes!("../tests/vectors/credentials/p256-leaf-key.der"))
      .expect("fixture key");
    let p384 = EcdsaP384SecretKey::from_pkcs8_der(include_bytes!("../tests/vectors/credentials/p384-leaf-key.der"))
      .expect("fixture key");
    let errors = [
      sign_p256(&p256, b"message", |out| {
        out[..17].fill(0x42);
        Err(())
      }),
      sign_p384(&p384, b"message", |out| {
        out[..17].fill(0x42);
        Err(())
      }),
    ];
    for result in errors {
      assert!(matches!(result, Err(Error::General(message)) if message == "TLS signing failed"));
    }
  }
}

#[cfg(all(feature = "rsa", feature = "getrandom"))]
mod rsa {
  use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};
  use core::fmt;
  use std::sync::Mutex;

  use rscrypto::{RsaPrivateKey, RsaPrivateScratch, RsaPublicKeyPolicy};
  use rustls::Error;
  use rustls::crypto::SignatureScheme;
  use rustls::crypto::{Signer, SigningKey};
  use rustls::pki_types::{PrivateKeyDer, SubjectPublicKeyInfoDer};

  const SCHEMES: &[SignatureScheme] = &[
    SignatureScheme::RSA_PSS_SHA512,
    SignatureScheme::RSA_PSS_SHA384,
    SignatureScheme::RSA_PSS_SHA256,
  ];

  pub(super) fn load(der: &PrivateKeyDer<'_>, random: &'static dyn rustls::crypto::SecureRandom) -> Option<Key> {
    let policy = RsaPublicKeyPolicy::modern_verification();
    let key = match der {
      PrivateKeyDer::Pkcs1(value) => RsaPrivateKey::from_pkcs1_der_with_policy(value.secret_pkcs1_der(), &policy),
      PrivateKeyDer::Pkcs8(value) => RsaPrivateKey::from_pkcs8_der_with_policy(value.secret_pkcs8_der(), &policy),
      _ => return None,
    }
    .ok()?;
    let spki = key.public_key().to_spki_der();
    Some(Key {
      key: Arc::new(key),
      spki,
      random,
    })
  }

  #[derive(Debug)]
  pub(super) struct Key {
    key: Arc<RsaPrivateKey>,
    spki: Vec<u8>,
    random: &'static dyn rustls::crypto::SecureRandom,
  }

  impl SigningKey for Key {
    fn choose_scheme(&self, offered: &[SignatureScheme]) -> Option<Box<dyn Signer>> {
      SCHEMES
        .iter()
        .copied()
        .find(|scheme| offered.contains(scheme))
        .map(|scheme| {
          Box::new(Sign {
            scratch: Mutex::new(self.key.private_scratch()),
            key: Arc::clone(&self.key),
            scheme,
            random: self.random,
          }) as Box<dyn Signer>
        })
    }

    fn public_key(&self) -> Option<SubjectPublicKeyInfoDer<'_>> {
      Some(SubjectPublicKeyInfoDer::from(self.spki.as_slice()))
    }
  }

  struct Sign {
    key: Arc<RsaPrivateKey>,
    scratch: Mutex<RsaPrivateScratch>,
    scheme: SignatureScheme,
    random: &'static dyn rustls::crypto::SecureRandom,
  }

  impl fmt::Debug for Sign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.debug_struct("RsaSigner")
        .field("scheme", &self.scheme)
        .finish_non_exhaustive()
    }
  }

  impl Sign {
    fn sign_with_random_fill<E>(
      &self,
      message: &[u8],
      fill: impl FnMut(&mut [u8]) -> Result<(), E>,
    ) -> Result<Vec<u8>, Error> {
      let mut scratch = self.scratch.lock().map_err(|_| super::signing_failed())?;
      let mut signature = vec![0; self.key.signature_len()];
      self
        .key
        .sign_tls13_signature_scheme_with_random_fill_and_scratch(
          u16::from_be_bytes(self.scheme.to_array()),
          message,
          &mut signature,
          &mut scratch,
          fill,
        )
        .map_err(|_| super::signing_failed())?;
      Ok(signature)
    }
  }

  impl Signer for Sign {
    fn sign(self: Box<Self>, message: &[u8]) -> Result<Vec<u8>, Error> {
      self.sign_with_random_fill(message, |out| self.random.fill(out))
    }

    fn scheme(&self) -> SignatureScheme {
      self.scheme
    }
  }

  #[cfg(test)]
  #[test]
  fn rsa_partial_entropy_failure_does_not_release_a_signature() {
    let key = RsaPrivateKey::from_pkcs8_der(include_bytes!("../tests/vectors/credentials/rsa3072-leaf-key.der"))
      .expect("fixture key");
    let signer = Sign {
      scratch: Mutex::new(key.private_scratch()),
      key: Arc::new(key),
      scheme: SignatureScheme::RSA_PSS_SHA256,
      random: crate::random::SECURE_RANDOM,
    };
    let result = signer.sign_with_random_fill(b"message", |out| {
      out[..17].fill(0x42);
      Err(())
    });
    assert!(matches!(result, Err(Error::General(message)) if message == "TLS signing failed"));
    // A failed entropy request must leave reusable scratch, not a partial signature.
    let scheme = signer.scheme;
    let signature = Box::new(signer).sign(b"message").expect("sign after entropy failure");
    let peer = rustls_ring::DEFAULT_TLS13_PROVIDER.signature_verification_algorithms;
    let verifier = peer
      .mapping()
      .iter()
      .find(|(candidate, _)| *candidate == scheme)
      .expect("RSA-PSS verifier")
      .1[0];
    verifier
      .verify_signature(
        include_bytes!("../tests/vectors/credentials/rsa3072-public.der"),
        b"message",
        &signature,
      )
      .expect("independent verification");
  }
}
