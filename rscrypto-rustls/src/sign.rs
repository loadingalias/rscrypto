//! Private-key loading and TLS signing through rscrypto.
//!
//! PKCS#8 supports each enabled signing algorithm; SEC1 supports ECDSA, and
//! PKCS#1 supports RSA. RSA signing keys have 2,048–8,192-bit moduli by default
//! (`RsaSigningKeySizes` can require 3,072 bits) and public exponent 65,537.
//! RSA keys sign RSA-PSS, and also PKCS#1 v1.5 for TLS 1.2 peers; rustls never
//! offers PKCS#1 v1.5 in TLS 1.3. Each ECDSA key signs with one scheme in every
//! TLS version: P-256 with SHA-256, and P-384 with SHA-384. [`Provider::new`]
//! takes the entropy source for key validation and signing; [`KEY_PROVIDER`]
//! uses OS entropy and requires `getrandom`. With `chacha20-rng`,
//! `Provider::scoped` loads keys for replayable runs.

use alloc::boxed::Box;
use rustls::Error;
use rustls::crypto::KeyProvider;
use rustls::crypto::SigningKey;
use rustls::pki_types::PrivateKeyDer;

#[cfg(any(feature = "ml-dsa", feature = "mldsa65-ed25519", feature = "mldsa65-ecdsa-p256"))]
macro_rules! impl_tls_signer {
  ($entropy_error:path) => {
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
        self
          .secret
          .sign_with(message, |out| self.random.fill(out).map_err(|_| $entropy_error))
      }

      fn scheme(&self) -> SignatureScheme {
        self.secret.scheme()
      }
    }
  };
}

#[cfg(any(feature = "mldsa65-ed25519", feature = "mldsa65-ecdsa-p256"))]
mod composite;
#[cfg(feature = "ml-dsa")]
mod mldsa;

/// Load enabled signing keys through rscrypto's validated key codecs.
///
/// Containers larger than 16 KiB, unsupported formats, and invalid keys are
/// rejected. Successful keys return SPKI for `rustls::crypto::Credentials::new`.
/// Owned input DER is cleared on every return path. This provider uses OS entropy
/// and prepares ML-DSA keys for repeated signing; see [`Provider::compact`].
#[cfg(feature = "getrandom")]
pub static KEY_PROVIDER: &dyn KeyProvider = &Provider::new(crate::random::SECURE_RANDOM);

/// Private-key loader with a caller-selected entropy source.
///
/// Loaded keys retain that source for signing. Ed25519 does not request entropy.
#[derive(Debug)]
pub struct Provider {
  random: &'static dyn rustls::crypto::SecureRandom,
  /// Whether each loaded RSA key seeds its own blinding generator from `random` at load.
  #[cfg(all(feature = "rsa", feature = "chacha20-rng"))]
  scoped: bool,
  #[cfg(any(
    feature = "ml-dsa",
    feature = "rsa",
    feature = "mldsa65-ed25519",
    feature = "mldsa65-ecdsa-p256"
  ))]
  compact: bool,
  #[cfg(feature = "rsa")]
  rsa_sizes: RsaSigningKeySizes,
}

/// RSA signing-key modulus lengths that a [`Provider`] loads.
///
/// Both accept only public exponent 65,537. Peer keys in certificates and
/// handshake signatures are verified separately and are not affected.
#[cfg(feature = "rsa")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RsaSigningKeySizes {
  /// 2,048 to 8,192 bits, the default, so deployed RSA-2048 certificates work.
  ///
  /// RSA-2048 gives about 112-bit security (NIST SP 800-57 Part 1).
  Bits2048To8192,
  /// 3,072 to 8,192 bits, about 128-bit security or more.
  Bits3072To8192,
}

impl Provider {
  /// Use `random` for key validation, blinding, and randomized signing.
  ///
  /// The source must provide cryptographically secure random bytes. Its failures
  /// propagate without fallback to the OS. Key formats, bounds, and owned DER
  /// cleanup match [`KEY_PROVIDER`]. The blinding refreshes of an RSA key are
  /// draws by a shared owner, which `replay::ScopedRandom` refuses; for that
  /// source, use `Provider::scoped`.
  #[must_use]
  pub const fn new(random: &'static dyn rustls::crypto::SecureRandom) -> Self {
    Self {
      random,
      #[cfg(all(feature = "rsa", feature = "chacha20-rng"))]
      scoped: false,
      #[cfg(any(
        feature = "ml-dsa",
        feature = "rsa",
        feature = "mldsa65-ed25519",
        feature = "mldsa65-ecdsa-p256"
      ))]
      compact: false,
      #[cfg(feature = "rsa")]
      rsa_sizes: RsaSigningKeySizes::Bits2048To8192,
    }
  }

  /// Load keys for replayable runs over [`crate::replay::ScopedRandom`].
  ///
  /// Each signature draws its salt, nonce, or hedge from the stream of the
  /// connection that signs. An RSA key also seeds its own blinding generator at
  /// load, from 32 bytes of the scope that loads it, and draws each blinding
  /// refresh from that generator, never from a connection. So load keys inside
  /// a scope; see [Shared owners](crate::replay#shared-owners). The other
  /// options compose as with [`Provider::new`].
  #[cfg(feature = "chacha20-rng")]
  #[must_use]
  pub const fn scoped() -> Self {
    Self {
      #[cfg(feature = "rsa")]
      scoped: true,
      ..Self::new(&crate::replay::SCOPED)
    }
  }

  /// Load only RSA signing keys whose moduli fall in `sizes`.
  ///
  /// Keys outside the range are rejected like any invalid key. The default is
  /// [`RsaSigningKeySizes::Bits2048To8192`].
  #[cfg(feature = "rsa")]
  #[must_use]
  pub const fn rsa_key_sizes(self, sizes: RsaSigningKeySizes) -> Self {
    Self {
      rsa_sizes: sizes,
      ..self
    }
  }

  /// Keep loaded ML-DSA and RSA keys compact instead of prepared.
  ///
  /// By default each loaded ML-DSA key is prepared once and keeps its
  /// transformed secrets and public matrix: 28,800, 48,256, or 81,024 bytes
  /// for ML-DSA-44, 65, or 87, in place of the 2,560, 4,032, or 4,896-byte
  /// compact key. A compact key prepares that storage again for every
  /// signature, which is slower and produces the same signatures.
  ///
  /// By default each loaded RSA key also keeps idle signing workspaces, at
  /// most one per available CPU and never more than 64. A workspace holds
  /// private-operation scratch and a blinding pair that the next signature
  /// squares instead of drawing, inverting, and exponentiating a new factor;
  /// see [`rscrypto::RsaSigningWorkspace`]. A compact RSA key builds a
  /// workspace and draws a new blinding factor for every signature. Other
  /// algorithms are unaffected.
  #[cfg(any(
    feature = "ml-dsa",
    feature = "rsa",
    feature = "mldsa65-ed25519",
    feature = "mldsa65-ecdsa-p256"
  ))]
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
    let _random = self.random;
    #[cfg(any(feature = "mldsa65-ed25519", feature = "mldsa65-ecdsa-p256"))]
    if let Some(key) = composite::load(&key_der, _random, self.compact) {
      return Ok(Box::new(key));
    }
    #[cfg(feature = "ml-dsa")]
    if let Some(key) = mldsa::load(&key_der, _random, self.compact) {
      return Ok(Box::new(key));
    }
    #[cfg(feature = "rsa")]
    if let Some(key) = rsa::load(
      &key_der,
      _random,
      self.compact,
      self.rsa_sizes,
      #[cfg(feature = "chacha20-rng")]
      self.scoped,
    )? {
      return Ok(Box::new(key));
    }
    #[cfg(any(feature = "ed25519", feature = "ecdsa-p256", feature = "ecdsa-p384"))]
    if let Some(key) = curve::load(&key_der, _random)? {
      return Ok(Box::new(key));
    }
    Err(key_rejected())
  }
}

fn key_rejected() -> Error {
  Error::General("invalid or unsupported TLS private key".into())
}

#[cfg(any(
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "rsa",
  feature = "ml-dsa",
  feature = "mldsa65-ed25519",
  feature = "mldsa65-ecdsa-p256"
))]
fn signing_failed() -> Error {
  Error::General("TLS signing failed".into())
}

#[cfg(any(feature = "ed25519", feature = "ecdsa-p256", feature = "ecdsa-p384"))]
mod curve {
  use alloc::{boxed::Box, sync::Arc, vec::Vec};
  use rustls::Error;
  use rustls::crypto::SignatureScheme;
  use rustls::crypto::{Signer, SigningKey};
  use rustls::pki_types::{PrivateKeyDer, SubjectPublicKeyInfoDer};

  #[cfg(feature = "ecdsa-p256")]
  use rscrypto::EcdsaP256SecretKey;
  #[cfg(feature = "ecdsa-p384")]
  use rscrypto::EcdsaP384SecretKey;
  #[cfg(feature = "ed25519")]
  use rscrypto::{Ed25519Keypair, Ed25519SecretKey};

  pub(super) fn load(
    der: &PrivateKeyDer<'_>,
    _random: &'static dyn rustls::crypto::SecureRandom,
  ) -> Result<Option<Key>, Error> {
    #[cfg(feature = "ecdsa-p256")]
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
    #[cfg(feature = "ecdsa-p384")]
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
      // The configuration-bound key is expanded once; signing reuses the expanded secret and
      // public key instead of deriving both for every signature.
      let key = Ed25519Keypair::from_secret_key(key);
      let spki = key.public_key().to_spki_der().to_vec();
      return Ok(Some(Key {
        secret: Arc::new(Secret::Ed25519(key)),
        spki,
      }));
    }
    Ok(None)
  }

  #[derive(Debug)]
  #[cfg_attr(
    all(feature = "ed25519", any(feature = "ecdsa-p256", feature = "ecdsa-p384")),
    expect(
      clippy::large_enum_variant,
      reason = "each configured key holds one `Secret` in an `Arc`; boxing the keypair would add an allocation and \
                another moved-from secret copy"
    )
  )]
  enum Secret {
    #[cfg(feature = "ecdsa-p256")]
    P256(EcdsaP256SecretKey, &'static dyn rustls::crypto::SecureRandom),
    #[cfg(feature = "ecdsa-p384")]
    P384(EcdsaP384SecretKey, &'static dyn rustls::crypto::SecureRandom),
    #[cfg(feature = "ed25519")]
    Ed25519(Ed25519Keypair),
  }

  impl Secret {
    fn scheme(&self) -> SignatureScheme {
      match self {
        #[cfg(feature = "ecdsa-p256")]
        Self::P256(..) => SignatureScheme::ECDSA_NISTP256_SHA256,
        #[cfg(feature = "ecdsa-p384")]
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

  #[cfg(feature = "ecdsa-p256")]
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

  #[cfg(feature = "ecdsa-p384")]
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
        #[cfg(feature = "ecdsa-p256")]
        Secret::P256(key, random) => sign_p256(key, message, |out| random.fill(out)),
        #[cfg(feature = "ecdsa-p384")]
        Secret::P384(key, random) => sign_p384(key, message, |out| random.fill(out)),
        #[cfg(feature = "ed25519")]
        Secret::Ed25519(key) => Ok(key.sign(message).as_bytes().to_vec()),
      }
    }

    fn scheme(&self) -> SignatureScheme {
      self.secret.scheme()
    }
  }

  #[cfg(all(test, feature = "ecdsa-p256", feature = "ecdsa-p384"))]
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

#[cfg(feature = "rsa")]
mod rsa {
  use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};
  use core::fmt;
  use std::sync::Mutex;

  use rscrypto::{
    RsaPkcs1v15Profile, RsaPrivateKey, RsaPssProfile, RsaPublicKeyPolicy, RsaSignatureProfile, RsaSigningWorkspace,
  };
  use rustls::Error;
  use rustls::crypto::SignatureScheme;
  use rustls::crypto::{GetRandomFailed, SecureRandom};
  use rustls::crypto::{Signer, SigningKey};
  use rustls::pki_types::{PrivateKeyDer, SubjectPublicKeyInfoDer};

  // Preference order. TLS 1.3 permits only the RSA-PSS schemes, and rustls
  // removes the others from TLS 1.3 offers. PKCS#1 v1.5 serves TLS 1.2 peers
  // that offer nothing else, such as servers requesting client certificates.
  // Loaded keys are `rsaEncryption` keys (the PKCS#8 import rejects
  // `id-RSASSA-PSS`), for which both encodings and `rsa_pss_rsae_*` are valid.
  const SCHEMES: &[(SignatureScheme, RsaSignatureProfile)] = &[
    (
      SignatureScheme::RSA_PSS_SHA512,
      RsaSignatureProfile::pss(RsaPssProfile::Sha512),
    ),
    (
      SignatureScheme::RSA_PSS_SHA384,
      RsaSignatureProfile::pss(RsaPssProfile::Sha384),
    ),
    (
      SignatureScheme::RSA_PSS_SHA256,
      RsaSignatureProfile::pss(RsaPssProfile::Sha256),
    ),
    (
      SignatureScheme::RSA_PKCS1_SHA512,
      RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha512),
    ),
    (
      SignatureScheme::RSA_PKCS1_SHA384,
      RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha384),
    ),
    (
      SignatureScheme::RSA_PKCS1_SHA256,
      RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256),
    ),
  ];

  /// Idle signing workspaces that one loaded key keeps, at most.
  const MAX_IDLE_WORKSPACES: usize = 64;

  /// Load an RSA key, or `None` if `der` holds no valid RSA key in `sizes`.
  ///
  /// With `scoped`, the key seeds its own blinding generator from 32 bytes of
  /// `random`, which fails the load if `random` fails.
  pub(super) fn load(
    der: &PrivateKeyDer<'_>,
    random: &'static dyn SecureRandom,
    compact: bool,
    sizes: super::RsaSigningKeySizes,
    #[cfg(feature = "chacha20-rng")] scoped: bool,
  ) -> Result<Option<Key>, Error> {
    let policy = match sizes {
      super::RsaSigningKeySizes::Bits2048To8192 => RsaPublicKeyPolicy::modulus_2048_to_8192(),
      super::RsaSigningKeySizes::Bits3072To8192 => RsaPublicKeyPolicy::modulus_3072_to_8192(),
    };
    let key = match der {
      PrivateKeyDer::Pkcs1(value) => RsaPrivateKey::from_pkcs1_der_with_policy(value.secret_pkcs1_der(), &policy),
      PrivateKeyDer::Pkcs8(value) => RsaPrivateKey::from_pkcs8_der_with_policy(value.secret_pkcs8_der(), &policy),
      _ => return Ok(None),
    };
    let Ok(key) = key else {
      return Ok(None);
    };
    #[cfg(feature = "chacha20-rng")]
    let blinding = if scoped {
      Some(crate::replay::Stream::new(rscrypto::ChaCha20Rng::try_generate_with(
        |seed| random.fill(seed),
      )?))
    } else {
      None
    };
    let spki = key.public_key().to_spki_der();
    let idle_limit = if compact {
      0
    } else {
      std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(MAX_IDLE_WORKSPACES)
    };
    Ok(Some(Key {
      shared: Arc::new(Shared {
        key,
        idle: Mutex::new(Vec::new()),
        idle_limit,
        #[cfg(feature = "chacha20-rng")]
        blinding,
      }),
      spki,
      random,
    }))
  }

  /// A loaded key and its idle workspaces, shared by the key and its signers.
  struct Shared {
    key: RsaPrivateKey,
    /// Workspaces that no signature is using; at most `idle_limit`. Signing
    /// clears a workspace's scratch before it returns, and the workspace keeps
    /// its blinding pair between signatures.
    idle: Mutex<Vec<RsaSigningWorkspace>>,
    /// Zero for a compact key, which keeps no workspace.
    idle_limit: usize,
    /// The key's own blinding generator, with `Provider::scoped`.
    #[cfg(feature = "chacha20-rng")]
    blinding: Option<crate::replay::Stream>,
  }

  impl Shared {
    /// Take an idle workspace, or build one.
    fn take(&self) -> RsaSigningWorkspace {
      // Only `Vec::pop` and `Vec::push` run under the lock, so a poisoned
      // lock still holds whole workspaces.
      let idle = self
        .idle
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .pop();
      idle.unwrap_or_else(|| self.key.signing_workspace())
    }

    /// Fill a blinding refresh: from the key's own generator, or from `random`
    /// as a shared owner, never from the stream of a connection.
    fn fill_blinding(&self, random: &dyn SecureRandom, output: &mut [u8]) -> Result<(), GetRandomFailed> {
      #[cfg(feature = "chacha20-rng")]
      if let Some(stream) = &self.blinding {
        return stream.fill(output);
      }
      crate::owner_fill(random, output)
    }

    /// Keep `workspace` for the next signature, or drop it at the limit.
    fn give_back(&self, workspace: RsaSigningWorkspace) {
      let mut idle = self.idle.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
      if idle.len() < self.idle_limit {
        idle.push(workspace);
      }
    }
  }

  impl fmt::Debug for Shared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.debug_struct("RsaKey")
        .field("key", &self.key)
        .field("idle_limit", &self.idle_limit)
        .finish_non_exhaustive()
    }
  }

  #[derive(Debug)]
  pub(super) struct Key {
    shared: Arc<Shared>,
    spki: Vec<u8>,
    random: &'static dyn rustls::crypto::SecureRandom,
  }

  impl SigningKey for Key {
    fn choose_scheme(&self, offered: &[SignatureScheme]) -> Option<Box<dyn Signer>> {
      SCHEMES
        .iter()
        .copied()
        .find(|(scheme, _)| offered.contains(scheme))
        .map(|(scheme, profile)| {
          // rustls asks for a signer on every ClientHello, including resumptions that never
          // sign, so a workspace is taken only when a signature is produced.
          Box::new(Sign {
            shared: Arc::clone(&self.shared),
            scheme,
            profile,
            random: self.random,
          }) as Box<dyn Signer>
        })
    }

    fn public_key(&self) -> Option<SubjectPublicKeyInfoDer<'_>> {
      Some(SubjectPublicKeyInfoDer::from(self.spki.as_slice()))
    }
  }

  struct Sign {
    shared: Arc<Shared>,
    scheme: SignatureScheme,
    profile: RsaSignatureProfile,
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
    /// Sign with the salt from `fill_salt` and each blinding refresh from `fill_blinding`.
    fn sign_with<E, F>(
      &self,
      message: &[u8],
      fill_salt: impl FnOnce(&mut [u8]) -> Result<(), E>,
      fill_blinding: impl FnMut(&mut [u8]) -> Result<(), F>,
    ) -> Result<Vec<u8>, Error> {
      let key = &self.shared.key;
      let mut signature = vec![0; key.signature_len()];
      let mut workspace = self.shared.take();
      // An error clears the workspace's pair; dropping it releases its buffers.
      key
        .sign_signature_with_random_fill_and_workspace(
          self.profile,
          message,
          &mut signature,
          &mut workspace,
          fill_salt,
          fill_blinding,
        )
        .map_err(|_| super::signing_failed())?;
      self.shared.give_back(workspace);
      Ok(signature)
    }
  }

  impl Signer for Sign {
    fn sign(self: Box<Self>, message: &[u8]) -> Result<Vec<u8>, Error> {
      // The salt belongs to this signature, so it comes from the signing
      // connection. The blinding pair belongs to the shared key.
      self.sign_with(
        message,
        |out| self.random.fill(out),
        |out| self.shared.fill_blinding(self.random, out),
      )
    }

    fn scheme(&self) -> SignatureScheme {
      self.scheme
    }
  }

  #[cfg(test)]
  fn idle_workspaces(key: &Key) -> usize {
    key.shared.idle.lock().expect("no test panics under the lock").len()
  }

  #[cfg(test)]
  #[test]
  fn rsa_schemes_match_an_independent_signer_and_fail_closed_on_entropy() {
    use rustls::pki_types::PrivatePkcs8KeyDer;

    let fixtures: [(&[u8], &[u8]); 2] = [
      (
        include_bytes!("../tests/vectors/credentials/rsa2048-leaf-key.der"),
        include_bytes!("../tests/vectors/credentials/rsa2048-public.der"),
      ),
      (
        include_bytes!("../tests/vectors/credentials/rsa3072-leaf-key.der"),
        include_bytes!("../tests/vectors/credentials/rsa3072-public.der"),
      ),
    ];
    let peer = rustls_ring::DEFAULT_PROVIDER;
    for ((der, public), compact) in fixtures
      .into_iter()
      .flat_map(|fixture| [(fixture, false), (fixture, true)])
    {
      let peer_key = peer
        .key_provider
        .load_private_key(PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(der.to_vec())))
        .expect("independent key");
      let key = load(
        &PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(der.to_vec())),
        &crate::tests::RANDOM,
        compact,
        super::RsaSigningKeySizes::Bits2048To8192,
        #[cfg(feature = "chacha20-rng")]
        false,
      )
      .expect("load")
      .expect("fixture key");
      let kept = usize::from(!compact);
      // RSA-PSS wins whenever a peer offers it: TLS 1.3 offers carry only RSA-PSS,
      // and TLS 1.2 peers get PKCS#1 v1.5 only when they offer nothing else.
      let preferred = key
        .choose_scheme(&[SignatureScheme::RSA_PKCS1_SHA512, SignatureScheme::RSA_PSS_SHA256])
        .expect("offered RSA scheme");
      assert_eq!(preferred.scheme(), SignatureScheme::RSA_PSS_SHA256);
      assert!(
        key
          .choose_scheme(&[SignatureScheme::RSA_PKCS1_SHA1, SignatureScheme::ECDSA_NISTP256_SHA256])
          .is_none()
      );
      assert_eq!(idle_workspaces(&key), 0, "choosing a scheme takes no workspace");
      for &(scheme, profile) in SCHEMES {
        let signature = key
          .choose_scheme(&[scheme])
          .expect("offered scheme")
          .sign(b"message")
          .expect("sign");
        let verifier = peer
          .signature_verification_algorithms
          .mapping()
          .iter()
          .find(|(candidate, _)| *candidate == scheme)
          .expect("independent verifier")
          .1[0];
        verifier
          .verify_signature(public, b"message", &signature)
          .expect("independent verification");
        assert!(verifier.verify_signature(public, b"massage", &signature).is_err());
        if matches!(profile, RsaSignatureProfile::Pkcs1v15(_)) {
          // PKCS#1 v1.5 is deterministic, so the independent signer's bytes are a known answer.
          let expected = peer_key
            .choose_scheme(&[scheme])
            .expect("independent scheme")
            .sign(b"message")
            .expect("independent signature");
          assert_eq!(signature, expected, "{scheme:?}");
        }
        assert_eq!(idle_workspaces(&key), kept, "{scheme:?}");
      }
      // Sequential signatures reuse one workspace across blinding refreshes,
      // which come every 32 signatures, and still give the known answer.
      let scheme = SignatureScheme::RSA_PKCS1_SHA256;
      let expected = peer_key
        .choose_scheme(&[scheme])
        .expect("independent scheme")
        .sign(b"message")
        .expect("independent signature");
      for index in 0..40 {
        let signature = key
          .choose_scheme(&[scheme])
          .expect("offered scheme")
          .sign(b"message")
          .expect("sign");
        assert_eq!(signature, expected, "signature {index}");
        assert_eq!(idle_workspaces(&key), kept, "signature {index}");
      }
      // A failed salt (PSS) or blinding request releases no partial signature
      // and drops the workspace it used, and the signer still signs afterwards.
      // Holding the idle workspace makes each attempt build a new one, so
      // every scheme requests blinding entropy.
      let fail = |out: &mut [u8]| {
        let partial = out.len().min(17);
        out[..partial].fill(0x42);
        Err(())
      };
      let fill = |out: &mut [u8]| crate::tests::RANDOM.fill(out);
      for &(scheme, profile) in SCHEMES {
        let failing_salt = matches!(profile, RsaSignatureProfile::Pss { .. });
        for salt_fails in [false, true]
          .into_iter()
          .filter(|&salt_fails| failing_salt || !salt_fails)
        {
          let signer = Sign {
            shared: Arc::clone(&key.shared),
            scheme,
            profile,
            random: &crate::tests::RANDOM,
          };
          let held = signer.shared.take();
          let result = if salt_fails {
            signer.sign_with(b"message", fail, fill)
          } else {
            signer.sign_with(b"message", fill, fail)
          };
          assert!(matches!(result, Err(Error::General(message)) if message == "TLS signing failed"));
          assert_eq!(idle_workspaces(&key), 0, "{scheme:?}");
          signer.shared.give_back(held);
          Box::new(signer).sign(b"message").expect("sign after entropy failure");
          assert_eq!(idle_workspaces(&key), kept, "{scheme:?}");
        }
      }
    }
  }
}
