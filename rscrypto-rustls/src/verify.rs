//! Certificate and TLS signature verification through rscrypto.

/// Enabled certificate algorithms and TLS signature-scheme mappings.
#[cfg(any(
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "rsa",
  feature = "ml-dsa"
))]
pub const SUPPORTED_ALGORITHMS: rustls::crypto::WebPkiSupportedAlgorithms = implementation::ALGORITHMS;

/// Enabled certificate algorithms, including encodings used only in certificate chains.
///
/// Use this list with [`rustls::crypto::WebPkiSupportedAlgorithms::new`] when selecting a subset
/// of the TLS signature schemes in [`SUPPORTED_ALGORITHMS`].
#[cfg(any(
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "rsa",
  feature = "ml-dsa"
))]
pub const ALL_ALGORITHMS: &[&dyn rustls::pki_types::SignatureVerificationAlgorithm] = implementation::ALL;
#[cfg(any(
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "rsa",
  feature = "ml-dsa"
))]
pub use verifiers::{PreparedClientVerifier, PreparedServerVerifier};

#[cfg(any(
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "rsa",
  feature = "ml-dsa"
))]
mod verifiers {
  //! Certificate verifiers that prepare operator-configured trust anchors once.
  //!
  //! Each verifier wraps rustls's WebPKI verifier for the same roots, this
  //! crate's algorithms, and no revocation lists. A prepared anchor key is
  //! prepared from exactly the bytes webpki passes back for that anchor, so
  //! every result equals the wrapped verifier's. Keys a peer sends are imported
  //! per signature, as before, and never retained.

  use alloc::{borrow::Cow, sync::Arc, vec::Vec};
  use core::hash::Hasher;

  use rustls::client::danger::{HandshakeSignatureValid, ServerIdentity, ServerVerifier, SignatureVerificationInput};
  use rustls::client::{WebPkiServerVerifier, verify_identity_signed_by_trust_anchor, verify_server_name};
  use rustls::crypto::{
    CryptoProvider, GetRandomFailed, Identity, SignatureScheme, VerifiedIdentity, WebPkiSupportedAlgorithms,
  };
  use rustls::enums::CertificateType;
  use rustls::pki_types::SignatureVerificationAlgorithm;
  use rustls::server::danger::{ClientIdentity, ClientVerifier};
  use rustls::server::{ParsedCertificate, VerifierBuilderError, WebPkiClientVerifier};
  use rustls::{DistinguishedName, Error, RootCertStore};

  use super::implementation::{Anchors, PreparedAlgorithm, prepared_algorithms};

  /// Server-certificate verifier with P-256, P-384 and ML-DSA trust-anchor keys prepared once.
  ///
  /// Validation, name checks, errors, signature schemes and configuration
  /// hashing are those of [`WebPkiServerVerifier`] built from the same roots
  /// with [`SUPPORTED_ALGORITHMS`](super::SUPPORTED_ALGORITHMS) and without
  /// revocation lists; use that verifier when revocation checking is needed.
  /// Install it with `ClientConfig::builder(..).dangerous().with_custom_certificate_verifier`.
  #[derive(Debug)]
  pub struct PreparedServerVerifier {
    stock: WebPkiServerVerifier,
    roots: Arc<RootCertStore>,
    anchors: Anchors,
  }

  impl PreparedServerVerifier {
    /// Prepare `roots`' P-256, P-384 and ML-DSA anchor keys; other anchors import per use.
    ///
    /// # Errors
    ///
    /// Returns [`VerifierBuilderError::NoRootAnchors`] for an empty store, as
    /// the wrapped builder does.
    pub fn new(roots: Arc<RootCertStore>) -> Result<Self, VerifierBuilderError> {
      Self::with_algorithms(roots, super::SUPPORTED_ALGORITHMS)
    }

    /// Like [`Self::new`], with the TLS signature schemes of `supported`.
    ///
    /// `supported` must be built from [`ALL_ALGORITHMS`](super::ALL_ALGORITHMS),
    /// for example `WebPkiSupportedAlgorithms::new(ALL_ALGORITHMS, mapping)`.
    /// Its mapping selects the advertised and accepted handshake schemes;
    /// certificate chains always use `ALL_ALGORITHMS`.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn with_algorithms(
      roots: Arc<RootCertStore>,
      supported: WebPkiSupportedAlgorithms,
    ) -> Result<Self, VerifierBuilderError> {
      let stock = WebPkiServerVerifier::builder(Arc::clone(&roots), &verification_provider(supported)).build()?;
      let anchors = Anchors::new(&roots);
      Ok(Self { stock, roots, anchors })
    }

    /// Bytes retained by prepared anchor keys and their lookup keys, excluding map overhead.
    ///
    /// On 64-bit targets: 1,425 per P-256 anchor, 2,001 per P-384 anchor, and
    /// 21,864, 38,888 or 68,200 per ML-DSA-44, 65 or 87 anchor.
    #[must_use]
    pub const fn retained_anchor_bytes(&self) -> usize {
      self.anchors.retained_bytes()
    }
  }

  impl ServerVerifier for PreparedServerVerifier {
    fn verify_identity<'a>(&self, identity: &ServerIdentity<'a, '_>) -> Result<VerifiedIdentity<'a>, Error> {
      let Identity::X509(certificates) = identity.identity else {
        return self.stock.verify_identity(identity);
      };
      // The same public rustls checks the wrapped verifier runs without CRLs.
      let cert = ParsedCertificate::try_from(&certificates.end_entity)?;
      let algorithms = prepared_algorithms(&self.anchors);
      verify_identity_signed_by_trust_anchor(
        &cert,
        &self.roots,
        &certificates.intermediates,
        identity.now,
        &references(&algorithms),
      )?;
      verify_server_name(&cert, identity.server_name)?;
      Ok(VerifiedIdentity::assertion(identity.identity.clone()))
    }

    fn verify_tls12_signature(&self, input: &SignatureVerificationInput<'_>) -> Result<HandshakeSignatureValid, Error> {
      self.stock.verify_tls12_signature(input)
    }

    fn verify_tls13_signature(&self, input: &SignatureVerificationInput<'_>) -> Result<HandshakeSignatureValid, Error> {
      self.stock.verify_tls13_signature(input)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
      self.stock.supported_verify_schemes()
    }

    fn request_ocsp_response(&self) -> bool {
      self.stock.request_ocsp_response()
    }

    fn supported_certificate_types(&self) -> &'static [CertificateType] {
      self.stock.supported_certificate_types()
    }

    fn root_hint_subjects(&self) -> Option<Arc<[DistinguishedName]>> {
      self.stock.root_hint_subjects()
    }

    fn hash_config(&self, h: &mut dyn Hasher) {
      self.stock.hash_config(h);
    }
  }

  /// Client-certificate verifier, requiring authentication, with P-256, P-384
  /// and ML-DSA trust-anchor keys prepared once.
  ///
  /// Results, errors, root hints and signature schemes are those of
  /// [`WebPkiClientVerifier`] built from the same roots with
  /// [`SUPPORTED_ALGORITHMS`](super::SUPPORTED_ALGORITHMS), mandatory client
  /// authentication and no revocation lists. A rejected chain is re-verified by
  /// that verifier so its exact error is returned; that doubles only failure work.
  /// Install it with `ServerConfig::builder(..).with_client_cert_verifier`.
  #[derive(Debug)]
  pub struct PreparedClientVerifier {
    stock: WebPkiClientVerifier,
    roots: Arc<RootCertStore>,
    anchors: Anchors,
  }

  impl PreparedClientVerifier {
    /// Prepare `roots`' P-256, P-384 and ML-DSA anchor keys; other anchors import per use.
    ///
    /// # Errors
    ///
    /// Returns [`VerifierBuilderError::NoRootAnchors`] for an empty store, as
    /// the wrapped builder does.
    pub fn new(roots: Arc<RootCertStore>) -> Result<Self, VerifierBuilderError> {
      Self::with_algorithms(roots, super::SUPPORTED_ALGORITHMS)
    }

    /// Like [`Self::new`], with the TLS signature schemes of `supported`.
    ///
    /// `supported` must be built from [`ALL_ALGORITHMS`](super::ALL_ALGORITHMS),
    /// for example `WebPkiSupportedAlgorithms::new(ALL_ALGORITHMS, mapping)`.
    /// Its mapping selects the advertised and accepted handshake schemes;
    /// certificate chains always use `ALL_ALGORITHMS`.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn with_algorithms(
      roots: Arc<RootCertStore>,
      supported: WebPkiSupportedAlgorithms,
    ) -> Result<Self, VerifierBuilderError> {
      let stock = WebPkiClientVerifier::builder(Arc::clone(&roots), &verification_provider(supported)).build()?;
      let anchors = Anchors::new(&roots);
      Ok(Self { stock, roots, anchors })
    }

    /// Bytes retained by prepared anchor keys and their lookup keys, excluding map overhead.
    ///
    /// On 64-bit targets: 1,425 per P-256 anchor, 2,001 per P-384 anchor, and
    /// 21,864, 38,888 or 68,200 per ML-DSA-44, 65 or 87 anchor.
    #[must_use]
    pub const fn retained_anchor_bytes(&self) -> usize {
      self.anchors.retained_bytes()
    }
  }

  impl ClientVerifier for PreparedClientVerifier {
    fn verify_identity<'a>(&self, identity: &ClientIdentity<'a, '_>) -> Result<VerifiedIdentity<'a>, Error> {
      // The wrapped verifier's own checks: client-auth EKU, no revocation.
      if let Identity::X509(certificates) = identity.identity
        && let Ok(cert) = webpki::EndEntityCert::try_from(&certificates.end_entity)
      {
        let algorithms = prepared_algorithms(&self.anchors);
        let algorithms = references(&algorithms);
        if webpki::PathBuilder::new(
          &certificates.intermediates,
          None,
          &webpki::ExtendedKeyUsage::CLIENT_AUTH,
          &algorithms,
          &self.roots.roots,
        )
        .build(&cert, identity.now)
        .is_ok()
        {
          return Ok(VerifiedIdentity::assertion(identity.identity.clone()));
        }
      }
      // Rustls maps webpki errors privately; its verifier supplies the exact one.
      match self.stock.verify_identity(identity) {
        Err(error) => Err(error),
        Ok(_) => Err(Error::General("certificate verification failed".into())),
      }
    }

    fn verify_tls12_signature(&self, input: &SignatureVerificationInput<'_>) -> Result<HandshakeSignatureValid, Error> {
      self.stock.verify_tls12_signature(input)
    }

    fn verify_tls13_signature(&self, input: &SignatureVerificationInput<'_>) -> Result<HandshakeSignatureValid, Error> {
      self.stock.verify_tls13_signature(input)
    }

    fn root_hint_subjects(&self) -> Arc<[DistinguishedName]> {
      self.stock.root_hint_subjects()
    }

    fn client_auth_mandatory(&self) -> bool {
      self.stock.client_auth_mandatory()
    }

    fn offer_client_auth(&self) -> bool {
      self.stock.offer_client_auth()
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
      self.stock.supported_verify_schemes()
    }

    fn supported_certificate_types(&self) -> &'static [CertificateType] {
      self.stock.supported_certificate_types()
    }
  }

  fn references<'a>(algorithms: &'a [PreparedAlgorithm<'_>]) -> Vec<&'a dyn SignatureVerificationAlgorithm> {
    algorithms
      .iter()
      .map(|algorithm| algorithm as &dyn SignatureVerificationAlgorithm)
      .collect()
  }

  // The wrapped builders read only `signature_verification_algorithms`. The
  // other fields are inert so this provider can never negotiate or sign.
  fn verification_provider(supported: WebPkiSupportedAlgorithms) -> CryptoProvider {
    CryptoProvider {
      tls12_cipher_suites: Cow::Borrowed(&[]),
      tls13_cipher_suites: Cow::Borrowed(&[]),
      kx_groups: Cow::Borrowed(&[]),
      signature_verification_algorithms: supported,
      secure_random: &NoRandom,
      key_provider: crate::sign::KEY_PROVIDER,
      ticketer_factory: &NoTickets,
    }
  }

  #[derive(Debug)]
  struct NoRandom;

  impl rustls::crypto::SecureRandom for NoRandom {
    fn fill(&self, _: &mut [u8]) -> Result<(), GetRandomFailed> {
      Err(GetRandomFailed)
    }
  }

  #[derive(Debug)]
  struct NoTickets;

  impl rustls::crypto::TicketerFactory for NoTickets {
    fn ticketer(&self) -> Result<Arc<dyn rustls::crypto::TicketProducer>, Error> {
      Err(Error::General("verification-only provider".into()))
    }
  }
}

#[cfg(any(
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "rsa",
  feature = "ml-dsa"
))]
mod implementation {

  #[cfg(feature = "ml-dsa")]
  use alloc::alloc::Global;
  #[cfg(any(feature = "ecdsa-p256", feature = "ecdsa-p384", feature = "ml-dsa"))]
  use alloc::boxed::Box;
  use alloc::vec::Vec;
  #[cfg(feature = "ecdsa-p256")]
  use rscrypto::{EcdsaP256PublicKey, EcdsaP256Signature};
  #[cfg(feature = "ecdsa-p384")]
  use rscrypto::{EcdsaP384PublicKey, EcdsaP384Signature};
  #[cfg(feature = "ed25519")]
  use rscrypto::{Ed25519PublicKey, Ed25519Signature};
  #[cfg(feature = "ml-dsa")]
  use rscrypto::{
    MlDsa44PreparedPublicKey, MlDsa44PreparedPublicKeyStorage, MlDsa65PreparedPublicKey,
    MlDsa65PreparedPublicKeyStorage, MlDsa87PreparedPublicKey, MlDsa87PreparedPublicKeyStorage,
  };
  #[cfg(feature = "rsa")]
  use rscrypto::{RsaPublicKey, RsaPublicKeyPolicy, RsaSignatureProfile};
  use rustls::RootCertStore;
  use rustls::crypto::SignatureScheme;
  use rustls::crypto::WebPkiSupportedAlgorithms;
  use rustls::pki_types::{AlgorithmIdentifier, InvalidSignature, SignatureVerificationAlgorithm, alg_id};
  #[cfg(any(feature = "ecdsa-p256", feature = "ecdsa-p384", feature = "ml-dsa"))]
  use std::collections::HashMap;
  #[derive(Clone, Copy, Debug)]
  enum Operation {
    #[cfg(feature = "ecdsa-p256")]
    P256Sha256,
    #[cfg(feature = "ecdsa-p256")]
    P256Sha384,
    #[cfg(feature = "ecdsa-p384")]
    P384Sha256,
    #[cfg(feature = "ecdsa-p384")]
    P384Sha384,
    #[cfg(feature = "ed25519")]
    Ed25519,
    #[cfg(feature = "rsa")]
    Rsa(SignatureScheme),
    #[cfg(feature = "ml-dsa")]
    MlDsa(SignatureScheme),
  }

  #[derive(Clone, Copy, Debug)]
  struct Algorithm {
    public_key: AlgorithmIdentifier,
    signature: AlgorithmIdentifier,
    operation: Operation,
  }

  impl SignatureVerificationAlgorithm for Algorithm {
    fn public_key_alg_id(&self) -> AlgorithmIdentifier {
      self.public_key
    }

    fn signature_alg_id(&self) -> AlgorithmIdentifier {
      self.signature
    }

    fn verify_signature(&self, key: &[u8], message: &[u8], signature: &[u8]) -> Result<(), InvalidSignature> {
      verify(self.operation, None, key, message, signature)
    }
  }

  /// Verify with the key prepared from exactly `key`'s bytes when `anchors`
  /// holds one; otherwise import `key` for this signature. Both paths reach the
  /// same decision.
  fn verify(
    operation: Operation,
    anchors: Option<&Anchors>,
    key: &[u8],
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), InvalidSignature> {
    let _ = anchors;
    match operation {
      #[cfg(feature = "ecdsa-p256")]
      Operation::P256Sha256 | Operation::P256Sha384 => {
        let signature = EcdsaP256Signature::from_der(signature).map_err(|_| InvalidSignature)?;
        match (anchors.and_then(|anchors| anchors.p256.get(key)), operation) {
          (Some(prepared), Operation::P256Sha256) => prepared.verify(message, &signature),
          (Some(prepared), _) => prepared.verify_sha384(message, &signature),
          // A key that is not a prepared anchor verifies once, so skip its table.
          (None, Operation::P256Sha256) => EcdsaP256PublicKey::verify_sec1(key, message, &signature),
          (None, _) => EcdsaP256PublicKey::verify_sec1_sha384(key, message, &signature),
        }
      }
      #[cfg(feature = "ecdsa-p384")]
      Operation::P384Sha256 | Operation::P384Sha384 => {
        let signature = EcdsaP384Signature::from_der(signature).map_err(|_| InvalidSignature)?;
        match (anchors.and_then(|anchors| anchors.p384.get(key)), operation) {
          (Some(prepared), Operation::P384Sha384) => prepared.verify(message, &signature),
          (Some(prepared), _) => prepared.verify_sha256(message, &signature),
          // A key that is not a prepared anchor verifies once, so skip its table.
          (None, Operation::P384Sha384) => EcdsaP384PublicKey::verify_sec1(key, message, &signature),
          (None, _) => EcdsaP384PublicKey::verify_sec1_sha256(key, message, &signature),
        }
      }
      #[cfg(feature = "ed25519")]
      Operation::Ed25519 => {
        let signature = Ed25519Signature::from_bytes(signature.try_into().map_err(|_| InvalidSignature)?);
        let key = Ed25519PublicKey::from_bytes(key.try_into().map_err(|_| InvalidSignature)?);
        key.verify(message, &signature)
      }
      #[cfg(feature = "ml-dsa")]
      Operation::MlDsa(scheme) => {
        macro_rules! verify {
          ($public:ident, $signature:ident, $prepared:ident) => {{
            let signature = rscrypto::$signature::try_from_slice(signature).map_err(|_| InvalidSignature)?;
            match anchors.and_then(|anchors| anchors.$prepared.get(key)) {
              Some(prepared) => prepared.verify(message, &signature),
              None => rscrypto::$public::try_from_slice(key)
                .map_err(|_| InvalidSignature)?
                .verify(message, &signature),
            }
          }};
        }
        match scheme {
          SignatureScheme::ML_DSA_44 => verify!(MlDsa44PublicKey, MlDsa44Signature, ml_dsa_44),
          SignatureScheme::ML_DSA_65 => verify!(MlDsa65PublicKey, MlDsa65Signature, ml_dsa_65),
          _ => verify!(MlDsa87PublicKey, MlDsa87Signature, ml_dsa_87),
        }
      }
      #[cfg(feature = "rsa")]
      Operation::Rsa(scheme) => {
        // Preserve the deployed WebPKI policy, independently of the stricter
        // policy for new TLS signing keys. Bound the public exponent's work.
        let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_odd_exponents();
        let signature_lengths = policy.min_modulus_bits().div_ceil(8)..=policy.max_modulus_bits().div_ceil(8);
        if !signature_lengths.contains(&signature.len()) {
          return Err(InvalidSignature);
        }
        let key = RsaPublicKey::from_pkcs1_der_with_policy(key, &policy).map_err(|_| InvalidSignature)?;
        if key.public_exponent().as_u64() >= 1u64 << 33 {
          return Err(InvalidSignature);
        }
        let profile = RsaSignatureProfile::from_tls_certificate_signature_scheme(u16::from_be_bytes(scheme.to_array()))
          .map_err(|_| InvalidSignature)?;
        key.verify_signature(profile, message, signature)
      }
    }
    .map_err(|_| InvalidSignature)
  }

  /// Verification keys prepared once from operator-configured trust anchors.
  ///
  /// Each map is keyed by the anchor's exact subjectPublicKey bytes, which
  /// webpki passes back unchanged when that anchor issues a certificate.
  /// Peer-supplied keys are never inserted. ECDSA import builds a per-key
  /// table, and ML-DSA preparation expands the matrix and transformed key;
  /// RSA and Ed25519 anchors keep the import path.
  #[derive(Debug, Default)]
  pub(super) struct Anchors {
    #[cfg(feature = "ecdsa-p256")]
    p256: HashMap<Box<[u8]>, EcdsaP256PublicKey>,
    #[cfg(feature = "ecdsa-p384")]
    p384: HashMap<Box<[u8]>, EcdsaP384PublicKey>,
    #[cfg(feature = "ml-dsa")]
    ml_dsa_44: HashMap<Box<[u8]>, MlDsa44PreparedPublicKey<Box<MlDsa44PreparedPublicKeyStorage>>>,
    #[cfg(feature = "ml-dsa")]
    ml_dsa_65: HashMap<Box<[u8]>, MlDsa65PreparedPublicKey<Box<MlDsa65PreparedPublicKeyStorage>>>,
    #[cfg(feature = "ml-dsa")]
    ml_dsa_87: HashMap<Box<[u8]>, MlDsa87PreparedPublicKey<Box<MlDsa87PreparedPublicKeyStorage>>>,
    retained: usize,
  }

  impl Anchors {
    pub(super) fn new(roots: &RootCertStore) -> Self {
      let mut anchors = Self::default();
      for anchor in &roots.roots {
        // Anchors that do not parse or import stay on the import path, where
        // webpki and the algorithm reject or accept them as before.
        if let Some((algorithm, key)) = spki_parts(anchor.subject_public_key_info.as_ref()) {
          anchors.insert(algorithm, key);
        }
      }
      anchors
    }

    fn insert(&mut self, algorithm: &[u8], key: &[u8]) {
      let _ = (algorithm, key);
      #[cfg(feature = "ecdsa-p256")]
      if algorithm == alg_id::ECDSA_P256.as_ref()
        && let Ok(prepared) = EcdsaP256PublicKey::from_sec1_bytes(key)
        && self.p256.insert(key.into(), prepared).is_none()
      {
        self.retained = self
          .retained
          .strict_add(key.len())
          .strict_add(size_of::<EcdsaP256PublicKey>());
      }
      #[cfg(feature = "ecdsa-p384")]
      if algorithm == alg_id::ECDSA_P384.as_ref()
        && let Ok(prepared) = EcdsaP384PublicKey::from_sec1_bytes(key)
        && self.p384.insert(key.into(), prepared).is_none()
      {
        self.retained = self
          .retained
          .strict_add(key.len())
          .strict_add(size_of::<EcdsaP384PublicKey>());
      }
      #[cfg(feature = "ml-dsa")]
      {
        macro_rules! prepare {
          ($algorithm:ident, $map:ident, $public:ident, $storage:ident) => {
            if algorithm == alg_id::$algorithm.as_ref()
              && let Ok(public) = rscrypto::$public::try_from_slice(key)
              && let Ok(prepared) = public.prepare_in(Global)
              && self.$map.insert(key.into(), prepared).is_none()
            {
              self.retained = self
                .retained
                .strict_add(key.len())
                .strict_add(size_of::<Box<$storage>>())
                .strict_add(size_of::<$storage>());
            }
          };
        }
        prepare!(ML_DSA_44, ml_dsa_44, MlDsa44PublicKey, MlDsa44PreparedPublicKeyStorage);
        prepare!(ML_DSA_65, ml_dsa_65, MlDsa65PublicKey, MlDsa65PreparedPublicKeyStorage);
        prepare!(ML_DSA_87, ml_dsa_87, MlDsa87PublicKey, MlDsa87PreparedPublicKeyStorage);
      }
    }

    /// Bytes retained by prepared keys and their lookup keys, excluding map overhead.
    pub(super) const fn retained_bytes(&self) -> usize {
      self.retained
    }
  }

  /// Split DER SubjectPublicKeyInfo contents into the AlgorithmIdentifier
  /// contents and the BIT STRING key bytes, as webpki does for trust anchors.
  fn spki_parts(spki: &[u8]) -> Option<(&[u8], &[u8])> {
    fn element(input: &[u8], tag: u8) -> Option<(&[u8], &[u8])> {
      let (&found, rest) = input.split_first()?;
      let (&first, rest) = rest.split_first()?;
      let (length, rest) = match first {
        0x00..=0x7f => (usize::from(first), rest),
        0x81 => {
          let (&length, rest) = rest.split_first()?;
          (length >= 0x80).then_some((usize::from(length), rest))?
        }
        0x82 => {
          let (length, rest) = rest.split_first_chunk::<2>()?;
          let length = usize::from(u16::from_be_bytes(*length));
          (length >= 0x100).then_some((length, rest))?
        }
        _ => return None,
      };
      (found == tag && rest.len() >= length).then(|| rest.split_at(length))
    }
    let (algorithm, rest) = element(spki, 0x30)?;
    let (key, rest) = element(rest, 0x03)?;
    let (&unused_bits, key) = key.split_first()?;
    (rest.is_empty() && unused_bits == 0).then_some((algorithm, key))
  }

  /// One of this crate's algorithms, consulting prepared anchors before import.
  #[derive(Debug)]
  pub(super) struct PreparedAlgorithm<'a> {
    algorithm: &'static Algorithm,
    anchors: &'a Anchors,
  }

  impl SignatureVerificationAlgorithm for PreparedAlgorithm<'_> {
    fn public_key_alg_id(&self) -> AlgorithmIdentifier {
      self.algorithm.public_key
    }

    fn signature_alg_id(&self) -> AlgorithmIdentifier {
      self.algorithm.signature
    }

    fn verify_signature(&self, key: &[u8], message: &[u8], signature: &[u8]) -> Result<(), InvalidSignature> {
      verify(self.algorithm.operation, Some(self.anchors), key, message, signature)
    }
  }

  /// This crate's certificate algorithms, in [`ALL`]'s order, over `anchors`.
  pub(super) fn prepared_algorithms(anchors: &Anchors) -> Vec<PreparedAlgorithm<'_>> {
    CONCRETE
      .iter()
      .map(|&algorithm| PreparedAlgorithm { algorithm, anchors })
      .collect()
  }

  #[cfg(feature = "ecdsa-p256")]
  static P256_SHA256: Algorithm = Algorithm {
    public_key: alg_id::ECDSA_P256,
    signature: alg_id::ECDSA_SHA256,
    operation: Operation::P256Sha256,
  };
  #[cfg(feature = "ecdsa-p256")]
  static P256_SHA384: Algorithm = Algorithm {
    public_key: alg_id::ECDSA_P256,
    signature: alg_id::ECDSA_SHA384,
    operation: Operation::P256Sha384,
  };
  #[cfg(feature = "ecdsa-p384")]
  static P384_SHA256: Algorithm = Algorithm {
    public_key: alg_id::ECDSA_P384,
    signature: alg_id::ECDSA_SHA256,
    operation: Operation::P384Sha256,
  };
  #[cfg(feature = "ecdsa-p384")]
  static P384_SHA384: Algorithm = Algorithm {
    public_key: alg_id::ECDSA_P384,
    signature: alg_id::ECDSA_SHA384,
    operation: Operation::P384Sha384,
  };
  #[cfg(feature = "ed25519")]
  static ED25519: Algorithm = Algorithm {
    public_key: alg_id::ED25519,
    signature: alg_id::ED25519,
    operation: Operation::Ed25519,
  };
  #[cfg(feature = "rsa")]
  static RSA_PSS_SHA256: Algorithm = Algorithm {
    public_key: alg_id::RSA_ENCRYPTION,
    signature: alg_id::RSA_PSS_SHA256,
    operation: Operation::Rsa(SignatureScheme::RSA_PSS_SHA256),
  };
  #[cfg(feature = "rsa")]
  static RSA_PSS_SHA384: Algorithm = Algorithm {
    public_key: alg_id::RSA_ENCRYPTION,
    signature: alg_id::RSA_PSS_SHA384,
    operation: Operation::Rsa(SignatureScheme::RSA_PSS_SHA384),
  };
  #[cfg(feature = "rsa")]
  static RSA_PSS_SHA512: Algorithm = Algorithm {
    public_key: alg_id::RSA_ENCRYPTION,
    signature: alg_id::RSA_PSS_SHA512,
    operation: Operation::Rsa(SignatureScheme::RSA_PSS_SHA512),
  };
  #[cfg(feature = "rsa")]
  static RSA_PKCS1_SHA256: Algorithm = Algorithm {
    public_key: alg_id::RSA_ENCRYPTION,
    signature: alg_id::RSA_PKCS1_SHA256,
    operation: Operation::Rsa(SignatureScheme::RSA_PKCS1_SHA256),
  };
  #[cfg(feature = "rsa")]
  static RSA_PKCS1_SHA384: Algorithm = Algorithm {
    public_key: alg_id::RSA_ENCRYPTION,
    signature: alg_id::RSA_PKCS1_SHA384,
    operation: Operation::Rsa(SignatureScheme::RSA_PKCS1_SHA384),
  };
  #[cfg(feature = "rsa")]
  static RSA_PKCS1_SHA512: Algorithm = Algorithm {
    public_key: alg_id::RSA_ENCRYPTION,
    signature: alg_id::RSA_PKCS1_SHA512,
    operation: Operation::Rsa(SignatureScheme::RSA_PKCS1_SHA512),
  };

  // RFC 4055 requires verifiers to also accept absent PKCS#1 signature parameters.
  #[cfg(feature = "rsa")]
  static RSA_PKCS1_SHA256_ABSENT_PARAMS: Algorithm = Algorithm {
    signature: AlgorithmIdentifier::from_slice(&[0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b]),
    ..RSA_PKCS1_SHA256
  };
  #[cfg(feature = "rsa")]
  static RSA_PKCS1_SHA384_ABSENT_PARAMS: Algorithm = Algorithm {
    signature: AlgorithmIdentifier::from_slice(&[0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0c]),
    ..RSA_PKCS1_SHA384
  };
  #[cfg(feature = "rsa")]
  static RSA_PKCS1_SHA512_ABSENT_PARAMS: Algorithm = Algorithm {
    signature: AlgorithmIdentifier::from_slice(&[0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0d]),
    ..RSA_PKCS1_SHA512
  };

  #[cfg(feature = "ml-dsa")]
  static ML_DSA_44: Algorithm = Algorithm {
    public_key: alg_id::ML_DSA_44,
    signature: alg_id::ML_DSA_44,
    operation: Operation::MlDsa(SignatureScheme::ML_DSA_44),
  };
  #[cfg(feature = "ml-dsa")]
  static ML_DSA_65: Algorithm = Algorithm {
    public_key: alg_id::ML_DSA_65,
    signature: alg_id::ML_DSA_65,
    operation: Operation::MlDsa(SignatureScheme::ML_DSA_65),
  };
  #[cfg(feature = "ml-dsa")]
  static ML_DSA_87: Algorithm = Algorithm {
    public_key: alg_id::ML_DSA_87,
    signature: alg_id::ML_DSA_87,
    operation: Operation::MlDsa(SignatureScheme::ML_DSA_87),
  };

  // One list defines both the certificate algorithms and their concrete
  // operations, so the prepared verifiers cover exactly the same set.
  macro_rules! algorithms {
    ($($(#[$cfg:meta])* $name:ident,)*) => {
      pub(super) const ALL: &[&dyn SignatureVerificationAlgorithm] = &[$($(#[$cfg])* &$name,)*];
      const CONCRETE: &[&Algorithm] = &[$($(#[$cfg])* &$name,)*];
    };
  }
  algorithms!(
    #[cfg(feature = "ml-dsa")]
    ML_DSA_44,
    #[cfg(feature = "ml-dsa")]
    ML_DSA_65,
    #[cfg(feature = "ml-dsa")]
    ML_DSA_87,
    #[cfg(feature = "ecdsa-p256")]
    P256_SHA256,
    #[cfg(feature = "ecdsa-p256")]
    P256_SHA384,
    #[cfg(feature = "ecdsa-p384")]
    P384_SHA256,
    #[cfg(feature = "ecdsa-p384")]
    P384_SHA384,
    #[cfg(feature = "ed25519")]
    ED25519,
    #[cfg(feature = "rsa")]
    RSA_PSS_SHA256,
    #[cfg(feature = "rsa")]
    RSA_PSS_SHA384,
    #[cfg(feature = "rsa")]
    RSA_PSS_SHA512,
    #[cfg(feature = "rsa")]
    RSA_PKCS1_SHA256,
    #[cfg(feature = "rsa")]
    RSA_PKCS1_SHA384,
    #[cfg(feature = "rsa")]
    RSA_PKCS1_SHA512,
    #[cfg(feature = "rsa")]
    RSA_PKCS1_SHA256_ABSENT_PARAMS,
    #[cfg(feature = "rsa")]
    RSA_PKCS1_SHA384_ABSENT_PARAMS,
    #[cfg(feature = "rsa")]
    RSA_PKCS1_SHA512_ABSENT_PARAMS,
  );
  pub(super) const ALGORITHMS: WebPkiSupportedAlgorithms = {
    const MAPPING: &[(SignatureScheme, &[&dyn SignatureVerificationAlgorithm])] = &[
      #[cfg(feature = "ml-dsa")]
      (SignatureScheme::ML_DSA_87, &[&ML_DSA_87]),
      #[cfg(feature = "ml-dsa")]
      (SignatureScheme::ML_DSA_65, &[&ML_DSA_65]),
      #[cfg(feature = "ml-dsa")]
      (SignatureScheme::ML_DSA_44, &[&ML_DSA_44]),
      // TLS 1.3 binds the curve to its hash. Certificate paths use `all`,
      // including the cross-hash combinations, independently of this mapping.
      #[cfg(feature = "ecdsa-p384")]
      (SignatureScheme::ECDSA_NISTP384_SHA384, &[&P384_SHA384]),
      #[cfg(feature = "ecdsa-p256")]
      (SignatureScheme::ECDSA_NISTP256_SHA256, &[&P256_SHA256]),
      #[cfg(feature = "ed25519")]
      (SignatureScheme::ED25519, &[&ED25519]),
      #[cfg(feature = "rsa")]
      (SignatureScheme::RSA_PSS_SHA512, &[&RSA_PSS_SHA512]),
      #[cfg(feature = "rsa")]
      (SignatureScheme::RSA_PSS_SHA384, &[&RSA_PSS_SHA384]),
      #[cfg(feature = "rsa")]
      (SignatureScheme::RSA_PSS_SHA256, &[&RSA_PSS_SHA256]),
      #[cfg(feature = "rsa")]
      (SignatureScheme::RSA_PKCS1_SHA512, &[&RSA_PKCS1_SHA512]),
      #[cfg(feature = "rsa")]
      (SignatureScheme::RSA_PKCS1_SHA384, &[&RSA_PKCS1_SHA384]),
      #[cfg(feature = "rsa")]
      (SignatureScheme::RSA_PKCS1_SHA256, &[&RSA_PKCS1_SHA256]),
    ];
    match WebPkiSupportedAlgorithms::new(ALL, MAPPING) {
      Ok(algorithms) => algorithms,
      Err(_) => panic!("enabled signature mappings are nonempty"),
    }
  };
}
