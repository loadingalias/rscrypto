//! TLS 1.3 client cryptography over a caller-owned entropy source.

use alloc::{borrow::Cow, boxed::Box, sync::Arc};

use rustls::Error;
use rustls::crypto::kx::SupportedKxGroup;
use rustls::crypto::{CryptoProvider, KeyProvider, SecureRandom, SigningKey, TicketProducer, TicketerFactory};
use rustls::pki_types::PrivateKeyDer;

use crate::{cipher_suite, verify};

/// TLS 1.3 client cryptography over a caller-owned entropy source.
///
/// [`Self::provider`] offers the enabled cipher suites, the enabled key
/// exchange groups (hybrid first, then X25519, P-256, and P-384), and the
/// certificate algorithms of [`verify::SUPPORTED_ALGORITHMS`]. Every entropy
/// request goes to the caller's source, and its failures propagate without
/// fallback. The provider loads no private keys and builds no ticketers, so it
/// supports neither client authentication nor serving. It does not need `getrandom`.
/// It offers TLS 1.3 only; to enable TLS 1.2, set the returned provider's
/// `tls12_cipher_suites` to `cipher_suite::ALL_TLS12_CIPHER_SUITES`.
/// For client certificates, set its `key_provider` to a static
/// [`crate::sign::Provider`] over the same source.
///
/// The groups live in this value, so build it in a `static`:
///
/// ```
/// # #[cfg(all(feature = "hkdf", feature = "aes-gcm", feature = "x25519", feature = "ecdsa-p256"))]
/// # {
/// use std::sync::Arc;
/// use rustls::crypto::{GetRandomFailed, SecureRandom};
///
/// #[derive(Debug)]
/// struct Entropy;
///
/// impl SecureRandom for Entropy {
///   fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
///     // Fill `output` from the operating system; this placeholder refuses.
///     let _ = output;
///     Err(GetRandomFailed)
///   }
/// }
///
/// static ENTROPY: Entropy = Entropy;
/// static CLIENT: rscrypto_rustls::ClientProvider = rscrypto_rustls::ClientProvider::new(&ENTROPY);
///
/// let config = rustls::ClientConfig::builder(Arc::new(CLIENT.provider()))
///   .with_root_certificates(rustls::RootCertStore::empty())
///   .with_no_client_auth()?;
/// # let _ = config;
/// # }
/// # Ok::<(), rustls::Error>(())
/// ```
#[derive(Debug)]
pub struct ClientProvider {
  random: &'static dyn SecureRandom,
  #[cfg(all(feature = "x25519", feature = "ml-kem"))]
  hybrid: crate::kx_group::X25519MlKem768Group,
  #[cfg(feature = "x25519")]
  x25519: crate::kx_group::X25519Group,
  #[cfg(feature = "p256-ecdh")]
  p256: crate::kx_group::P256Group,
  #[cfg(feature = "p384-ecdh")]
  p384: crate::kx_group::P384Group,
}

impl ClientProvider {
  /// Use `random` for every entropy request of the client provider.
  ///
  /// The source must provide cryptographically secure random bytes.
  #[must_use]
  pub const fn new(random: &'static dyn SecureRandom) -> Self {
    Self {
      random,
      #[cfg(all(feature = "x25519", feature = "ml-kem"))]
      hybrid: crate::kx_group::X25519MlKem768Group::new(random),
      #[cfg(feature = "x25519")]
      x25519: crate::kx_group::X25519Group::new(random),
      #[cfg(feature = "p256-ecdh")]
      p256: crate::kx_group::P256Group::new(random),
      #[cfg(feature = "p384-ecdh")]
      p384: crate::kx_group::P384Group::new(random),
    }
  }

  /// The client provider; pass it to rustls's `ClientConfig` builder.
  #[must_use]
  pub fn provider(&'static self) -> CryptoProvider {
    let kx_groups: &[&'static dyn SupportedKxGroup] = &[
      #[cfg(all(feature = "x25519", feature = "ml-kem"))]
      &self.hybrid,
      #[cfg(feature = "x25519")]
      &self.x25519,
      #[cfg(feature = "p256-ecdh")]
      &self.p256,
      #[cfg(feature = "p384-ecdh")]
      &self.p384,
    ];
    CryptoProvider {
      tls12_cipher_suites: Cow::Borrowed(&[]),
      tls13_cipher_suites: Cow::Borrowed(cipher_suite::ALL_CIPHER_SUITES),
      kx_groups: Cow::Owned(kx_groups.to_vec()),
      signature_verification_algorithms: verify::SUPPORTED_ALGORITHMS,
      secure_random: self.random,
      key_provider: &NoKeys,
      ticketer_factory: &NoTickets,
    }
  }
}

/// Refuses every private key and clears the owned DER.
#[derive(Debug)]
pub(crate) struct NoKeys;

impl KeyProvider for NoKeys {
  fn load_private_key(&self, key_der: PrivateKeyDer<'static>) -> Result<Box<dyn SigningKey>, Error> {
    // pki-types does not clear owned DER on drop.
    drop(zeroize::Zeroizing::new(key_der));
    Err(Error::General("this provider loads no private keys".into()))
  }
}

/// Refuses to build ticketers.
#[derive(Debug)]
pub(crate) struct NoTickets;

impl TicketerFactory for NoTickets {
  fn ticketer(&self) -> Result<Arc<dyn TicketProducer>, Error> {
    Err(Error::General("this provider builds no ticketers".into()))
  }
}
