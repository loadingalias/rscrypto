//! TLS 1.3 provider integration for rustls, backed by rscrypto.
//!
//! The provider implements TLS 1.3 cryptography, including hybrid key exchange.
//! The crate requires `std`; its algorithm features forward to rscrypto's leaf features.

#![forbid(unsafe_code)]

extern crate alloc;

#[cfg(any(feature = "aes-gcm", feature = "chacha20poly1305"))]
pub mod aead;
pub mod cipher_suite;
#[cfg(any(
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "rsa",
  feature = "ml-dsa"
))]
mod client;
#[cfg(any(feature = "sha2", feature = "hmac", feature = "hkdf"))]
pub mod hash;
pub mod kx_group;
#[cfg(any(feature = "aes-gcm", feature = "chacha20poly1305"))]
pub mod quic;
#[cfg(feature = "quinn")]
pub mod quinn;
#[cfg(feature = "getrandom")]
pub mod random;
pub mod sign;
#[cfg(all(feature = "aes-gcm", feature = "getrandom"))]
pub mod ticket;
pub mod verify;

#[cfg(any(
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "rsa",
  feature = "ml-dsa"
))]
pub use client::ClientProvider;

/// TLS 1.3 cryptography with hybrid key exchange preferred and classical fallback.
///
/// The enabled Cargo features determine the suite, group, and signature lists.
/// Pass this provider explicitly to rustls's configuration builder. This
/// constructor requires `getrandom` and at least one signature-verification
/// feature; rustls rejects an empty signature-verification catalog. For a client
/// over caller entropy, without `getrandom`, use [`ClientProvider`].
///
/// ```
/// # #[cfg(feature = "default")]
/// # {
/// use std::sync::Arc;
/// let provider = Arc::new(rscrypto_rustls::default_provider());
/// let config = rustls::ClientConfig::builder(provider)
///     .with_root_certificates(rustls::RootCertStore::empty())
///     .with_no_client_auth()?;
/// // Populate the trust store before connecting to a peer.
/// # let _ = config;
/// # }
/// # Ok::<(), rustls::Error>(())
/// ```
#[cfg(all(
  feature = "getrandom",
  any(
    feature = "ecdsa-p256",
    feature = "ecdsa-p384",
    feature = "ed25519",
    feature = "rsa",
    feature = "ml-dsa"
  )
))]
#[must_use]
pub fn default_provider() -> rustls::crypto::CryptoProvider {
  rustls::crypto::CryptoProvider {
    tls12_cipher_suites: alloc::borrow::Cow::Borrowed(&[]),
    tls13_cipher_suites: alloc::borrow::Cow::Borrowed(cipher_suite::ALL_CIPHER_SUITES),
    kx_groups: alloc::borrow::Cow::Borrowed(kx_group::ALL_KX_GROUPS),
    signature_verification_algorithms: verify::SUPPORTED_ALGORITHMS,
    secure_random: random::SECURE_RANDOM,
    key_provider: sign::KEY_PROVIDER,
    ticketer_factory: &TicketerFactory,
  }
}

/// TLS 1.3 cryptography requiring hybrid key establishment.
///
/// Classical-only peers fail negotiation. This selects key exchange; it does
/// not require a post-quantum certificate chain or add fresh entropy at KeyUpdate.
#[cfg(all(
  feature = "getrandom",
  feature = "x25519",
  feature = "ml-kem",
  any(
    feature = "ecdsa-p256",
    feature = "ecdsa-p384",
    feature = "ed25519",
    feature = "rsa",
    feature = "ml-dsa"
  )
))]
#[must_use]
pub fn post_quantum_required_provider() -> rustls::crypto::CryptoProvider {
  let mut provider = default_provider();
  provider
    .kx_groups
    .to_mut()
    .retain(|group| group.name() == rustls::crypto::kx::NamedGroup::X25519MLKEM768);
  provider
}

#[cfg(all(
  feature = "getrandom",
  any(
    feature = "ecdsa-p256",
    feature = "ecdsa-p384",
    feature = "ed25519",
    feature = "rsa",
    feature = "ml-dsa"
  )
))]
#[derive(Debug)]
struct TicketerFactory;

#[cfg(all(
  feature = "getrandom",
  any(
    feature = "ecdsa-p256",
    feature = "ecdsa-p384",
    feature = "ed25519",
    feature = "rsa",
    feature = "ml-dsa"
  )
))]
impl rustls::crypto::TicketerFactory for TicketerFactory {
  fn ticketer(&self) -> Result<alloc::sync::Arc<dyn rustls::crypto::TicketProducer>, rustls::Error> {
    #[cfg(feature = "aes-gcm")]
    {
      ticket::ticketer()
    }
    #[cfg(not(feature = "aes-gcm"))]
    {
      Err(rustls::Error::General(
        "stateless tickets require the aes-gcm feature".into(),
      ))
    }
  }
}

#[cfg(test)]
mod tests;
