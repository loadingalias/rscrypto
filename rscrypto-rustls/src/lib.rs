//! TLS 1.3 provider integration for rustls, backed by rscrypto.
//!
//! The provider implements TLS 1.3 cryptography, including hybrid key exchange.
//! The crate requires `std`; its algorithm features forward to rscrypto's leaf features.

#![forbid(unsafe_code)]

extern crate alloc;

#[cfg(any(feature = "aes-gcm", feature = "chacha20poly1305"))]
pub mod aead;
pub mod cipher_suite;
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

/// TLS 1.3 cryptography with hybrid key exchange preferred and classical fallback.
///
/// The enabled Cargo features determine the suite, group, and signature lists.
/// Pass this provider explicitly to rustls's `builder_with_provider`.
///
/// ```
/// # #[cfg(feature = "default")]
/// # {
/// use std::sync::Arc;
/// let provider = Arc::new(rscrypto_rustls::default_provider());
/// let config = rustls::ClientConfig::builder_with_provider(provider)
///     .with_protocol_versions(&[&rustls::version::TLS13])?
///     .with_root_certificates(rustls::RootCertStore::empty())
///     .with_no_client_auth();
/// // Populate the trust store before connecting to a peer.
/// # let _ = config;
/// # }
/// # Ok::<(), rustls::Error>(())
/// ```
#[cfg(feature = "getrandom")]
#[must_use]
pub fn default_provider() -> rustls::crypto::CryptoProvider {
  rustls::crypto::CryptoProvider {
    cipher_suites: cipher_suite::ALL_CIPHER_SUITES.to_vec(),
    kx_groups: kx_group::ALL_KX_GROUPS.to_vec(),
    signature_verification_algorithms: verify::SUPPORTED_ALGORITHMS,
    secure_random: random::SECURE_RANDOM,
    key_provider: sign::KEY_PROVIDER,
  }
}

/// TLS 1.3 cryptography requiring hybrid key establishment.
///
/// Classical-only peers fail negotiation. This selects key exchange; it does
/// not require a post-quantum certificate chain or add fresh entropy at KeyUpdate.
#[cfg(all(feature = "getrandom", feature = "x25519", feature = "ml-kem"))]
#[must_use]
pub fn post_quantum_required_provider() -> rustls::crypto::CryptoProvider {
  let mut provider = default_provider();
  provider
    .kx_groups
    .retain(|group| group.name() == rustls::NamedGroup::X25519MLKEM768);
  provider
}

#[cfg(test)]
mod tests;
