//! TLS 1.3 and TLS 1.2 cipher suites.
//!
//! Each TLS 1.3 suite includes QUIC packet and header protection. The TLS 1.2
//! suites are ECDHE with an AEAD only (no CBC or RSA key transport), are TCP
//! only, and are opt-in: add [`ALL_TLS12_CIPHER_SUITES`] to a provider's
//! `tls12_cipher_suites`.

#[cfg(all(
  any(feature = "hmac", feature = "hkdf"),
  any(feature = "aes-gcm", feature = "chacha20poly1305")
))]
use rustls::Tls12CipherSuite;
use rustls::Tls13CipherSuite;
#[cfg(all(
  any(feature = "hmac", feature = "hkdf"),
  any(feature = "aes-gcm", feature = "chacha20poly1305")
))]
use rustls::crypto::SignatureScheme;
#[cfg(all(
  any(feature = "hmac", feature = "hkdf"),
  any(feature = "aes-gcm", feature = "chacha20poly1305")
))]
use rustls::crypto::kx::KeyExchangeAlgorithm;
#[cfg(all(
  any(feature = "hmac", feature = "hkdf"),
  any(feature = "aes-gcm", feature = "chacha20poly1305")
))]
use rustls::crypto::{CipherSuite, CipherSuiteCommon};

/// TLS_AES_128_GCM_SHA256, including AES-128 QUIC header protection.
#[cfg(all(feature = "hkdf", feature = "aes-gcm"))]
pub static TLS13_AES_128_GCM_SHA256: &Tls13CipherSuite = &Tls13CipherSuite {
  common: CipherSuiteCommon {
    suite: CipherSuite::TLS13_AES_128_GCM_SHA256,
    hash_provider: crate::hash::SHA256,
    // RFC 9846 §5.5: conservative bound for full-sized TLS records.
    confidentiality_limit: 1 << 24,
  },
  protocol_version: rustls::version::TLS13_VERSION,
  hkdf_provider: crate::hash::HKDF_SHA256,
  aead_alg: crate::aead::AES_128_GCM,
  quic: Some(crate::quic::AES_128_GCM),
};

/// TLS_AES_256_GCM_SHA384, including AES-256 QUIC header protection.
#[cfg(all(feature = "hkdf", feature = "aes-gcm"))]
pub static TLS13_AES_256_GCM_SHA384: &Tls13CipherSuite = &Tls13CipherSuite {
  common: CipherSuiteCommon {
    suite: CipherSuite::TLS13_AES_256_GCM_SHA384,
    hash_provider: crate::hash::SHA384,
    confidentiality_limit: 1 << 24,
  },
  protocol_version: rustls::version::TLS13_VERSION,
  hkdf_provider: crate::hash::HKDF_SHA384,
  aead_alg: crate::aead::AES_256_GCM,
  quic: Some(crate::quic::AES_256_GCM),
};

/// TLS_CHACHA20_POLY1305_SHA256, including ChaCha20 QUIC header protection.
#[cfg(all(feature = "hkdf", feature = "chacha20poly1305"))]
pub static TLS13_CHACHA20_POLY1305_SHA256: &Tls13CipherSuite = &Tls13CipherSuite {
  common: CipherSuiteCommon {
    suite: CipherSuite::TLS13_CHACHA20_POLY1305_SHA256,
    hash_provider: crate::hash::SHA256,
    confidentiality_limit: u64::MAX,
  },
  protocol_version: rustls::version::TLS13_VERSION,
  hkdf_provider: crate::hash::HKDF_SHA256,
  aead_alg: crate::aead::CHACHA20_POLY1305,
  quic: Some(crate::quic::CHACHA20_POLY1305),
};

/// All enabled suites, in the provider's default preference order.
pub static ALL_CIPHER_SUITES: &[&Tls13CipherSuite] = &[
  #[cfg(all(feature = "hkdf", feature = "aes-gcm"))]
  TLS13_AES_256_GCM_SHA384,
  #[cfg(all(feature = "hkdf", feature = "aes-gcm"))]
  TLS13_AES_128_GCM_SHA256,
  #[cfg(all(feature = "hkdf", feature = "chacha20poly1305"))]
  TLS13_CHACHA20_POLY1305_SHA256,
];

/// TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384 (RFC 5289), for TLS 1.2.
#[cfg(all(any(feature = "hmac", feature = "hkdf"), feature = "aes-gcm"))]
pub static TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384: &Tls12CipherSuite = &Tls12CipherSuite {
  common: CipherSuiteCommon {
    suite: CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
    hash_provider: crate::hash::SHA384,
    // The AES-GCM record analysis of RFC 9846 §5.5 does not depend on the TLS
    // version: the same conservative bound for full-sized records.
    confidentiality_limit: 1 << 24,
  },
  protocol_version: rustls::version::TLS12_VERSION,
  prf_provider: crate::hash::PRF_SHA384,
  kx: KeyExchangeAlgorithm::ECDHE,
  sign: TLS12_ECDSA_SCHEMES,
  aead_alg: crate::aead::TLS12_AES_256_GCM,
};

/// TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256 (RFC 5289), for TLS 1.2.
#[cfg(all(any(feature = "hmac", feature = "hkdf"), feature = "aes-gcm"))]
pub static TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256: &Tls12CipherSuite = &Tls12CipherSuite {
  common: CipherSuiteCommon {
    suite: CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
    hash_provider: crate::hash::SHA256,
    confidentiality_limit: 1 << 24,
  },
  protocol_version: rustls::version::TLS12_VERSION,
  prf_provider: crate::hash::PRF_SHA256,
  kx: KeyExchangeAlgorithm::ECDHE,
  sign: TLS12_ECDSA_SCHEMES,
  aead_alg: crate::aead::TLS12_AES_128_GCM,
};

/// TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256 (RFC 7905), for TLS 1.2.
#[cfg(all(any(feature = "hmac", feature = "hkdf"), feature = "chacha20poly1305"))]
pub static TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256: &Tls12CipherSuite = &Tls12CipherSuite {
  common: CipherSuiteCommon {
    suite: CipherSuite::TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
    hash_provider: crate::hash::SHA256,
    confidentiality_limit: u64::MAX,
  },
  protocol_version: rustls::version::TLS12_VERSION,
  prf_provider: crate::hash::PRF_SHA256,
  kx: KeyExchangeAlgorithm::ECDHE,
  sign: TLS12_ECDSA_SCHEMES,
  aead_alg: crate::aead::TLS12_CHACHA20_POLY1305,
};

/// TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384 (RFC 5289), for TLS 1.2.
#[cfg(all(any(feature = "hmac", feature = "hkdf"), feature = "aes-gcm"))]
pub static TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384: &Tls12CipherSuite = &Tls12CipherSuite {
  common: CipherSuiteCommon {
    suite: CipherSuite::TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
    hash_provider: crate::hash::SHA384,
    confidentiality_limit: 1 << 24,
  },
  protocol_version: rustls::version::TLS12_VERSION,
  prf_provider: crate::hash::PRF_SHA384,
  kx: KeyExchangeAlgorithm::ECDHE,
  sign: TLS12_RSA_SCHEMES,
  aead_alg: crate::aead::TLS12_AES_256_GCM,
};

/// TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256 (RFC 5289), for TLS 1.2.
#[cfg(all(any(feature = "hmac", feature = "hkdf"), feature = "aes-gcm"))]
pub static TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256: &Tls12CipherSuite = &Tls12CipherSuite {
  common: CipherSuiteCommon {
    suite: CipherSuite::TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
    hash_provider: crate::hash::SHA256,
    confidentiality_limit: 1 << 24,
  },
  protocol_version: rustls::version::TLS12_VERSION,
  prf_provider: crate::hash::PRF_SHA256,
  kx: KeyExchangeAlgorithm::ECDHE,
  sign: TLS12_RSA_SCHEMES,
  aead_alg: crate::aead::TLS12_AES_128_GCM,
};

/// TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256 (RFC 7905), for TLS 1.2.
#[cfg(all(any(feature = "hmac", feature = "hkdf"), feature = "chacha20poly1305"))]
pub static TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256: &Tls12CipherSuite = &Tls12CipherSuite {
  common: CipherSuiteCommon {
    suite: CipherSuite::TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
    hash_provider: crate::hash::SHA256,
    confidentiality_limit: u64::MAX,
  },
  protocol_version: rustls::version::TLS12_VERSION,
  prf_provider: crate::hash::PRF_SHA256,
  kx: KeyExchangeAlgorithm::ECDHE,
  sign: TLS12_RSA_SCHEMES,
  aead_alg: crate::aead::TLS12_CHACHA20_POLY1305,
};

/// All enabled TLS 1.2 suites, in the provider's preference order.
///
/// The AEAD order matches [`ALL_CIPHER_SUITES`]. ECDSA suites precede RSA so a
/// server holding both certificate types signs with the smaller, faster key.
/// No provider constructor includes these suites, and QUIC rejects them.
/// Enable them on an owned provider, and require the extended master secret:
///
/// ```
/// # #[cfg(feature = "default")]
/// # {
/// use std::sync::Arc;
///
/// let mut provider = rscrypto_rustls::default_provider();
/// provider.tls12_cipher_suites = rscrypto_rustls::cipher_suite::ALL_TLS12_CIPHER_SUITES.into();
/// let mut config = rustls::ClientConfig::builder(Arc::new(provider))
///   .with_root_certificates(rustls::RootCertStore::empty())
///   .with_no_client_auth()?;
/// // Refuse TLS 1.2 peers without the extended master secret (RFC 7627).
/// config.require_ems = true;
/// # let _ = config;
/// # }
/// # Ok::<(), rustls::Error>(())
/// ```
#[cfg(all(
  any(feature = "hmac", feature = "hkdf"),
  any(feature = "aes-gcm", feature = "chacha20poly1305")
))]
pub static ALL_TLS12_CIPHER_SUITES: &[&Tls12CipherSuite] = &[
  #[cfg(feature = "aes-gcm")]
  TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
  #[cfg(feature = "aes-gcm")]
  TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
  #[cfg(feature = "chacha20poly1305")]
  TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
  #[cfg(feature = "aes-gcm")]
  TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
  #[cfg(feature = "aes-gcm")]
  TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
  #[cfg(feature = "chacha20poly1305")]
  TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
];

// RFC 8422 §5.1.1 permits EdDSA under the ECDHE_ECDSA suites. TLS 1.2 binds
// no ECDSA scheme to a curve (RFC 5246 §7.4.1.4.1); the signing key and the
// verifier's algorithms decide which curves each scheme covers.
#[cfg(all(
  any(feature = "hmac", feature = "hkdf"),
  any(feature = "aes-gcm", feature = "chacha20poly1305")
))]
static TLS12_ECDSA_SCHEMES: &[SignatureScheme] = &[
  SignatureScheme::ED25519,
  SignatureScheme::ECDSA_NISTP384_SHA384,
  SignatureScheme::ECDSA_NISTP256_SHA256,
];

#[cfg(all(
  any(feature = "hmac", feature = "hkdf"),
  any(feature = "aes-gcm", feature = "chacha20poly1305")
))]
static TLS12_RSA_SCHEMES: &[SignatureScheme] = &[
  SignatureScheme::RSA_PSS_SHA512,
  SignatureScheme::RSA_PSS_SHA384,
  SignatureScheme::RSA_PSS_SHA256,
  SignatureScheme::RSA_PKCS1_SHA512,
  SignatureScheme::RSA_PKCS1_SHA384,
  SignatureScheme::RSA_PKCS1_SHA256,
];
