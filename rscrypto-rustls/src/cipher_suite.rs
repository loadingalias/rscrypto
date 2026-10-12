//! TLS 1.3 suites. Each includes QUIC packet and header protection.

use rustls::Tls13CipherSuite;
#[cfg(all(feature = "hkdf", any(feature = "aes-gcm", feature = "chacha20poly1305")))]
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
