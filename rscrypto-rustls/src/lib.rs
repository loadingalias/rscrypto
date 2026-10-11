//! TLS 1.3 provider integration for rustls, backed by rscrypto.
//!
//! This unpublished crate provides symmetric TLS and QUIC adapters.
//! Provider constructors follow with authentication.
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
#[cfg(feature = "getrandom")]
pub mod random;

#[cfg(all(test, feature = "hkdf", feature = "aes-gcm", feature = "chacha20poly1305"))]
mod tests;
