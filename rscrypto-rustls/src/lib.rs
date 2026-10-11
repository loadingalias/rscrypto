//! TLS 1.3 provider integration for rustls, backed by rscrypto.
//!
//! This unpublished crate currently contains the dependency and feature skeleton.
//! Cryptographic adapters and provider constructors are not implemented yet.
//! The crate requires `std`; its algorithm features forward to rscrypto's leaf features.

#![forbid(unsafe_code)]
