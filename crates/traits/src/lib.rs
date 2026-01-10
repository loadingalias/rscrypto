//! Core cryptographic traits for rscrypto.
//!
//! This crate provides the foundational traits that all rscrypto implementations
//! conform to. It is `no_std` compatible and has zero dependencies.
//!
//! # Trait Hierarchy
//!
//! | Trait | Purpose | Examples |
//! |-------|---------|----------|
//! | [`Checksum`] | Non-cryptographic checksums | CRC32, CRC64, xxHash |
//! | [`ChecksumCombine`] | Parallel checksum combination | CRC with O(log n) combine |
//!
//! Future traits (not yet implemented):
//!
//! - `Digest` - Cryptographic hash functions (Blake3, SHA-2, SHA-3)
//! - `Mac` - Message authentication codes (HMAC, Poly1305, CMAC)
//! - `Aead` - Authenticated encryption (AES-GCM, ChaCha20-Poly1305, AEGIS)
//! - `Cipher` - Block/stream ciphers (AES, ChaCha20)
//! - `Kdf` - Key derivation functions (HKDF, Argon2, scrypt)
//!
//! # Error Types
//!
//! - [`VerificationError`] - Opaque error for MAC/AEAD/signature verification
//!
//! # Fallibility Discipline
//!
//! This crate denies `unwrap`, `expect`, and indexing in non-test code to ensure
//! all error paths are handled explicitly.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::indexing_slicing))]
#![no_std]

#[cfg(feature = "std")]
extern crate std;

mod checksum;
pub mod error;

pub use checksum::{Checksum, ChecksumCombine};
pub use error::VerificationError;
