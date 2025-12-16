//! Core cryptographic traits for rscrypto.
//!
//! This crate provides the foundational traits that all rscrypto implementations
//! conform to. It is `no_std` compatible and has zero dependencies.
//!
//! # Trait Hierarchy
//!
//! ```text
//! Checksum (CRC, xxHash, etc.)
//!     └── ChecksumCombine (parallel processing support)
//!
//! Digest (Blake3, SHA2, SHA3)
//!     └── ExtendableOutput (XOF: SHAKE, Blake3)
//!
//! Mac (HMAC, Poly1305, CMAC)
//!
//! Aead (AES-GCM, ChaCha20-Poly1305, AEGIS)
//!
//! Cipher (AES, ChaCha20)
//!
//! Kdf (HKDF, Argon2, scrypt)
//! ```
//!
//! # Error Types
//!
//! - [`VerificationError`] - Opaque error for MAC/AEAD/signature verification

#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod checksum;
pub mod error;

pub use checksum::{Checksum, ChecksumCombine};
pub use error::VerificationError;
