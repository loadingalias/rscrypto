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
//! | [`Digest`] | Cryptographic digests | BLAKE3, SHA-2 |
//! | [`Mac`] | Message authentication codes | HMAC-SHA256 |
//! | [`Xof`] | Extendable-output functions | BLAKE3 XOF |
//! | [`FastHash`] | Fast non-cryptographic hashes (**NOT CRYPTO**) | XXH3, rapidhash |
//!
//! Future traits (not yet implemented):
//!
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
mod checksum;
pub mod ct;
mod digest;
pub mod error;
mod fast_hash;
#[cfg(feature = "std")]
pub mod io;
mod mac;
mod xof;

pub use checksum::{Checksum, ChecksumCombine};
pub use digest::Digest;
pub use error::VerificationError;
pub use fast_hash::FastHash;
pub use mac::Mac;
pub use xof::Xof;
