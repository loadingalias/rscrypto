//! Core cryptographic traits for rscrypto.
//!
//! This module provides the foundational traits that all rscrypto implementations
//! conform to. It is `no_std` compatible and has zero dependencies.
//!
//! # Trait Hierarchy
//!
//! | Trait | Purpose | Examples |
//! |-------|---------|----------|
//! | `Aead` | Authenticated encryption with associated data | XChaCha20-Poly1305 |
//! | [`Checksum`] | Non-cryptographic checksums | CRC32, CRC64 |
//! | [`ChecksumCombine`] | Parallel checksum combination | CRC with O(log n) combine |
//! | [`Digest`] | Cryptographic digests | BLAKE3, SHA-2 |
//! | [`Mac`] | Message authentication codes | HMAC-SHA256 |
//! | [`Xof`] | Extendable-output functions | BLAKE3 XOF |
//! | [`FastHash`] | Fast non-cryptographic hashes (**NOT CRYPTO**) | XXH3, rapidhash |
//!
//! These traits deliberately share a small set of repeated patterns:
//! streaming `new`/`update`/`finalize`/`reset` for stateful algorithms,
//! one-shot helpers for in-memory inputs, and opaque verification failures.
//!
//! # Error Types
//!
//! - [`VerificationError`] - Opaque error for MAC/AEAD/signature verification
//!
//! # Fallibility Discipline
//!
//! This module denies `unwrap`, `expect`, and indexing in non-test code to ensure
//! all error paths are handled explicitly.
#[cfg(any(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305",
  feature = "aegis256",
  feature = "ascon-aead"
))]
mod aead;
mod checksum;
pub mod ct;
mod digest;
pub mod error;
mod fast_hash;
#[cfg(feature = "std")]
pub mod io;
mod mac;
mod xof;

#[cfg(any(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305",
  feature = "aegis256",
  feature = "ascon-aead"
))]
pub use aead::Aead;
pub use checksum::{Checksum, ChecksumCombine};
pub use ct::ConstantTimeEq;
pub use digest::Digest;
pub use error::VerificationError;
pub use fast_hash::FastHash;
pub use mac::Mac;
pub use xof::Xof;
