//! Cryptographic digests and fast non-cryptographic hashes.
//!
//! This module is `no_std` compatible and has zero library dependencies outside
//! the rscrypto workspace. Dev-only dependencies are used for oracle testing
//! and benchmarking.
//!
//! # Quick Start
//!
//! ```rust
//! use rscrypto::{Digest, FastHash, Sha256, Shake256, Xof, Xxh3};
//!
//! let digest = Sha256::digest(b"hello");
//!
//! let mut streaming = Sha256::new();
//! streaming.update(b"he");
//! streaming.update(b"llo");
//! assert_eq!(streaming.finalize(), digest);
//!
//! let mut xof = Shake256::xof(b"hello");
//! let mut out = [0u8; 32];
//! xof.squeeze(&mut out);
//! assert_ne!(out, [0u8; 32]);
//!
//! let fast = Xxh3::hash(b"hello");
//! assert_ne!(fast, 0);
//! ```
//!
//! # Feature Selection
//!
//! Pick leaves for minimum size and bundles for category intent:
//!
//! ```toml
//! [dependencies]
//! # Smallest SHA-2-only build
//! rscrypto = { version = "0.1", default-features = false, features = ["sha2"] }
//!
//! # All cryptographic hashes
//! rscrypto = { version = "0.1", default-features = false, features = ["crypto-hashes"] }
//!
//! # Fast non-cryptographic hashes only
//! rscrypto = { version = "0.1", default-features = false, features = ["fast-hashes"] }
//!
//! # Everything hash-related
//! rscrypto = { version = "0.1", default-features = false, features = ["hashes"] }
//! ```
//!
//! # API Conventions
//!
//! - Fixed-output digests use `Type::digest(data)` for one-shot and `new` / `update` / `finalize` /
//!   `reset` for streaming.
//! - XOFs use `Type::xof(data)` for one-shot and `finalize_xof()` for streaming squeeze readers.
//! - Fast hashes are one-shot only and implement [`crate::traits::FastHash`].
//!
//! # Modules
//!
//! - `crypto` - Cryptographic hash functions (safe by default).
//! - `fast` - Non-cryptographic hashes (**NOT CRYPTO**).
//! - `introspect` (requires `diag` feature) - Advanced kernel selection reporting.
//!
//! # Advanced
//!
//! Dispatch is automatic by default.
//!
//! - Use `crate::hashes::introspect` (requires `diag` feature) for kernel reporting and size-based
//!   dispatch details.
//! - Use `crate::hashes::fast` for explicit fast-hash family access.
#[doc(hidden)]
pub(crate) mod common;
#[cfg(any(
  feature = "sha2",
  feature = "sha3",
  feature = "blake2b",
  feature = "blake2s",
  feature = "blake3",
  feature = "ascon-hash"
))]
pub mod crypto;
#[cfg(any(feature = "xxh3", feature = "rapidhash"))]
pub mod fast;
#[cfg(feature = "diag")]
pub mod introspect;
#[cfg(all(
  feature = "std",
  any(
    feature = "sha2",
    feature = "sha3",
    feature = "blake2b",
    feature = "blake2s",
    feature = "blake3",
    feature = "ascon-hash"
  )
))]
pub mod io;

#[cfg(any(feature = "sha2", test))]
mod util;

// Re-export I/O adapters (requires std)
#[cfg(all(
  feature = "std",
  any(
    feature = "sha2",
    feature = "sha3",
    feature = "blake2b",
    feature = "blake2s",
    feature = "blake3",
    feature = "ascon-hash"
  )
))]
pub use io::{DigestReader, DigestWriter};

pub use crate::traits::{Digest, FastHash, Xof};
