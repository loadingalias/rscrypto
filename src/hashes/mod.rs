//! Cryptographic digests and fast non-cryptographic hashes.
//!
//! This crate is `no_std` compatible and has zero library dependencies outside
//! the rscrypto workspace. Dev-only dependencies are used for oracle testing
//! and benchmarking.
//!
//! # Modules
//!
//! - [`crypto`] - Cryptographic hash functions (safe by default).
//! - [`fast`] - Non-cryptographic hashes (**NOT CRYPTO**).
//! - [`introspect`] - Advanced kernel selection reporting.
//!
//! # Advanced
//!
//! Dispatch is automatic by default.
//!
//! - Use [`crate::hashes::introspect`] for kernel reporting and size-based dispatch details.
//! - Use [`crate::hashes::fast`] for explicit fast-hash family access.
#[doc(hidden)]
pub(crate) mod common;
pub mod crypto;
pub mod fast;
pub mod introspect;
#[cfg(feature = "std")]
pub mod io;

mod util;

// Re-export I/O adapters (requires std)
#[cfg(feature = "std")]
pub use io::{DigestReader, DigestWriter};

pub use crate::traits::{Digest, FastHash, Xof};
