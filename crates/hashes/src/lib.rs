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
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::indexing_slicing))]
#![no_std]

#[cfg(feature = "std")]
extern crate std;

pub mod crypto;
pub mod fast;

mod util;

pub use traits::{Digest, FastHash, Xof};
