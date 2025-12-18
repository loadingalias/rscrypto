//! Cryptographic hash functions with automatic SIMD acceleration.
//!
//! # Algorithms
//!
//! - **Blake3** - Modern, parallel, fastest secure hash
//! - **SHA2** (SHA-256, SHA-384, SHA-512)
//! - **SHA3** (SHA3-256, SHA3-384, SHA3-512, SHAKE)
//!
//! // Fallibility discipline: deny unwrap/expect in production, allow in tests.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::indexing_slicing))]
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;
