//! Cryptographic hash functions with automatic SIMD acceleration.
//!
//! # Algorithms
//!
//! - **Blake3** - Modern, parallel, fastest secure hash
//! - **SHA2** (SHA-256, SHA-384, SHA-512)
//! - **SHA3** (SHA3-256, SHA3-384, SHA3-512, SHAKE)

#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;
