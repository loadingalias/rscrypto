//! I/O adapters for cryptographic digests.
//!
//! This module provides [`DigestReader`] and [`DigestWriter`] which wrap
//! [`std::io::Read`] and [`std::io::Write`] implementations to compute digests
//! transparently during I/O operations.
//!
//! # Performance
//!
//! - Zero-cost abstraction: All methods are `#[inline]`
//! - Vectored I/O support: Uses the `update_vectored` method when available
//! - Correctness: Only hashes bytes actually transferred (handles short reads/writes)
//!
//! # Example
//!
//! ```rust,ignore
//! use hashes::crypto::blake3::Blake3;
//! use std::fs::File;
//! use std::io::Read;
//!
//! let file = File::open("data.bin")?;
//! let mut reader = Blake3::reader(file);
//! let mut contents = Vec::new();
//! reader.read_to_end(&mut contents)?;
//! println!("Blake3: {:?}", reader.digest());
//! ```

pub use traits::io::{DigestReader, DigestWriter};
