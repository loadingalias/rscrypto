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
//! ```rust
//! use std::io::{Cursor, Read};
//!
//! use hashes::crypto::Blake3;
//! use traits::Digest as _;
//!
//! let mut reader = Blake3::reader(Cursor::new(b"hello world".to_vec()));
//! let mut contents = Vec::new();
//! reader.read_to_end(&mut contents)?;
//! assert_eq!(contents, b"hello world");
//! assert_eq!(reader.digest(), Blake3::digest(&contents));
//! # Ok::<(), std::io::Error>(())
//! ```

pub use traits::io::{DigestReader, DigestWriter};
