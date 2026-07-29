//! I/O adapters for cryptographic digests.
//!
//! This module provides [`DigestReader`] and [`DigestWriter`] which wrap
//! [`std::io::Read`] and [`std::io::Write`] implementations to compute digests
//! transparently during I/O operations.
//!
//! Vectored adapters update the digest once for each transferred slice. All
//! adapters hash only bytes actually transferred, including short reads and
//! writes.
//!
//! # Example
//!
//! ```rust
//! # #[cfg(feature = "sha2")]
//! # {
//! use std::io::{Cursor, Read};
//!
//! use rscrypto::{Digest as _, Sha256};
//!
//! let mut reader = Sha256::reader(Cursor::new(b"hello world".to_vec()));
//! let mut contents = Vec::new();
//! let bytes = reader.read_to_end(&mut contents).unwrap();
//! assert_eq!(bytes, b"hello world".len());
//! assert_eq!(contents, b"hello world");
//! assert_eq!(reader.digest(), Sha256::digest(&contents));
//! # }
//! ```

pub use crate::traits::io::{DigestReader, DigestWriter};
