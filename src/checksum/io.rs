//! I/O adapters for checksum computation.
//!
//! This module provides [`ChecksumReader`] and [`ChecksumWriter`] which wrap
//! [`std::io::Read`] and [`std::io::Write`] implementations to compute checksums
//! transparently during I/O operations.
//!
//! # Performance
//!
//! - Inline-friendly methods over the checksum traits
//! - Vectored I/O support: Uses the `update_vectored` method when available
//! - Correctness: Only hashes bytes actually transferred (handles short reads/writes)
//!
//! # Example
//!
//! ```rust
//! # #[cfg(feature = "crc32")]
//! # {
//! use std::io::{Cursor, Read};
//!
//! use rscrypto::{
//!   Checksum as _,
//!   checksum::{ChecksumReader, Crc32C},
//! };
//!
//! let mut reader = Crc32C::reader(Cursor::new(b"hello world".to_vec()));
//! let mut contents = Vec::new();
//! reader.read_to_end(&mut contents).unwrap();
//! assert_eq!(contents, b"hello world");
//! assert_eq!(reader.checksum(), Crc32C::checksum(&contents));
//! # }
//! ```

pub use crate::traits::io::{ChecksumReader, ChecksumWriter};
