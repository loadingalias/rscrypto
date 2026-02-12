//! I/O adapters for checksum computation.
//!
//! This module provides [`ChecksumReader`] and [`ChecksumWriter`] which wrap
//! [`std::io::Read`] and [`std::io::Write`] implementations to compute checksums
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
//! use checksum::{ChecksumReader, Crc32C};
//! use traits::Checksum as _;
//!
//! let mut reader = Crc32C::reader(Cursor::new(b"hello world".to_vec()));
//! let mut contents = Vec::new();
//! reader.read_to_end(&mut contents)?;
//! assert_eq!(contents, b"hello world");
//! assert_eq!(reader.crc(), Crc32C::checksum(&contents));
//! # Ok::<(), std::io::Error>(())
//! ```

pub use traits::io::{ChecksumReader, ChecksumWriter};
