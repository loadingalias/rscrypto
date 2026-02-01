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
//! ```rust,ignore
//! use checksum::{Crc32C, ChecksumReader};
//! use std::fs::File;
//! use std::io::Read;
//!
//! let file = File::open("data.bin")?;
//! let mut reader = Crc32C::reader(file);
//! let mut contents = Vec::new();
//! reader.read_to_end(&mut contents)?;
//! println!("CRC: {:08x}", reader.crc());
//! ```

pub use traits::io::{ChecksumReader, ChecksumWriter};
