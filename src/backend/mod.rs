//! Internal backend support for rscrypto.
//!
//! The public crate surface exposes platform detection and algorithm-level
//! introspection. This module holds internal caching and shared backend kernels
//! so algorithms can reuse implementation building blocks without creating
//! public-surface or feature-graph coupling.
#[cfg(any(feature = "ascon-hash", feature = "ascon-aead"))]
pub mod ascon;
#[cfg(any(
  feature = "crc16",
  feature = "crc24",
  feature = "crc32",
  feature = "crc64",
  feature = "sha2",
  feature = "sha3",
  feature = "blake3",
  feature = "ascon-hash",
  feature = "xxh3",
  feature = "rapidhash",
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305",
  feature = "aegis256",
  feature = "ascon-aead"
))]
pub mod cache;
#[cfg(any(feature = "ed25519", feature = "x25519"))]
pub mod curve25519;
