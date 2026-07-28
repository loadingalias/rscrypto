//! Cryptographic hash functions.

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::indexing_slicing))]

#[cfg(any(feature = "blake2b", feature = "blake2s"))]
use core::fmt;

#[cfg(feature = "ascon-hash")]
pub mod ascon;
#[cfg(feature = "blake2b")]
pub mod blake2b;
#[cfg(feature = "blake2s")]
pub mod blake2s;
#[cfg(feature = "blake3")]
pub mod blake3;
#[cfg(feature = "sha3")]
mod cshake;
#[cfg(any(feature = "sha2", feature = "blake3",))]
pub(crate) mod dispatch_util;
#[cfg(feature = "sha3")]
pub(crate) mod keccak;
#[cfg(feature = "sha2")]
pub mod sha224;
#[cfg(feature = "sha2")]
pub mod sha256;
#[cfg(feature = "sha3")]
pub mod sha3;
#[cfg(feature = "sha2")]
pub mod sha384;
#[cfg(feature = "sha2")]
pub mod sha512;
#[cfg(feature = "sha2")]
pub mod sha512_256;
#[cfg(feature = "sha3")]
pub(crate) mod sp800185;

#[cfg(feature = "ascon-hash")]
pub use ascon::{AsconCxof128, AsconCxof128Reader, AsconHash256, AsconXof, AsconXofReader};
#[cfg(all(feature = "diag", feature = "blake2b"))]
pub use blake2b::diag_blake2b256_keyed_digest_portable;
#[cfg(feature = "blake2b")]
pub use blake2b::{Blake2b, Blake2b256, Blake2b512, Blake2bKey, Blake2bParams};
#[cfg(all(feature = "diag", feature = "blake2s"))]
pub use blake2s::diag_blake2s256_keyed_digest_portable;
#[cfg(feature = "blake2s")]
pub use blake2s::{Blake2s128, Blake2s256, Blake2sKey, Blake2sParams};
#[cfg(all(feature = "diag", feature = "blake3"))]
pub use blake3::diag_blake3_keyed_digest_portable;
#[cfg(feature = "blake3")]
pub use blake3::{Blake3, Blake3KeyedHash, Blake3XofReader};
#[cfg(feature = "sha3")]
pub use cshake::{Cshake128, Cshake128XofReader, Cshake256, Cshake256XofReader};
#[cfg(feature = "sha3")]
pub use sha3::{Sha3_224, Sha3_256, Sha3_384, Sha3_512, Shake128, Shake128XofReader, Shake256, Shake256XofReader};
#[cfg(feature = "sha2")]
pub use sha224::Sha224;
#[cfg(feature = "sha2")]
pub use sha256::Sha256;
#[cfg(feature = "sha2")]
pub use sha384::Sha384;
#[cfg(feature = "sha2")]
pub use sha512::Sha512;
#[cfg(feature = "sha2")]
pub use sha512_256::Sha512_256;

/// Invalid BLAKE2 key, parameter, or output length.
#[cfg(any(feature = "blake2b", feature = "blake2s"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Blake2Error {
  /// A keyed operation requires a non-empty key within the algorithm's limit.
  InvalidKeyLength,
  /// A variable BLAKE2b output must contain 1 to 64 bytes.
  InvalidOutputLength,
  /// The caller-provided output buffer does not match the configured length.
  OutputLengthMismatch,
}

#[cfg(any(feature = "blake2b", feature = "blake2s"))]
impl fmt::Display for Blake2Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::InvalidKeyLength => f.write_str("BLAKE2 key length is invalid"),
      Self::InvalidOutputLength => f.write_str("BLAKE2 output length is invalid"),
      Self::OutputLengthMismatch => f.write_str("BLAKE2 output buffer length does not match the configured length"),
    }
  }
}

#[cfg(any(feature = "blake2b", feature = "blake2s"))]
impl core::error::Error for Blake2Error {}
