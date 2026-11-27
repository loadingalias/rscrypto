//! Cryptographic hash functions.

#[cfg(any(feature = "blake2b", feature = "blake2s"))]
use core::fmt;

#[cfg(feature = "ascon-hash")]
pub mod ascon;
#[cfg(any(feature = "blake2b", feature = "blake2s"))]
#[macro_use]
mod blake2_macros;
#[cfg(feature = "blake2b")]
pub mod blake2b;
#[cfg(feature = "blake2s")]
pub mod blake2s;
#[cfg(feature = "blake3")]
pub mod blake3;
#[cfg(feature = "sha3")]
mod cshake;
#[cfg(feature = "blake3")]
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
#[cfg(feature = "blake2b")]
pub use blake2b::{Blake2b, Blake2b256, Blake2b512, Blake2bKey, Blake2bParams};
#[cfg(feature = "blake2s")]
pub use blake2s::{Blake2s128, Blake2s256, Blake2sKey, Blake2sParams};
#[cfg(feature = "blake3")]
pub use blake3::{Blake3, Blake3DeriveContext, Blake3KeyedHash, Blake3XofReader};
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

/// An operation that uses one SHA-2 compression function.
///
/// `Sha256KernelId::with` and `Sha512KernelId::with` runs it with the function
/// item of the selected kernel. Each kernel thus gets its own copy, and each
/// compression in it is a direct call.
#[cfg(feature = "sha2")]
pub(crate) trait WithCompress<S> {
  type Output;

  fn run(self, compress: impl Fn(&mut S, &[u8])) -> Self::Output;
}

/// Compress the whole blocks of `.1` into the state `.0`.
#[cfg(feature = "sha2")]
pub(crate) struct Compress<'a, S>(pub(crate) &'a mut S, pub(crate) &'a [u8]);

#[cfg(feature = "sha2")]
impl<S> WithCompress<S> for Compress<'_, S> {
  type Output = ();

  #[inline(always)]
  fn run(self, compress: impl Fn(&mut S, &[u8])) {
    compress(self.0, self.1);
  }
}

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
