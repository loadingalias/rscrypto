//! Cryptographic hash functions.

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::indexing_slicing))]

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
#[cfg(any(
  feature = "sha2",
  feature = "blake3",
  all(feature = "ascon-hash", any(test, feature = "std"))
))]
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
pub use blake2b::{Blake2b, Blake2b256, Blake2b512, Blake2bParams};
#[cfg(feature = "blake2s")]
pub use blake2s::{Blake2s128, Blake2s256, Blake2sParams};
#[cfg(feature = "blake3")]
pub use blake3::{Blake3, Blake3XofReader};
#[cfg(feature = "sha3")]
pub use cshake::{Cshake256, Cshake256XofReader};
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
