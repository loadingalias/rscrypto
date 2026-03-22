//! Cryptographic hash functions.

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::indexing_slicing))]

pub mod ascon;
pub mod blake3;
pub(crate) mod dispatch_util;
pub(crate) mod keccak;
pub mod sha224;
pub mod sha256;
pub mod sha3;
pub mod sha384;
pub mod sha512;
pub mod sha512_256;

pub use ascon::{AsconHash256, AsconXof, AsconXofReader};
pub use blake3::{Blake3, Blake3Xof};
pub use sha3::{Sha3_224, Sha3_256, Sha3_384, Sha3_512, Shake128, Shake128Xof, Shake256, Shake256Xof};
pub use sha224::Sha224;
pub use sha256::Sha256;
pub use sha384::Sha384;
pub use sha512::Sha512;
pub use sha512_256::Sha512_256;
