//! Fast non-cryptographic hashes (**NOT CRYPTO**).
//!
//! This module intentionally requires explicit opt-in. Do not use these hashes
//! for signatures, MACs, key derivation, or anything requiring cryptographic
//! security.

#[cfg(feature = "aeshash")]
pub mod aeshash;
#[cfg(feature = "rapidhash")]
pub mod rapidhash;
#[cfg(feature = "xxh3")]
pub mod xxh3;

#[cfg(feature = "aeshash")]
pub use aeshash::{AesHash64, AesHash128};
#[cfg(all(feature = "rapidhash", feature = "alloc"))]
pub use rapidhash::{RapidBuildHasher, RapidHasher};
#[cfg(feature = "rapidhash")]
pub use rapidhash::{RapidHash64, RapidHash64 as RapidHash, RapidHash128, RapidHashFast64, RapidHashFast128};
#[cfg(feature = "xxh3")]
pub use xxh3::{Xxh3_64, Xxh3_64 as Xxh3, Xxh3_128};
#[cfg(all(feature = "xxh3", feature = "alloc"))]
pub use xxh3::{Xxh3BuildHasher, Xxh3Hasher};
