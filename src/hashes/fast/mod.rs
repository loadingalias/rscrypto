//! Fast non-cryptographic hashes (**NOT CRYPTO**).
//!
//! This module intentionally requires explicit opt-in. Do not use these hashes
//! for signatures, MACs, key derivation, or anything requiring cryptographic
//! security.

#[cfg(feature = "rapidhash")]
pub mod rapidhash;
#[cfg(feature = "xxh3")]
pub mod xxh3;

#[cfg(feature = "rapidhash")]
pub use rapidhash::{RapidBuildHasher, RapidHasher, RapidStreamHasher};
#[cfg(feature = "rapidhash")]
pub use rapidhash::{RapidHash64, RapidHash64 as RapidHash, RapidHash128, RapidHashFast64, RapidHashFast128};
#[cfg(feature = "xxh3")]
pub use xxh3::{Xxh3_64, Xxh3_64 as Xxh3, Xxh3_128};
#[cfg(feature = "xxh3")]
pub use xxh3::{Xxh3_128Hasher, Xxh3BuildHasher, Xxh3Hasher};
