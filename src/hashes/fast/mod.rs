//! Fast non-cryptographic hashes (**NOT CRYPTO**).
//!
//! This module intentionally requires explicit opt-in. Do not use these hashes
//! for signatures, MACs, key derivation, or anything requiring cryptographic
//! security.

#[cfg(feature = "rapidhash")]
pub mod rapidhash;
#[cfg(feature = "xxh3")]
pub mod xxh3;
#[cfg(feature = "xxh32")]
pub mod xxh32;
#[cfg(feature = "xxh64")]
pub mod xxh64;

#[cfg(feature = "rapidhash")]
pub use rapidhash::{RapidHash64, RapidHasher, RapidRandomState, RapidSeededState, RapidStreamHasher};
#[cfg(feature = "xxh3")]
pub use xxh3::{Xxh3_64, Xxh3_64 as Xxh3, Xxh3_128};
#[cfg(feature = "xxh3")]
pub use xxh3::{Xxh3_128Hasher, Xxh3BuildHasher, Xxh3Hasher};
#[cfg(feature = "xxh32")]
pub use xxh32::{Xxh32, Xxh32Hasher};
#[cfg(feature = "xxh64")]
pub use xxh64::{Xxh64, Xxh64Hasher};
