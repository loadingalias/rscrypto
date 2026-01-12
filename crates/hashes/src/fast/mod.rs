//! Fast non-cryptographic hashes (**NOT CRYPTO**).
//!
//! This module intentionally requires explicit opt-in. Do not use these hashes
//! for signatures, MACs, key derivation, or anything requiring cryptographic
//! security.

pub mod rapidhash;
pub mod siphash;
pub mod xxh3;

pub use rapidhash::{RapidHash64, RapidHash128};
pub use siphash::{SipHash13, SipHash24};
pub use xxh3::{Xxh3_64, Xxh3_128};
