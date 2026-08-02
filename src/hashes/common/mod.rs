//! Shared hash implementation utilities.
//!
//! This module currently owns the AArch64 BLAKE3 prefetch helper.

#[cfg(all(feature = "blake3", target_arch = "aarch64"))]
pub mod prefetch;
