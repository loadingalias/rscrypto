//! Internal backend support for rscrypto.
//!
//! The public crate surface exposes platform detection and algorithm-level
//! introspection. This module only holds internal caching used by runtime
//! dispatch so that algorithms can detect capabilities once and reuse the
//! selected implementation.
pub mod cache;
