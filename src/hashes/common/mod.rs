//! Common utilities for hash computation.
//!
//! This module provides:
//! - Cross-architecture SIMD abstractions for hash primitives
//! - Generic kernel selection and dispatch infrastructure
//! - Software prefetch helpers for optimal memory access patterns
//! - Shared compression patterns for hash algorithms
//!
//! # Design Philosophy
//!
//! The patterns here mirror `checksum::common` to maintain consistency
//! across the rscrypto crate ecosystem. Hash algorithms share many
//! optimization techniques (SIMD vectorization, prefetch hints, etc.)
//! that can be centralized here.

pub mod kernels;
pub mod prefetch;
pub mod simd;
