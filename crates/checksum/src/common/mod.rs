//! Common utilities for CRC computation.
//!
//! This module provides:
//! - Const-fn lookup table generation for all CRC sizes
//! - GF(2) matrix operations for `combine()` implementation
//! - PCLMULQDQ/PMULL folding constants for hardware acceleration

// CLMUL constants only used on SIMD architectures (x86_64, aarch64)
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod clmul;
pub mod combine;
pub mod tables;
