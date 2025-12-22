//! Common utilities for CRC computation.
//!
//! This module provides:
//! - Const-fn lookup table generation for all CRC sizes
//! - GF(2) matrix operations for `combine()` implementation
//! - PCLMULQDQ/PMULL folding constants for hardware acceleration

// CLMUL folding constants and helpers (used by SIMD CRC backends).
#[cfg(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  all(target_arch = "powerpc64", target_endian = "little")
))]
pub mod clmul;
pub mod combine;
pub mod tables;
