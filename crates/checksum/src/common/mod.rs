//! Common utilities for CRC computation.
//!
//! This module provides:
//! - Bitwise reference implementations for correctness verification
//! - Portable slice-by-N implementations for all CRC widths
//! - Const-fn lookup table generation for all CRC sizes
//! - GF(2) matrix operations for `combine()` implementation
//! - Generic kernel selection and dispatch infrastructure
//! - Generic test harnesses for CRC property testing
//! - PCLMULQDQ/PMULL folding constants for hardware acceleration

// CLMUL folding constants and helpers (used by SIMD CRC backends).
#[cfg(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "powerpc64",
  target_arch = "s390x",
  target_arch = "riscv64"
))]
pub mod clmul;
pub mod combine;
pub mod kernels;
pub mod portable;
pub mod reference;
pub mod tables;
#[cfg(test)]
pub mod tests;
