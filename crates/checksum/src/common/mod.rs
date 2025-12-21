//! Common utilities for CRC computation.
//!
//! This module provides:
//! - Const-fn lookup table generation for all CRC sizes
//! - GF(2) matrix operations for `combine()` implementation
//! - PCLMULQDQ/PMULL folding constants for hardware acceleration

pub mod clmul;
pub mod combine;
pub mod tables;
