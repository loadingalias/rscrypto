//! Kernel name table and helpers for CRC-16.

/// Portable slice-by-4 kernel name.
pub const PORTABLE_SLICE4: &str = crate::common::kernels::PORTABLE_SLICE4;
/// Portable slice-by-8 kernel name.
pub const PORTABLE_SLICE8: &str = crate::common::kernels::PORTABLE_SLICE8;
/// Portable auto-selection kernel name (slice4 vs slice8 by length).
pub const PORTABLE_AUTO: &str = "portable/auto";
