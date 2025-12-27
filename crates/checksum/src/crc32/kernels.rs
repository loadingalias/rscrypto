//! Static kernel name tables and dispatch helpers for CRC-32.
//!
//! This module mirrors the CRC-64 layout and centralizes all kernel names so
//! both name introspection and dispatch can share the same identifiers.

/// Portable fallback kernel name.
pub use kernels::PORTABLE_SLICE16 as PORTABLE;

use crate::{common::kernels, dispatchers::Crc32Fn};

// Generate CRC32-specific dispatch functions using the common macro.
crate::define_crc_dispatch!(Crc32Fn, u32);

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Name Tables (per architecture)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
  use super::super::x86_64 as arch;
  use crate::dispatchers::Crc32Fn;

  /// SSE4.2 `crc32` instruction kernel name (CRC-32C only).
  pub const CRC32C_NAMES: &[&str] = &[
    "x86_64/crc32c",
    "x86_64/crc32c",
    "x86_64/crc32c",
    "x86_64/crc32c",
    "x86_64/crc32c",
  ];

  /// CRC-32C SSE4.2 kernel function array (no multi-stream variants yet).
  pub const CRC32C: [Crc32Fn; 5] = [
    arch::crc32c_sse42_safe,
    arch::crc32c_sse42_safe,
    arch::crc32c_sse42_safe,
    arch::crc32c_sse42_safe,
    arch::crc32c_sse42_safe,
  ];
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
  use super::super::aarch64 as arch;
  use crate::dispatchers::Crc32Fn;

  /// ARMv8 CRC extension kernel names.
  pub const CRC32_NAMES: &[&str] = &[
    "aarch64/crc32",
    "aarch64/crc32",
    "aarch64/crc32",
    "aarch64/crc32",
    "aarch64/crc32",
  ];

  pub const CRC32C_NAMES: &[&str] = &[
    "aarch64/crc32c",
    "aarch64/crc32c",
    "aarch64/crc32c",
    "aarch64/crc32c",
    "aarch64/crc32c",
  ];

  /// CRC-32 (IEEE) CRC-extension kernel function array (no multi-stream variants yet).
  pub const CRC32: [Crc32Fn; 5] = [
    arch::crc32_armv8_safe,
    arch::crc32_armv8_safe,
    arch::crc32_armv8_safe,
    arch::crc32_armv8_safe,
    arch::crc32_armv8_safe,
  ];

  /// CRC-32C (Castagnoli) CRC-extension kernel function array (no multi-stream variants yet).
  pub const CRC32C: [Crc32Fn; 5] = [
    arch::crc32c_armv8_safe,
    arch::crc32c_armv8_safe,
    arch::crc32c_armv8_safe,
    arch::crc32c_armv8_safe,
    arch::crc32c_armv8_safe,
  ];
}
