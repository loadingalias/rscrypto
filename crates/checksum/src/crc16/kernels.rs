//! Static kernel name tables for CRC-16.
//!
//! This mirrors the structure used by CRC-32/CRC-64: keep all kernel names in
//! one place so both introspection and dispatch share identifiers.

/// Portable slice-by-4 kernel name.
pub const PORTABLE_SLICE4: &str = crate::common::kernels::PORTABLE_SLICE4;
/// Portable slice-by-8 kernel name.
pub const PORTABLE_SLICE8: &str = crate::common::kernels::PORTABLE_SLICE8;
/// Portable auto-selection kernel name (slice4 vs slice8 by length).
pub const PORTABLE_AUTO: &str = "portable/auto";

/// x86_64 PCLMULQDQ folding kernel name.
#[cfg(target_arch = "x86_64")]
pub const X86_64_PCLMUL: &str = "x86_64/pclmul";

/// x86_64 VPCLMULQDQ folding kernel name (AVX-512).
#[cfg(target_arch = "x86_64")]
pub const X86_64_VPCLMUL: &str = "x86_64/vpclmul";

/// aarch64 PMULL folding kernel name.
#[cfg(target_arch = "aarch64")]
pub const AARCH64_PMULL: &str = "aarch64/pmull";

/// Portable kernel name table (ordered by increasing work per byte).
#[allow(dead_code)]
pub const PORTABLE_NAMES: &[&str] = &[PORTABLE_SLICE4, PORTABLE_SLICE8];

#[inline]
#[must_use]
pub const fn portable_name_for_len(len: usize, slice4_to_slice8: usize) -> &'static str {
  if len < slice4_to_slice8 {
    PORTABLE_SLICE4
  } else {
    PORTABLE_SLICE8
  }
}
