//! Target-matrix contract helpers.
//!
//! This module ties platform detection to `config/target-matrix.toml` so
//! architecture policy stays aligned with CI manifests.

use crate::caps::Arch;

const TARGET_MATRIX_MANIFEST: &str = include_str!("../../../config/target-matrix.toml");

#[inline]
#[must_use]
pub fn manifest_has_arch(arch: Arch) -> bool {
  match arch {
    Arch::X86_64 => TARGET_MATRIX_MANIFEST.contains("\"x86_64\""),
    Arch::Aarch64 => TARGET_MATRIX_MANIFEST.contains("\"aarch64\""),
    _ => true,
  }
}
