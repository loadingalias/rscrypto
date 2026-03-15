//! Target-matrix contract helpers.
//!
//! This module ties platform detection to `.config/target-matrix.json` so
//! architecture policy stays aligned with CI manifests.

use crate::platform::caps::Arch;

const TARGET_MATRIX_MANIFEST: &str = include_str!("../../.config/target-matrix.json");

#[inline]
const fn manifest_prefix_for_arch(arch: Arch) -> Option<&'static str> {
  match arch {
    Arch::X86_64 => Some("x86_64-"),
    Arch::Aarch64 => Some("aarch64-"),
    _ => None,
  }
}

#[inline]
fn matches_target_arch(target: &str, arch: Arch) -> bool {
  let Some(prefix) = manifest_prefix_for_arch(arch) else {
    return true;
  };
  target.starts_with(prefix)
}

/// Scan all quoted strings in the JSON manifest for a target triple matching
/// `arch`. The only quoted strings in the file are JSON keys (`"groups"`,
/// `"win"`, `"ci"`, etc.) and target triple values — keys never start with
/// an arch prefix like `x86_64-`, so false positives are impossible.
#[inline]
#[must_use]
pub fn manifest_has_arch(arch: Arch) -> bool {
  let Some(_) = manifest_prefix_for_arch(arch) else {
    return true;
  };

  let mut in_quote = false;
  let mut start = 0;

  for (i, b) in TARGET_MATRIX_MANIFEST.bytes().enumerate() {
    if b == b'"' {
      if in_quote {
        let value = &TARGET_MATRIX_MANIFEST[start..i];
        if matches_target_arch(value, arch) {
          return true;
        }
      } else {
        start = i.strict_add(1);
      }
      in_quote = !in_quote;
    }
  }

  false
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn manifest_contract_finds_commit_targets() {
    assert!(manifest_has_arch(Arch::X86_64));
    assert!(manifest_has_arch(Arch::Aarch64));
  }

  #[test]
  fn non_contract_arches_are_accepted() {
    assert!(manifest_has_arch(Arch::S390x));
    assert!(manifest_has_arch(Arch::Other));
  }
}
