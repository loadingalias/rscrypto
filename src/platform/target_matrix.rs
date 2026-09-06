//! Target-matrix contract helpers.
//!
//! This module ties platform detection to `.config/target-matrix.json` so
//! architecture policy stays aligned with the support catalog.

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

/// Scan quoted catalog strings for a target triple matching `arch`.
/// Repository validation owns the exact row schema; this embedded check keeps
/// runtime detection tied to a catalog that names the host architecture.
#[inline]
#[must_use]
pub(super) fn manifest_has_arch(arch: Arch) -> bool {
  let Some(_) = manifest_prefix_for_arch(arch) else {
    return true;
  };

  let mut in_quote = false;
  let mut start = 0;

  for (i, b) in TARGET_MATRIX_MANIFEST.bytes().enumerate() {
    if b == b'"' {
      if in_quote {
        if TARGET_MATRIX_MANIFEST
          .get(start..i)
          .is_some_and(|value| matches_target_arch(value, arch))
        {
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
