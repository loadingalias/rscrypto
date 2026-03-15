//! Target-matrix contract helpers.
//!
//! This module ties platform detection to `config/target-matrix.toml` so
//! architecture policy stays aligned with CI manifests.

use crate::platform::caps::Arch;

const TARGET_MATRIX_MANIFEST: &str = include_str!("../../config/target-matrix.toml");

#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
  Other,
  Groups,
  CiCommit,
}

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

#[inline]
fn quoted_strings_in<'a>(line: &'a str) -> impl Iterator<Item = &'a str> + 'a {
  line.split('"').skip(1).step_by(2)
}

#[inline]
#[must_use]
pub fn manifest_has_arch(arch: Arch) -> bool {
  let Some(_) = manifest_prefix_for_arch(arch) else {
    return true;
  };

  let mut section = Section::Other;
  let mut in_group_array = false;

  for raw_line in TARGET_MATRIX_MANIFEST.lines() {
    let line = raw_line.split('#').next().unwrap_or("").trim();
    if line.is_empty() {
      continue;
    }

    if line.starts_with("[[") {
      section = if line == "[[ci.commit]]" {
        Section::CiCommit
      } else {
        Section::Other
      };
      in_group_array = false;
      continue;
    }

    if line.starts_with('[') {
      section = if line == "[groups]" {
        Section::Groups
      } else {
        Section::Other
      };
      in_group_array = false;
      continue;
    }

    match section {
      Section::Groups => {
        if in_group_array || line.contains('=') {
          for value in quoted_strings_in(line) {
            if matches_target_arch(value, arch) {
              return true;
            }
          }

          if line.contains('=') {
            in_group_array = line.contains('[') && !line.contains(']');
          } else if line.contains(']') {
            in_group_array = false;
          }
        }
      }
      Section::CiCommit => {
        if let Some((key, value)) = line.split_once('=')
          && key.trim() == "name"
        {
          for name in quoted_strings_in(value) {
            if matches_target_arch(name, arch) {
              return true;
            }
          }
        }
      }
      Section::Other => {}
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
