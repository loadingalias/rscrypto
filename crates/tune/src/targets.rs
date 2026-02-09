//! Performance target contracts for tuning outputs.
//!
//! These targets are "must-meet" throughput floors by algorithm, architecture,
//! and size bucket. They provide a stable contract for tuning output quality
//! and prevent silent regressions while iterating kernel policy.
use std::sync::OnceLock;

use platform::TuneKind;

#[cfg(feature = "std")]
use crate::TuneResults;
use crate::hash::is_blake3_tuning_algo;

const TARGET_MATRIX_MANIFEST: &str = include_str!("../../../config/target-matrix.toml");
static TUNE_ARCHES: OnceLock<Vec<&'static str>> = OnceLock::new();

fn parse_tune_arches_from_manifest() -> Vec<&'static str> {
  for line in TARGET_MATRIX_MANIFEST.lines() {
    let trimmed = line.trim();
    if !trimmed.starts_with("arches") {
      continue;
    }
    let Some((_, rhs)) = trimmed.split_once('=') else {
      continue;
    };
    let rhs = rhs.trim();
    let rhs = rhs.trim_start_matches('[').trim_end_matches(']');
    let mut arches = Vec::new();
    for token in rhs.split(',') {
      let token = token.trim().trim_matches('"');
      if !token.is_empty() {
        arches.push(token);
      }
    }
    return arches;
  }
  Vec::new()
}

fn tune_arches() -> &'static [&'static str] {
  TUNE_ARCHES.get_or_init(parse_tune_arches_from_manifest).as_slice()
}

#[inline]
#[must_use]
fn blake3_algo_factor(algo: &str) -> f64 {
  match algo {
    "blake3-parent-fold" => 1.15,
    "blake3-parent" => 1.10,
    "blake3-chunk" => 1.05,
    // Streaming (`Hasher::update`) carries materially different overhead than
    // oneshot hashing. Keep explicit stream floors rather than deriving from
    // oneshot with optimistic multipliers.
    "blake3-stream4k" => 0.40,
    "blake3-stream4k-keyed" => 0.38,
    "blake3-stream4k-derive" => 0.36,
    "blake3-stream64" => 0.10,
    "blake3-stream64-keyed" => 0.09,
    "blake3-stream64-derive" => 0.085,
    _ => 1.00,
  }
}

/// Resolve a throughput floor (GiB/s) for an algorithm/architecture/size class.
///
/// Returns `None` when no explicit target is defined.
#[must_use]
pub fn class_target_gib_s(algo: &str, arch: &str, tune_kind: TuneKind, class: &str) -> Option<f64> {
  if !is_blake3_tuning_algo(algo) {
    return None;
  }
  if !tune_arches().contains(&arch) {
    return None;
  }

  let (xs, s, m, l) = match (arch, tune_kind) {
    // x86_64
    ("x86_64", TuneKind::Zen5 | TuneKind::Zen5c) => (1.20, 2.40, 6.00, 10.00),
    ("x86_64", TuneKind::Zen4) => (1.00, 2.00, 5.00, 8.50),
    ("x86_64", TuneKind::IntelGnr | TuneKind::IntelSpr) => (0.95, 1.90, 4.80, 8.00),
    ("x86_64", TuneKind::IntelIcl) => (0.90, 1.80, 4.20, 7.00),
    ("x86_64", TuneKind::Default | TuneKind::Portable) => (0.60, 1.20, 2.50, 4.00),
    // aarch64
    ("aarch64", TuneKind::AppleM5) => (0.90, 1.70, 4.00, 6.80),
    ("aarch64", TuneKind::AppleM4) => (0.85, 1.60, 3.80, 6.20),
    ("aarch64", TuneKind::AppleM1M3) => (0.75, 1.40, 3.30, 5.50),
    ("aarch64", TuneKind::Graviton5 | TuneKind::NeoverseV3) => (0.80, 1.50, 3.60, 5.80),
    ("aarch64", TuneKind::Graviton4 | TuneKind::NeoverseN3 | TuneKind::NvidiaGrace) => (0.75, 1.40, 3.20, 5.20),
    ("aarch64", TuneKind::Graviton3 | TuneKind::NeoverseN2 | TuneKind::AmpereAltra) => (0.65, 1.20, 2.80, 4.60),
    ("aarch64", TuneKind::Graviton2 | TuneKind::Aarch64Pmull) => (0.55, 1.00, 2.20, 3.80),
    ("aarch64", TuneKind::Default | TuneKind::Portable) => (0.50, 0.90, 2.00, 3.50),
    _ => return None,
  };

  let class_base = match class {
    "xs" => xs,
    "s" => s,
    "m" => m,
    "l" => l,
    _ => return None,
  };

  let algo_factor = blake3_algo_factor(algo);

  Some(class_base * algo_factor)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn targets_are_tunekind_and_algo_sensitive() {
    let zen5 = class_target_gib_s("blake3", "x86_64", TuneKind::Zen5, "l").unwrap();
    let icl = class_target_gib_s("blake3", "x86_64", TuneKind::IntelIcl, "l").unwrap();
    assert!(zen5 > icl);

    let stream64 = class_target_gib_s("blake3-stream64", "x86_64", TuneKind::Zen5, "s").unwrap();
    let stream4k = class_target_gib_s("blake3-stream4k", "x86_64", TuneKind::Zen5, "s").unwrap();
    assert!(stream4k > stream64);

    let parent = class_target_gib_s("blake3-parent-fold", "x86_64", TuneKind::Zen5, "m").unwrap();
    let digest = class_target_gib_s("blake3", "x86_64", TuneKind::Zen5, "m").unwrap();
    assert!(parent > digest);
  }

  #[test]
  fn streaming_targets_are_not_oneshot_scaled() {
    let oneshot_l = class_target_gib_s("blake3", "x86_64", TuneKind::IntelSpr, "l").unwrap();
    let stream4k_l = class_target_gib_s("blake3-stream4k", "x86_64", TuneKind::IntelSpr, "l").unwrap();
    let stream64_l = class_target_gib_s("blake3-stream64", "x86_64", TuneKind::IntelSpr, "l").unwrap();

    assert!(stream4k_l < oneshot_l);
    assert!(stream64_l < stream4k_l);
    // Guard against regressing back to the old overly optimistic floor.
    assert!(stream64_l < oneshot_l * 0.2);
  }

  #[test]
  fn tune_arches_match_target_manifest() {
    assert!(tune_arches().contains(&"x86_64"));
    assert!(tune_arches().contains(&"aarch64"));
  }
}

/// A class-level target miss (measured throughput below required floor).
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct TargetMiss {
  pub algo: &'static str,
  pub class: &'static str,
  pub measured_gib_s: f64,
  pub target_gib_s: f64,
}

/// Collect all class-level target misses in this tuning result set.
#[cfg(feature = "std")]
#[must_use]
pub fn collect_target_misses(results: &TuneResults) -> Vec<TargetMiss> {
  let mut misses = Vec::new();
  let arch = results.platform.arch;
  let tune_kind = results.platform.tune_kind;

  for algo in &results.algorithms {
    for class_best in &algo.size_class_best {
      let Some(target) = class_target_gib_s(algo.name, arch, tune_kind, class_best.class) else {
        continue;
      };

      if class_best.throughput_gib_s < target {
        misses.push(TargetMiss {
          algo: algo.name,
          class: class_best.class,
          measured_gib_s: class_best.throughput_gib_s,
          target_gib_s: target,
        });
      }
    }
  }

  misses
}
