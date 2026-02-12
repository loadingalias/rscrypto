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
    let mut arches = Vec::with_capacity(rhs.split(',').count());
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
fn class_index(class: &str) -> Option<usize> {
  Some(match class {
    "xs" => 0,
    "s" => 1,
    "m" => 2,
    "l" => 3,
    _ => return None,
  })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Blake3Surface {
  Oneshot,
  Stream64,
  Stream4k,
  Keyed,
  Derive,
  Xof,
}

#[inline]
#[must_use]
fn blake3_surface(algo: &str) -> Option<Blake3Surface> {
  Some(match algo {
    "blake3" | "blake3-chunk" | "blake3-parent" | "blake3-parent-fold" => Blake3Surface::Oneshot,
    "blake3-stream64"
    | "blake3-stream256"
    | "blake3-stream1k"
    | "blake3-stream-mixed"
    | "blake3-stream64-keyed"
    | "blake3-stream64-derive"
    | "blake3-stream64-xof"
    | "blake3-stream-mixed-xof" => Blake3Surface::Stream64,
    "blake3-stream4k" => Blake3Surface::Stream4k,
    "blake3-stream4k-keyed" => Blake3Surface::Keyed,
    "blake3-stream4k-derive" => Blake3Surface::Derive,
    "blake3-stream4k-xof" => Blake3Surface::Xof,
    _ => return None,
  })
}

#[derive(Clone, Copy)]
struct SurfaceFloors {
  oneshot: [f64; 4],
  stream64: [f64; 4],
  stream4k: [f64; 4],
  keyed: [f64; 4],
  derive: [f64; 4],
  xof: [f64; 4],
}

#[inline]
#[must_use]
fn surface_value(surface: Blake3Surface, idx: usize, floors: SurfaceFloors) -> f64 {
  match surface {
    Blake3Surface::Oneshot => floors.oneshot[idx],
    Blake3Surface::Stream64 => floors.stream64[idx],
    Blake3Surface::Stream4k => floors.stream4k[idx],
    Blake3Surface::Keyed => floors.keyed[idx],
    Blake3Surface::Derive => floors.derive[idx],
    Blake3Surface::Xof => floors.xof[idx],
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
  let surface = blake3_surface(algo)?;
  let idx = class_index(class)?;

  Some(match (arch, tune_kind) {
    // x86_64
    ("x86_64", TuneKind::Zen5 | TuneKind::Zen5c) => surface_value(
      surface,
      idx,
      SurfaceFloors {
        oneshot: [1.20, 2.40, 6.00, 10.00],
        stream64: [0.12, 0.24, 0.60, 1.00],
        stream4k: [0.48, 0.96, 2.40, 4.00],
        keyed: [0.46, 0.91, 2.28, 3.80],
        derive: [0.43, 0.86, 2.14, 3.60],
        xof: [0.40, 0.80, 2.00, 3.30],
      },
    ),
    ("x86_64", TuneKind::Zen4) => surface_value(
      surface,
      idx,
      SurfaceFloors {
        oneshot: [1.00, 2.00, 5.00, 8.50],
        stream64: [0.10, 0.20, 0.50, 0.85],
        stream4k: [0.40, 0.80, 2.00, 3.40],
        keyed: [0.38, 0.76, 1.90, 3.20],
        derive: [0.36, 0.72, 1.80, 3.00],
        xof: [0.34, 0.68, 1.70, 2.80],
      },
    ),
    ("x86_64", TuneKind::IntelGnr | TuneKind::IntelSpr) => surface_value(
      surface,
      idx,
      SurfaceFloors {
        oneshot: [0.95, 1.90, 4.80, 8.00],
        stream64: [0.095, 0.190, 0.480, 0.800],
        stream4k: [0.38, 0.76, 1.92, 3.20],
        keyed: [0.36, 0.72, 1.82, 3.00],
        derive: [0.34, 0.68, 1.72, 2.88],
        xof: [0.32, 0.64, 1.60, 2.64],
      },
    ),
    ("x86_64", TuneKind::IntelIcl) => surface_value(
      surface,
      idx,
      SurfaceFloors {
        oneshot: [0.90, 1.80, 4.20, 7.00],
        stream64: [0.090, 0.180, 0.420, 0.700],
        stream4k: [0.36, 0.72, 1.68, 2.80],
        keyed: [0.34, 0.68, 1.60, 2.62],
        derive: [0.32, 0.64, 1.50, 2.45],
        xof: [0.30, 0.60, 1.40, 2.20],
      },
    ),
    ("x86_64", TuneKind::Default | TuneKind::Portable) => surface_value(
      surface,
      idx,
      SurfaceFloors {
        oneshot: [0.60, 1.20, 2.50, 4.00],
        stream64: [0.060, 0.120, 0.250, 0.400],
        stream4k: [0.24, 0.48, 1.00, 1.60],
        keyed: [0.22, 0.44, 0.90, 1.45],
        derive: [0.20, 0.40, 0.82, 1.30],
        xof: [0.18, 0.36, 0.75, 1.20],
      },
    ),
    // aarch64
    ("aarch64", TuneKind::AppleM5) => surface_value(
      surface,
      idx,
      SurfaceFloors {
        oneshot: [0.90, 1.70, 4.00, 6.80],
        stream64: [0.090, 0.170, 0.400, 0.680],
        stream4k: [0.36, 0.68, 1.60, 2.72],
        keyed: [0.34, 0.64, 1.50, 2.58],
        derive: [0.32, 0.60, 1.40, 2.40],
        xof: [0.30, 0.56, 1.30, 2.20],
      },
    ),
    ("aarch64", TuneKind::AppleM4) => surface_value(
      surface,
      idx,
      SurfaceFloors {
        oneshot: [0.85, 1.60, 3.80, 6.20],
        stream64: [0.085, 0.160, 0.380, 0.620],
        stream4k: [0.34, 0.64, 1.52, 2.48],
        keyed: [0.32, 0.61, 1.44, 2.35],
        derive: [0.30, 0.58, 1.36, 2.22],
        xof: [0.28, 0.54, 1.25, 2.00],
      },
    ),
    ("aarch64", TuneKind::AppleM1M3) => surface_value(
      surface,
      idx,
      SurfaceFloors {
        oneshot: [0.75, 1.40, 3.30, 5.50],
        stream64: [0.075, 0.140, 0.330, 0.550],
        stream4k: [0.30, 0.56, 1.32, 2.20],
        keyed: [0.28, 0.53, 1.24, 2.05],
        derive: [0.26, 0.50, 1.16, 1.92],
        xof: [0.24, 0.46, 1.05, 1.75],
      },
    ),
    ("aarch64", TuneKind::Graviton5 | TuneKind::NeoverseV3) => surface_value(
      surface,
      idx,
      SurfaceFloors {
        oneshot: [0.80, 1.50, 3.60, 5.80],
        stream64: [0.080, 0.150, 0.360, 0.580],
        stream4k: [0.32, 0.60, 1.44, 2.32],
        keyed: [0.30, 0.57, 1.36, 2.18],
        derive: [0.28, 0.54, 1.28, 2.03],
        xof: [0.26, 0.50, 1.20, 1.90],
      },
    ),
    ("aarch64", TuneKind::Graviton4 | TuneKind::NeoverseN3 | TuneKind::NvidiaGrace) => surface_value(
      surface,
      idx,
      SurfaceFloors {
        oneshot: [0.75, 1.40, 3.20, 5.20],
        stream64: [0.075, 0.140, 0.320, 0.520],
        stream4k: [0.30, 0.56, 1.28, 2.08],
        keyed: [0.28, 0.53, 1.20, 1.95],
        derive: [0.26, 0.50, 1.14, 1.82],
        xof: [0.24, 0.46, 1.05, 1.70],
      },
    ),
    ("aarch64", TuneKind::Graviton3 | TuneKind::NeoverseN2 | TuneKind::AmpereAltra) => surface_value(
      surface,
      idx,
      SurfaceFloors {
        oneshot: [0.65, 1.20, 2.80, 4.60],
        stream64: [0.065, 0.120, 0.280, 0.460],
        stream4k: [0.26, 0.48, 1.12, 1.84],
        keyed: [0.24, 0.45, 1.04, 1.72],
        derive: [0.22, 0.42, 0.98, 1.60],
        xof: [0.20, 0.38, 0.90, 1.45],
      },
    ),
    ("aarch64", TuneKind::Graviton2 | TuneKind::Aarch64Pmull) => surface_value(
      surface,
      idx,
      SurfaceFloors {
        oneshot: [0.55, 1.00, 2.20, 3.80],
        stream64: [0.055, 0.100, 0.220, 0.380],
        stream4k: [0.22, 0.40, 0.88, 1.52],
        keyed: [0.20, 0.38, 0.80, 1.40],
        derive: [0.18, 0.35, 0.74, 1.28],
        xof: [0.16, 0.32, 0.66, 1.15],
      },
    ),
    ("aarch64", TuneKind::Default | TuneKind::Portable) => surface_value(
      surface,
      idx,
      SurfaceFloors {
        oneshot: [0.50, 0.90, 2.00, 3.50],
        stream64: [0.050, 0.090, 0.200, 0.350],
        stream4k: [0.20, 0.36, 0.80, 1.40],
        keyed: [0.18, 0.33, 0.74, 1.26],
        derive: [0.16, 0.30, 0.68, 1.12],
        xof: [0.14, 0.27, 0.60, 1.00],
      },
    ),
    _ => return None,
  })
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
  }

  #[test]
  fn keyed_and_derive_are_independent_surfaces() {
    let keyed = class_target_gib_s("blake3-stream4k-keyed", "x86_64", TuneKind::IntelSpr, "l").unwrap();
    let derive = class_target_gib_s("blake3-stream4k-derive", "x86_64", TuneKind::IntelSpr, "l").unwrap();
    let stream4k = class_target_gib_s("blake3-stream4k", "x86_64", TuneKind::IntelSpr, "l").unwrap();
    assert!(keyed < stream4k);
    assert!(derive < keyed);
  }

  #[test]
  fn streaming_targets_are_explicit_and_conservative() {
    let oneshot_l = class_target_gib_s("blake3", "x86_64", TuneKind::IntelSpr, "l").unwrap();
    let stream4k_l = class_target_gib_s("blake3-stream4k", "x86_64", TuneKind::IntelSpr, "l").unwrap();
    let stream64_l = class_target_gib_s("blake3-stream64", "x86_64", TuneKind::IntelSpr, "l").unwrap();

    assert!(stream4k_l < oneshot_l);
    assert!(stream64_l < stream4k_l);
    assert!(stream64_l <= 0.800);
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
  let mut misses = Vec::with_capacity(results.algorithms.len().saturating_mul(4));
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
