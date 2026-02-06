//! Apply `rscrypto-tune` results into dispatch kernel tables.
//!
//! This is the "pipeline" step: a machine can run `rscrypto-tune --apply`,
//! and generated artifacts are emitted for that platform's [`platform::TuneKind`].
//!
//! # Output Format
//!
//! Generates a `KernelTable` entry for `crates/checksum/src/dispatch.rs`:
//!
//! ```rust,ignore
//! pub static PLATFORM_TABLE: KernelTable = KernelTable {
//!   boundaries: [64, 256, 4096],
//!   xs: KernelSet { crc16_ccitt: ..., crc64_xz: ..., ... },
//!   s:  KernelSet { ... },
//!   m:  KernelSet { ... },
//!   l:  KernelSet { ... },
//! };
//! ```

use alloc::collections::BTreeSet;
use std::{
  collections::HashMap,
  fs, io,
  path::{Path, PathBuf},
};

use platform::TuneKind;

use crate::{AlgorithmResult, TuneResults, hash::BLAKE3_TUNING_CORPUS};

// ─────────────────────────────────────────────────────────────────────────────
// Size Class Boundaries
// ─────────────────────────────────────────────────────────────────────────────

/// Size class byte thresholds (must match dispatch.rs).
const SIZE_CLASS_XS_MAX: usize = 64;
const SIZE_CLASS_S_MAX: usize = 256;
const SIZE_CLASS_M_MAX: usize = 4096;

/// Size classes used in tuning and dispatch.
const SIZE_CLASSES: [&str; 4] = ["xs", "s", "m", "l"];

#[derive(Clone, Debug)]
struct GeneratedArtifact {
  path: PathBuf,
  contents: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Expression Mapping (tune results → dispatch.rs initializers)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct KernelChoice {
  base: String,
  streams: u8,
}

#[derive(Clone, Debug)]
struct KernelExpr {
  func: String,
  name: String,
}

#[derive(Clone, Copy, Debug)]
enum CrcVariant {
  Crc16Ccitt,
  Crc16Ibm,
  Crc24OpenPgp,
  Crc32Ieee,
  Crc32c,
  Crc64Xz,
  Crc64Nvme,
}

impl CrcVariant {
  fn module_ident(self) -> &'static str {
    match self {
      Self::Crc16Ccitt | Self::Crc16Ibm => "crc16_k",
      Self::Crc24OpenPgp => "crc24_k",
      Self::Crc32Ieee | Self::Crc32c => "crc32_k",
      Self::Crc64Xz | Self::Crc64Nvme => "crc64_k",
    }
  }

  fn const_prefix(self) -> &'static str {
    match self {
      Self::Crc16Ccitt => "CCITT",
      Self::Crc16Ibm => "IBM",
      Self::Crc24OpenPgp => "OPENPGP",
      Self::Crc32Ieee => "CRC32",
      Self::Crc32c => "CRC32C",
      Self::Crc64Xz => "XZ",
      Self::Crc64Nvme => "NVME",
    }
  }
}

/// Stream count → array index mapping used by checksum kernel arrays.
///
/// Arrays are ordered `[1-way, 2-way, 3/4-way, 7-way, 8-way]`, with duplicates
/// for architectures that don't implement all widths.
#[inline]
#[must_use]
const fn stream_to_index(streams: u8) -> usize {
  match streams {
    8 => 4,
    7 => 3,
    3 | 4 => 2,
    2 => 1,
    _ => 0,
  }
}

fn parse_arch_and_impl(base: &str) -> io::Result<(&str, &str)> {
  let (arch, imp) = base.split_once('/').ok_or_else(|| {
    io::Error::new(
      io::ErrorKind::InvalidData,
      format!("invalid kernel name (expected arch/impl): {base}"),
    )
  })?;
  Ok((arch, imp))
}

fn canonical_kernel_name(base: &str, streams: u8) -> String {
  if base == "reference" || base.starts_with("reference/") {
    return "reference".to_string();
  }
  if base == "portable" || base.starts_with("portable/") {
    return base.to_string();
  }
  if base.ends_with("-small") {
    return base.to_string();
  }
  if streams <= 1 || base.contains("-way") {
    return base.to_string();
  }
  format!("{base}-{streams}way")
}

fn family_const_suffix(imp: &str) -> Option<&'static str> {
  if imp == "vpclmul-4x512" {
    return Some("VPCLMUL_4X512");
  }
  let imp = imp.strip_suffix("-small").unwrap_or(imp);
  if imp.starts_with("sve2-pmull") {
    return Some("SVE2_PMULL");
  }
  if imp.starts_with("pmull-eor3") {
    return Some("PMULL_EOR3");
  }
  if imp.starts_with("pmull") {
    return Some("PMULL");
  }
  if imp.starts_with("fusion-vpclmul") {
    return Some("FUSION_VPCLMUL");
  }
  if imp.starts_with("fusion-avx512") {
    return Some("FUSION_AVX512");
  }
  if imp.starts_with("fusion-sse") {
    return Some("FUSION_SSE");
  }
  if imp.starts_with("vpclmul") {
    return Some("VPCLMUL");
  }
  if imp.starts_with("pclmul") {
    return Some("PCLMUL");
  }
  if imp.starts_with("hwcrc") {
    return Some("HWCRC");
  }
  if imp.starts_with("vpmsum") {
    return Some("VPMSUM");
  }
  if imp.starts_with("vgfm") {
    return Some("VGFM");
  }
  if imp.starts_with("zvbc") {
    return Some("ZVBC");
  }
  if imp.starts_with("zbc") {
    return Some("ZBC");
  }
  None
}

fn portable_kernel_expr(variant: CrcVariant, base: &str) -> io::Result<KernelExpr> {
  let func = match (variant, base) {
    (CrcVariant::Crc16Ccitt, "portable/slice4") => "crate::crc16::portable::crc16_ccitt_slice4",
    (CrcVariant::Crc16Ccitt, "portable/slice8") => "crate::crc16::portable::crc16_ccitt_slice8",
    (CrcVariant::Crc16Ibm, "portable/slice4") => "crate::crc16::portable::crc16_ibm_slice4",
    (CrcVariant::Crc16Ibm, "portable/slice8") => "crate::crc16::portable::crc16_ibm_slice8",
    (CrcVariant::Crc24OpenPgp, "portable/slice4") => "crate::crc24::portable::crc24_openpgp_slice4",
    (CrcVariant::Crc24OpenPgp, "portable/slice8") => "crate::crc24::portable::crc24_openpgp_slice8",
    (CrcVariant::Crc32Ieee, "portable/bytewise") => "crate::crc32::portable::crc32_bytewise_ieee",
    (CrcVariant::Crc32Ieee, "portable/slice16") => "crate::crc32::portable::crc32_slice16_ieee",
    (CrcVariant::Crc32c, "portable/bytewise") => "crate::crc32::portable::crc32c_bytewise",
    (CrcVariant::Crc32c, "portable/slice16") => "crate::crc32::portable::crc32c_slice16",
    (CrcVariant::Crc64Xz, "portable/slice16") => "crate::crc64::portable::crc64_slice16_xz",
    (CrcVariant::Crc64Nvme, "portable/slice16") => "crate::crc64::portable::crc64_slice16_nvme",
    // Fallbacks (older results / partial runs).
    (CrcVariant::Crc16Ccitt, "portable") => "crate::crc16::portable::crc16_ccitt_slice8",
    (CrcVariant::Crc16Ibm, "portable") => "crate::crc16::portable::crc16_ibm_slice8",
    (CrcVariant::Crc24OpenPgp, "portable") => "crate::crc24::portable::crc24_openpgp_slice8",
    (CrcVariant::Crc32Ieee, "portable") => "crate::crc32::portable::crc32_slice16_ieee",
    (CrcVariant::Crc32c, "portable") => "crate::crc32::portable::crc32c_slice16",
    (CrcVariant::Crc64Xz, "portable") => "crate::crc64::portable::crc64_slice16_xz",
    (CrcVariant::Crc64Nvme, "portable") => "crate::crc64::portable::crc64_slice16_nvme",
    _ => {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("unsupported portable kernel for {variant:?}: {base}"),
      ));
    }
  };

  let name = match (variant, base) {
    (CrcVariant::Crc16Ccitt | CrcVariant::Crc16Ibm | CrcVariant::Crc24OpenPgp, "portable") => "portable/slice8",
    (CrcVariant::Crc32Ieee | CrcVariant::Crc32c | CrcVariant::Crc64Xz | CrcVariant::Crc64Nvme, "portable") => {
      "portable/slice16"
    }
    _ => base,
  };

  Ok(KernelExpr {
    func: func.to_string(),
    name: format!("\"{name}\""),
  })
}

fn kernel_expr(variant: CrcVariant, base: &str, streams: u8) -> io::Result<KernelExpr> {
  if base == "portable" || base.starts_with("portable/") {
    return portable_kernel_expr(variant, base);
  }

  // We intentionally do not allow reference kernels in dispatch tables; they are
  // for verification and are not available as stable, addressable symbols from
  // `dispatch.rs`. If tuning ever selects one, treat it as an analysis bug.
  if base == "reference" || base.starts_with("reference/") {
    return Err(io::Error::new(
      io::ErrorKind::InvalidData,
      format!("reference kernel selected for {variant:?}; dispatch tables must not use reference kernels"),
    ));
  }

  let (_arch, imp) = parse_arch_and_impl(base)?;
  let Some(family) = family_const_suffix(imp) else {
    return Err(io::Error::new(
      io::ErrorKind::InvalidData,
      format!("unrecognized kernel family for {variant:?}: {base}"),
    ));
  };

  let module = variant.module_ident();
  let prefix = variant.const_prefix();

  // Special case: x86_64 VPCLMUL 4×512-bit CRC64 kernels.
  if matches!(variant, CrcVariant::Crc64Xz | CrcVariant::Crc64Nvme) && family == "VPCLMUL_4X512" {
    let func = format!("{module}::{prefix}_{family}");
    return Ok(KernelExpr {
      func,
      name: format!("\"{}\"", canonical_kernel_name(base, 1)),
    });
  }

  // Small kernel constants.
  if imp.ends_with("-small") {
    let func = if matches!(variant, CrcVariant::Crc64Xz | CrcVariant::Crc64Nvme) {
      format!("{module}::{prefix}_{family}_SMALL")
    } else {
      format!("{module}::{prefix}_{family}_SMALL_KERNEL")
    };
    return Ok(KernelExpr {
      func,
      name: format!("\"{}\"", canonical_kernel_name(base, 1)),
    });
  }

  // Streamed kernel array selection.
  let idx = stream_to_index(streams);
  let func = format!("{module}::{prefix}_{family}[{idx}]");
  Ok(KernelExpr {
    func,
    name: format!("\"{}\"", canonical_kernel_name(base, streams)),
  })
}

fn best_by_size_class(algo: &AlgorithmResult) -> HashMap<&'static str, KernelChoice> {
  let mut best: HashMap<&'static str, KernelChoice> = HashMap::new();

  for entry in &algo.size_class_best {
    best.insert(
      entry.class,
      KernelChoice {
        base: entry.kernel.clone(),
        streams: entry.streams,
      },
    );
  }

  // Fallback (older results / partial runs).
  if best.is_empty() {
    for class in SIZE_CLASSES {
      best.insert(
        class,
        KernelChoice {
          base: algo.best_kernel.to_string(),
          streams: algo.recommended_streams,
        },
      );
    }
  } else {
    for class in SIZE_CLASSES {
      best.entry(class).or_insert_with(|| KernelChoice {
        base: algo.best_kernel.to_string(),
        streams: algo.recommended_streams,
      });
    }
  }

  best
}

fn cap_terms_for_kernel(variant: CrcVariant, base: &str) -> io::Result<Vec<&'static str>> {
  if base == "portable" || base.starts_with("portable/") {
    return Ok(vec![]);
  }
  if base == "reference" || base.starts_with("reference/") {
    return Ok(vec![]);
  }

  let (arch, imp) = parse_arch_and_impl(base)?;
  let imp = imp.strip_suffix("-small").unwrap_or(imp);

  let terms = match arch {
    "x86_64" => {
      if imp.starts_with("pclmul") {
        vec!["platform::caps::x86::PCLMUL_READY"]
      } else if imp.starts_with("vpclmul") {
        vec!["platform::caps::x86::VPCLMUL_READY"]
      } else if imp.starts_with("hwcrc") {
        vec!["platform::caps::x86::CRC32C_READY"]
      } else if imp.starts_with("fusion-sse") {
        vec!["platform::caps::x86::CRC32C_READY", "platform::caps::x86::PCLMUL_READY"]
      } else if imp.starts_with("fusion-avx512") {
        vec![
          "platform::caps::x86::CRC32C_READY",
          "platform::caps::x86::PCLMUL_READY",
          "platform::caps::x86::AVX512_READY",
        ]
      } else if imp.starts_with("fusion-vpclmul") {
        vec![
          "platform::caps::x86::CRC32C_READY",
          "platform::caps::x86::VPCLMUL_READY",
        ]
      } else {
        vec![]
      }
    }
    "aarch64" => {
      if imp.starts_with("hwcrc") {
        vec!["platform::caps::aarch64::CRC_READY"]
      } else if imp.starts_with("pmull-eor3") {
        let mut t = vec!["platform::caps::aarch64::PMULL_EOR3_READY"];
        if matches!(variant, CrcVariant::Crc32Ieee | CrcVariant::Crc32c) {
          t.push("platform::caps::aarch64::CRC_READY");
        }
        t
      } else if imp.starts_with("sve2-pmull") {
        let mut t = vec![
          "platform::caps::aarch64::SVE2_PMULL",
          "platform::caps::aarch64::PMULL_READY",
        ];
        if matches!(variant, CrcVariant::Crc32Ieee | CrcVariant::Crc32c) {
          t.push("platform::caps::aarch64::CRC_READY");
        }
        t
      } else if imp.starts_with("pmull") {
        let mut t = vec!["platform::caps::aarch64::PMULL_READY"];
        if matches!(variant, CrcVariant::Crc32Ieee | CrcVariant::Crc32c) {
          t.push("platform::caps::aarch64::CRC_READY");
        }
        t
      } else {
        vec![]
      }
    }
    "power" => {
      if imp.starts_with("vpmsum") {
        vec!["platform::caps::power::VPMSUM_READY"]
      } else {
        vec![]
      }
    }
    "s390x" => {
      if imp.starts_with("vgfm") {
        vec!["platform::caps::s390x::VECTOR"]
      } else {
        vec![]
      }
    }
    "riscv64" => {
      if imp.starts_with("zvbc") {
        vec!["platform::caps::riscv::V", "platform::caps::riscv::ZVBC"]
      } else if imp.starts_with("zbc") {
        vec!["platform::caps::riscv::ZBC"]
      } else {
        vec![]
      }
    }
    _ => vec![],
  };

  Ok(terms)
}

fn requires_expr_for_crc_table(
  ccitt: &HashMap<&'static str, KernelChoice>,
  ibm: &HashMap<&'static str, KernelChoice>,
  openpgp: &HashMap<&'static str, KernelChoice>,
  crc32: &HashMap<&'static str, KernelChoice>,
  crc32c: &HashMap<&'static str, KernelChoice>,
  xz: &HashMap<&'static str, KernelChoice>,
  nvme: &HashMap<&'static str, KernelChoice>,
) -> io::Result<String> {
  let mut terms: BTreeSet<&'static str> = BTreeSet::new();

  let add = |variant: CrcVariant, choice: &KernelChoice, terms: &mut BTreeSet<&'static str>| -> io::Result<()> {
    for t in cap_terms_for_kernel(variant, &choice.base)? {
      terms.insert(t);
    }
    Ok(())
  };

  for class in SIZE_CLASSES {
    add(
      CrcVariant::Crc16Ccitt,
      ccitt.get(class).expect("missing class"),
      &mut terms,
    )?;
    add(CrcVariant::Crc16Ibm, ibm.get(class).expect("missing class"), &mut terms)?;
    add(
      CrcVariant::Crc24OpenPgp,
      openpgp.get(class).expect("missing class"),
      &mut terms,
    )?;
    add(
      CrcVariant::Crc32Ieee,
      crc32.get(class).expect("missing class"),
      &mut terms,
    )?;
    add(
      CrcVariant::Crc32c,
      crc32c.get(class).expect("missing class"),
      &mut terms,
    )?;
    add(CrcVariant::Crc64Xz, xz.get(class).expect("missing class"), &mut terms)?;
    add(
      CrcVariant::Crc64Nvme,
      nvme.get(class).expect("missing class"),
      &mut terms,
    )?;
  }

  if terms.is_empty() {
    return Ok("Caps::NONE".to_string());
  }

  let mut iter = terms.into_iter();
  let first = iter.next().unwrap();
  let mut expr = first.to_string();
  for t in iter {
    expr.push_str(".union(");
    expr.push_str(t);
    expr.push(')');
  }
  Ok(expr)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch Table Generation
// ─────────────────────────────────────────────────────────────────────────────

/// Algorithm set for building a kernel table.
struct AlgoSet<'a> {
  crc16_ccitt: &'a AlgorithmResult,
  crc16_ibm: &'a AlgorithmResult,
  crc24_openpgp: &'a AlgorithmResult,
  crc32_ieee: &'a AlgorithmResult,
  crc32c: &'a AlgorithmResult,
  crc64_xz: &'a AlgorithmResult,
  crc64_nvme: &'a AlgorithmResult,
}

struct CrcBestMaps<'a> {
  ccitt: &'a HashMap<&'static str, KernelChoice>,
  ibm: &'a HashMap<&'static str, KernelChoice>,
  openpgp: &'a HashMap<&'static str, KernelChoice>,
  crc32: &'a HashMap<&'static str, KernelChoice>,
  crc32c: &'a HashMap<&'static str, KernelChoice>,
  xz: &'a HashMap<&'static str, KernelChoice>,
  nvme: &'a HashMap<&'static str, KernelChoice>,
}

const CRC_ALGO_NAMES: &[&str] = &[
  "crc16-ccitt",
  "crc16-ibm",
  "crc24-openpgp",
  "crc32-ieee",
  "crc32c",
  "crc64-xz",
  "crc64-nvme",
];

fn algos(results: &TuneResults) -> Result<AlgoSet<'_>, io::Error> {
  fn find<'a>(results: &'a TuneResults, name: &str) -> Option<&'a AlgorithmResult> {
    results.algorithms.iter().find(|a| a.name == name)
  }

  let crc16_ccitt = find(results, "crc16-ccitt").ok_or_else(|| missing("crc16-ccitt"))?;
  let crc16_ibm = find(results, "crc16-ibm").ok_or_else(|| missing("crc16-ibm"))?;
  let crc24_openpgp = find(results, "crc24-openpgp").ok_or_else(|| missing("crc24-openpgp"))?;
  let crc32_ieee = find(results, "crc32-ieee").ok_or_else(|| missing("crc32-ieee"))?;
  let crc32c = find(results, "crc32c").ok_or_else(|| missing("crc32c"))?;
  let crc64_xz = find(results, "crc64-xz").ok_or_else(|| missing("crc64-xz"))?;
  let crc64_nvme = find(results, "crc64-nvme").ok_or_else(|| missing("crc64-nvme"))?;

  Ok(AlgoSet {
    crc16_ccitt,
    crc16_ibm,
    crc24_openpgp,
    crc32_ieee,
    crc32c,
    crc64_xz,
    crc64_nvme,
  })
}

fn missing(name: &str) -> io::Error {
  io::Error::new(io::ErrorKind::InvalidData, format!("missing algorithm result: {name}"))
}

fn checksum_table_ident(tune_kind: TuneKind) -> Option<&'static str> {
  // Keep this in sync with `checksum::dispatch::{exact_match,family_match,capability_match}`.
  Some(match tune_kind {
    TuneKind::Portable => "PORTABLE_TABLE",

    // aarch64 exact + family matches
    TuneKind::AppleM1M3 | TuneKind::AppleM4 | TuneKind::AppleM5 => "APPLE_M1M3_TABLE",
    TuneKind::Graviton2 => "GRAVITON2_TABLE",
    TuneKind::Graviton3
    | TuneKind::Graviton4
    | TuneKind::Graviton5
    | TuneKind::NeoverseN2
    | TuneKind::NeoverseN3
    | TuneKind::NeoverseV3
    | TuneKind::NvidiaGrace
    | TuneKind::AmpereAltra => "GRAVITON3_TABLE",
    TuneKind::Aarch64Pmull => "GENERIC_ARM_PMULL_TABLE",

    // x86_64 exact + family matches
    TuneKind::Zen4 => "ZEN4_TABLE",
    TuneKind::Zen5 | TuneKind::Zen5c => "ZEN5_TABLE",
    TuneKind::IntelSpr | TuneKind::IntelGnr => "INTEL_SPR_TABLE",
    TuneKind::IntelIcl => "GENERIC_X86_VPCLMUL_TABLE",

    // s390x exact matches
    TuneKind::Z13 => "S390X_Z13_TABLE",
    TuneKind::Z14 => "S390X_Z14_TABLE",
    TuneKind::Z15 => "S390X_Z15_TABLE",

    // power exact matches (Power7 maps to the conservative POWER8 table)
    TuneKind::Power7 | TuneKind::Power8 => "POWER8_TABLE",
    TuneKind::Power9 => "POWER9_TABLE",
    TuneKind::Power10 => "POWER10_TABLE",

    // Default/Custom are policy-driven and do not have a unique checksum table.
    TuneKind::Custom | TuneKind::Default => return None,
  })
}

fn should_apply_checksum_dispatch(results: &TuneResults) -> bool {
  results.algorithms.iter().any(|a| CRC_ALGO_NAMES.contains(&a.name))
}

fn generated_dir(repo_root: &Path) -> PathBuf {
  repo_root.join("crates/tune/generated")
}

fn generated_apply_path(repo_root: &Path, family: &str, name: &str, tune_kind: TuneKind) -> PathBuf {
  generated_dir(repo_root)
    .join(family)
    .join(name)
    .join(format!("{tune_kind:?}.rs").to_lowercase())
}

fn apply_checksum_dispatch_tables(repo_root: &Path, results: &TuneResults) -> io::Result<Vec<GeneratedArtifact>> {
  let tune_kind = results.platform.tune_kind;

  let missing: Vec<&'static str> = CRC_ALGO_NAMES
    .iter()
    .copied()
    .filter(|name| !results.algorithms.iter().any(|a| a.name == *name))
    .collect();
  if !missing.is_empty() {
    return Err(io::Error::new(
      io::ErrorKind::InvalidData,
      format!("incomplete CRC tuning results; missing: {}", missing.join(", ")),
    ));
  }

  let a = algos(results)?;
  let table_code = generate_kernel_table(tune_kind, &a)?;
  let table_ident = checksum_table_ident(tune_kind).ok_or_else(|| {
    io::Error::new(
      io::ErrorKind::InvalidData,
      format!("no checksum dispatch table identifier known for TuneKind::{tune_kind:?}"),
    )
  })?;

  let output_path = generated_apply_path(repo_root, "checksum", table_ident, tune_kind);
  Ok(vec![GeneratedArtifact {
    path: output_path,
    contents: table_code,
  }])
}

/// Generate a KernelSet entry for one size class.
fn generate_kernel_set(size_class: &'static str, best: &CrcBestMaps<'_>, indent: &str) -> io::Result<String> {
  let ccitt_k = best.ccitt.get(size_class).ok_or_else(|| missing("crc16-ccitt"))?;
  let ibm_k = best.ibm.get(size_class).ok_or_else(|| missing("crc16-ibm"))?;
  let openpgp_k = best.openpgp.get(size_class).ok_or_else(|| missing("crc24-openpgp"))?;
  let crc32_k = best.crc32.get(size_class).ok_or_else(|| missing("crc32-ieee"))?;
  let crc32c_k = best.crc32c.get(size_class).ok_or_else(|| missing("crc32c"))?;
  let xz_k = best.xz.get(size_class).ok_or_else(|| missing("crc64-xz"))?;
  let nvme_k = best.nvme.get(size_class).ok_or_else(|| missing("crc64-nvme"))?;

  let ccitt_e = kernel_expr(CrcVariant::Crc16Ccitt, &ccitt_k.base, ccitt_k.streams)?;
  let ibm_e = kernel_expr(CrcVariant::Crc16Ibm, &ibm_k.base, ibm_k.streams)?;
  let openpgp_e = kernel_expr(CrcVariant::Crc24OpenPgp, &openpgp_k.base, openpgp_k.streams)?;
  let crc32_e = kernel_expr(CrcVariant::Crc32Ieee, &crc32_k.base, crc32_k.streams)?;
  let crc32c_e = kernel_expr(CrcVariant::Crc32c, &crc32c_k.base, crc32c_k.streams)?;
  let xz_e = kernel_expr(CrcVariant::Crc64Xz, &xz_k.base, xz_k.streams)?;
  let nvme_e = kernel_expr(CrcVariant::Crc64Nvme, &nvme_k.base, nvme_k.streams)?;

  Ok(format!(
    "{indent}KernelSet {{\n{indent}  crc16_ccitt: {ccitt_func},\n{indent}  crc16_ccitt_name: {ccitt_name},\n{indent}  \
     crc16_ibm: {ibm_func},\n{indent}  crc16_ibm_name: {ibm_name},\n{indent}  crc24_openpgp: \
     {openpgp_func},\n{indent}  crc24_openpgp_name: {openpgp_name},\n{indent}  crc32_ieee: {crc32_func},\n{indent}  \
     crc32_ieee_name: {crc32_name},\n{indent}  crc32c: {crc32c_func},\n{indent}  crc32c_name: \
     {crc32c_name},\n{indent}  crc64_xz: {xz_func},\n{indent}  crc64_xz_name: {xz_name},\n{indent}  crc64_nvme: \
     {nvme_func},\n{indent}  crc64_nvme_name: {nvme_name},\n{indent}}}",
    ccitt_func = ccitt_e.func,
    ccitt_name = ccitt_e.name,
    ibm_func = ibm_e.func,
    ibm_name = ibm_e.name,
    openpgp_func = openpgp_e.func,
    openpgp_name = openpgp_e.name,
    crc32_func = crc32_e.func,
    crc32_name = crc32_e.name,
    crc32c_func = crc32c_e.func,
    crc32c_name = crc32c_e.name,
    xz_func = xz_e.func,
    xz_name = xz_e.name,
    nvme_func = nvme_e.func,
    nvme_name = nvme_e.name,
  ))
}

/// Generate a complete KernelTable entry.
fn generate_kernel_table(tune_kind: TuneKind, algos: &AlgoSet<'_>) -> io::Result<String> {
  let ccitt = best_by_size_class(algos.crc16_ccitt);
  let ibm = best_by_size_class(algos.crc16_ibm);
  let openpgp = best_by_size_class(algos.crc24_openpgp);
  let crc32 = best_by_size_class(algos.crc32_ieee);
  let crc32c = best_by_size_class(algos.crc32c);
  let xz = best_by_size_class(algos.crc64_xz);
  let nvme = best_by_size_class(algos.crc64_nvme);
  let best = CrcBestMaps {
    ccitt: &ccitt,
    ibm: &ibm,
    openpgp: &openpgp,
    crc32: &crc32,
    crc32c: &crc32c,
    xz: &xz,
    nvme: &nvme,
  };

  let kind_name = format!("{tune_kind:?}");
  let table_name = checksum_table_ident(tune_kind).ok_or_else(|| {
    io::Error::new(
      io::ErrorKind::InvalidData,
      format!("no checksum dispatch table identifier known for TuneKind::{kind_name}"),
    )
  })?;

  let requires = requires_expr_for_crc_table(&ccitt, &ibm, &openpgp, &crc32, &crc32c, &xz, &nvme)?;

  let xs_set = generate_kernel_set("xs", &best, "    ")?;
  let s_set = generate_kernel_set("s", &best, "    ")?;
  let m_set = generate_kernel_set("m", &best, "    ")?;
  let l_set = generate_kernel_set("l", &best, "    ")?;

  Ok(format!(
    "  // ───────────────────────────────────────────────────────────────────────────
  // {kind_name} Table
  //
  // Generated by rscrypto-tune. Do not edit manually.
  // ───────────────────────────────────────────────────────────────────────────
  pub static {table_name}: KernelTable = KernelTable {{
    requires: {requires},
    boundaries: [{SIZE_CLASS_XS_MAX}, {SIZE_CLASS_S_MAX}, {SIZE_CLASS_M_MAX}],

    xs: {xs_set},

    s: {s_set},

    m: {m_set},

    l: {l_set},
  }};
"
  ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Generated Artifact Mapping
// ─────────────────────────────────────────────────────────────────────────────

struct HashDispatchTarget {
  algo: &'static str,
  aliases: &'static [&'static str],
}

const HASH_DISPATCH_TARGETS: &[HashDispatchTarget] = &[
  HashDispatchTarget {
    algo: "sha224-compress",
    aliases: &["sha224"],
  },
  HashDispatchTarget {
    algo: "sha256-compress",
    aliases: &["sha256"],
  },
  HashDispatchTarget {
    algo: "sha384-compress",
    aliases: &["sha384"],
  },
  HashDispatchTarget {
    algo: "sha512-compress",
    aliases: &["sha512"],
  },
  HashDispatchTarget {
    algo: "sha512-224-compress",
    aliases: &["sha512-224"],
  },
  HashDispatchTarget {
    algo: "sha512-256-compress",
    aliases: &["sha512-256"],
  },
  HashDispatchTarget {
    algo: "blake2b-512-compress",
    aliases: &["blake2b-512"],
  },
  HashDispatchTarget {
    algo: "blake2s-256-compress",
    aliases: &["blake2s-256"],
  },
  HashDispatchTarget {
    algo: "blake3-chunk",
    aliases: &["blake3"],
  },
  HashDispatchTarget {
    algo: "xxh3",
    aliases: &[],
  },
  HashDispatchTarget {
    algo: "rapidhash",
    aliases: &[],
  },
  HashDispatchTarget {
    algo: "siphash",
    aliases: &[],
  },
  HashDispatchTarget {
    algo: "keccakf1600-permute",
    aliases: &[
      "keccakf1600",
      "sha3-224",
      "sha3-256",
      "sha3-384",
      "sha3-512",
      "shake128",
      "shake256",
    ],
  },
  HashDispatchTarget {
    algo: "ascon-hash256",
    aliases: &["ascon-xof128"],
  },
];

fn find_hash_algo<'a>(results: &'a TuneResults, target: &HashDispatchTarget) -> Option<&'a AlgorithmResult> {
  let primary = results.algorithms.iter().find(|a| a.name == target.algo);
  if primary.is_some() {
    return primary;
  }

  target
    .aliases
    .iter()
    .find_map(|&name| results.algorithms.iter().find(|a| a.name == name))
}

fn write_artifact(path: &Path, contents: &str) -> io::Result<()> {
  if let Some(parent) = path.parent() {
    fs::create_dir_all(parent)?;
  }

  let mut temp_path = path.to_path_buf();
  let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("tmp");
  temp_path.set_extension(format!("{ext}.rscrypto-tmp"));
  fs::write(&temp_path, contents)?;
  fs::rename(&temp_path, path)?;
  Ok(())
}

fn rollback_artifacts(backups: &[(PathBuf, Option<Vec<u8>>)]) {
  for (path, backup) in backups.iter().rev() {
    match backup {
      Some(bytes) => {
        if let Some(parent) = path.parent() {
          let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(path, bytes);
      }
      None => {
        let _ = fs::remove_file(path);
      }
    }
  }
}

fn write_artifacts_transactional(artifacts: &[GeneratedArtifact]) -> io::Result<()> {
  let mut backups: Vec<(PathBuf, Option<Vec<u8>>)> = Vec::with_capacity(artifacts.len());

  for artifact in artifacts {
    let previous = fs::read(&artifact.path).ok();
    backups.push((artifact.path.clone(), previous));

    if let Err(err) = write_artifact(&artifact.path, &artifact.contents) {
      rollback_artifacts(&backups);
      return Err(io::Error::new(
        err.kind(),
        format!(
          "failed to write generated artifact '{}': {err}",
          artifact.path.display()
        ),
      ));
    }
  }

  for artifact in artifacts {
    eprintln!("Generated: {}", artifact.path.display());
  }

  Ok(())
}

fn tune_kind_table_ident(tune_kind: TuneKind) -> &'static str {
  match tune_kind {
    TuneKind::Custom => "CUSTOM_TABLE",
    TuneKind::Default => "DEFAULT_KIND_TABLE",
    TuneKind::Portable => "PORTABLE_TABLE",
    TuneKind::Zen4 => "ZEN4_TABLE",
    TuneKind::Zen5 => "ZEN5_TABLE",
    TuneKind::Zen5c => "ZEN5C_TABLE",
    TuneKind::IntelSpr => "INTELSPR_TABLE",
    TuneKind::IntelGnr => "INTELGNR_TABLE",
    TuneKind::IntelIcl => "INTELICL_TABLE",
    TuneKind::AppleM1M3 => "APPLEM1M3_TABLE",
    TuneKind::AppleM4 => "APPLEM4_TABLE",
    TuneKind::AppleM5 => "APPLEM5_TABLE",
    TuneKind::Graviton2 => "GRAVITON2_TABLE",
    TuneKind::Graviton3 => "GRAVITON3_TABLE",
    TuneKind::Graviton4 => "GRAVITON4_TABLE",
    TuneKind::Graviton5 => "GRAVITON5_TABLE",
    TuneKind::NeoverseN2 => "NEOVERSEN2_TABLE",
    TuneKind::NeoverseN3 => "NEOVERSEN3_TABLE",
    TuneKind::NeoverseV3 => "NEOVERSEV3_TABLE",
    TuneKind::NvidiaGrace => "NVIDIAGRACE_TABLE",
    TuneKind::AmpereAltra => "AMPEREALTRA_TABLE",
    TuneKind::Aarch64Pmull => "AARCH64PMULL_TABLE",
    TuneKind::Z13 => "Z13_TABLE",
    TuneKind::Z14 => "Z14_TABLE",
    TuneKind::Z15 => "Z15_TABLE",
    TuneKind::Power7 => "POWER7_TABLE",
    TuneKind::Power8 => "POWER8_TABLE",
    TuneKind::Power9 => "POWER9_TABLE",
    TuneKind::Power10 => "POWER10_TABLE",
  }
}

#[cfg(test)]
fn tune_kind_streaming_table_ident(tune_kind: TuneKind) -> &'static str {
  match tune_kind {
    TuneKind::Custom => "CUSTOM_STREAMING_TABLE",
    TuneKind::Default => "DEFAULT_KIND_STREAMING_TABLE",
    TuneKind::Portable => "PORTABLE_STREAMING_TABLE",
    TuneKind::Zen4 => "ZEN4_STREAMING_TABLE",
    TuneKind::Zen5 => "ZEN5_STREAMING_TABLE",
    TuneKind::Zen5c => "ZEN5C_STREAMING_TABLE",
    TuneKind::IntelSpr => "INTELSPR_STREAMING_TABLE",
    TuneKind::IntelGnr => "INTELGNR_STREAMING_TABLE",
    TuneKind::IntelIcl => "INTELICL_STREAMING_TABLE",
    TuneKind::AppleM1M3 => "APPLEM1M3_STREAMING_TABLE",
    TuneKind::AppleM4 => "APPLEM4_STREAMING_TABLE",
    TuneKind::AppleM5 => "APPLEM5_STREAMING_TABLE",
    TuneKind::Graviton2 => "GRAVITON2_STREAMING_TABLE",
    TuneKind::Graviton3 => "GRAVITON3_STREAMING_TABLE",
    TuneKind::Graviton4 => "GRAVITON4_STREAMING_TABLE",
    TuneKind::Graviton5 => "GRAVITON5_STREAMING_TABLE",
    TuneKind::NeoverseN2 => "NEOVERSEN2_STREAMING_TABLE",
    TuneKind::NeoverseN3 => "NEOVERSEN3_STREAMING_TABLE",
    TuneKind::NeoverseV3 => "NEOVERSEV3_STREAMING_TABLE",
    TuneKind::NvidiaGrace => "NVIDIAGRACE_STREAMING_TABLE",
    TuneKind::AmpereAltra => "AMPEREALTRA_STREAMING_TABLE",
    TuneKind::Aarch64Pmull => "AARCH64PMULL_STREAMING_TABLE",
    TuneKind::Z13 => "Z13_STREAMING_TABLE",
    TuneKind::Z14 => "Z14_STREAMING_TABLE",
    TuneKind::Z15 => "Z15_STREAMING_TABLE",
    TuneKind::Power7 => "POWER7_STREAMING_TABLE",
    TuneKind::Power8 => "POWER8_STREAMING_TABLE",
    TuneKind::Power9 => "POWER9_STREAMING_TABLE",
    TuneKind::Power10 => "POWER10_STREAMING_TABLE",
  }
}

fn hash_kernel_expr(algo: &str, kernel_name: &str) -> &'static str {
  match algo {
    // BLAKE3 dispatch tables select a full kernel bundle (compress + chunk + parent + hash_many).
    "blake3-chunk" | "blake3" => match kernel_name {
      "portable" => "KernelId::Portable",
      // Runtime x86 hierarchy is AVX-512 > AVX2 > SSE4.1 > scalar.
      "x86_64/ssse3" => "KernelId::X86Sse41",
      "x86_64/sse4.1" => "KernelId::X86Sse41",
      "x86_64/avx2" => "KernelId::X86Avx2",
      "x86_64/avx512" => "KernelId::X86Avx512",
      "aarch64/neon" => "KernelId::Aarch64Neon",
      _ => "KernelId::Portable",
    },
    _ => match kernel_name {
      "portable" => "KernelId::Portable",
      _ => "KernelId::Portable",
    },
  }
}

fn find_threshold_value(algo: &AlgorithmResult, keys: &[&str]) -> Option<usize> {
  algo
    .thresholds
    .iter()
    .find_map(|(name, value)| (keys.iter().any(|k| *k == name) && *value != usize::MAX).then_some(*value))
}

fn median(values: &mut [usize]) -> Option<usize> {
  if values.is_empty() {
    return None;
  }
  values.sort_unstable();
  Some(values[values.len() / 2])
}

fn collect_blake3_threshold(results: &TuneResults, fallback: &AlgorithmResult, keys: &[&str]) -> Option<usize> {
  let mut values = Vec::new();

  for &(algo_name, _) in BLAKE3_TUNING_CORPUS {
    if algo_name.starts_with("blake3-stream") {
      continue;
    }
    if let Some(algo) = results.algorithms.iter().find(|a| a.name == algo_name)
      && let Some(v) = find_threshold_value(algo, keys)
    {
      values.push(v);
    }
  }

  if values.is_empty()
    && let Some(v) = find_threshold_value(fallback, keys)
  {
    values.push(v);
  }

  median(&mut values)
}

fn blake3_boundaries(results: &TuneResults, algo: &AlgorithmResult) -> [usize; 3] {
  let xs_max = SIZE_CLASS_XS_MAX;
  let mut s_max = SIZE_CLASS_S_MAX;
  let mut m_max = SIZE_CLASS_M_MAX;

  if let Some(crossover) = collect_blake3_threshold(results, algo, &["THRESHOLD_PORTABLE_TO_SIMD", "PORTABLE_TO_SIMD"])
  {
    s_max = crossover.saturating_sub(1);
  }
  if let Some(crossover) = collect_blake3_threshold(results, algo, &["THRESHOLD_SIMD_TO_WIDE", "SIMD_TO_WIDE"]) {
    m_max = crossover.saturating_sub(1);
  }

  // Keep monotonic size classes valid even when measured curves are noisy.
  s_max = s_max.max(xs_max);
  m_max = m_max.max(s_max);

  [xs_max, s_max, m_max]
}

#[derive(Clone, Copy, Debug)]
struct Blake3ParallelValues {
  min_bytes: usize,
  min_chunks: usize,
  max_threads: u8,
  spawn_cost_bytes: usize,
  merge_cost_bytes: usize,
  bytes_per_core_small: usize,
  bytes_per_core_medium: usize,
  bytes_per_core_large: usize,
  small_limit_bytes: usize,
  medium_limit_bytes: usize,
}

#[inline]
#[must_use]
fn default_blake3_parallel_values() -> Blake3ParallelValues {
  Blake3ParallelValues {
    min_bytes: 128 * 1024,
    min_chunks: 64,
    max_threads: 0,
    spawn_cost_bytes: 24 * 1024,
    merge_cost_bytes: 16 * 1024,
    bytes_per_core_small: 256 * 1024,
    bytes_per_core_medium: 128 * 1024,
    bytes_per_core_large: 64 * 1024,
    small_limit_bytes: 256 * 1024,
    medium_limit_bytes: 2 * 1024 * 1024,
  }
}

fn blake3_parallel_values(algo: &AlgorithmResult) -> Blake3ParallelValues {
  let mut v = default_blake3_parallel_values();

  for (suffix, value) in &algo.thresholds {
    match suffix.as_str() {
      "PARALLEL_MIN_BYTES" | "THRESHOLD_PARALLEL_MIN_BYTES" => v.min_bytes = *value,
      "PARALLEL_MIN_CHUNKS" | "THRESHOLD_PARALLEL_MIN_CHUNKS" => v.min_chunks = *value,
      "PARALLEL_MAX_THREADS" | "THRESHOLD_PARALLEL_MAX_THREADS" => {
        v.max_threads = (*value).min(u8::MAX as usize) as u8;
      }
      "PARALLEL_SPAWN_COST_BYTES" | "THRESHOLD_PARALLEL_SPAWN_COST_BYTES" => v.spawn_cost_bytes = *value,
      "PARALLEL_MERGE_COST_BYTES" | "THRESHOLD_PARALLEL_MERGE_COST_BYTES" => v.merge_cost_bytes = *value,
      "PARALLEL_BYTES_PER_CORE_SMALL" | "THRESHOLD_PARALLEL_BYTES_PER_CORE_SMALL" => v.bytes_per_core_small = *value,
      "PARALLEL_BYTES_PER_CORE_MEDIUM" | "THRESHOLD_PARALLEL_BYTES_PER_CORE_MEDIUM" => v.bytes_per_core_medium = *value,
      "PARALLEL_BYTES_PER_CORE_LARGE" | "THRESHOLD_PARALLEL_BYTES_PER_CORE_LARGE" => v.bytes_per_core_large = *value,
      "PARALLEL_SMALL_LIMIT_BYTES" | "THRESHOLD_PARALLEL_SMALL_LIMIT_BYTES" => v.small_limit_bytes = *value,
      "PARALLEL_MEDIUM_LIMIT_BYTES" | "THRESHOLD_PARALLEL_MEDIUM_LIMIT_BYTES" => v.medium_limit_bytes = *value,
      _ => {}
    }
  }

  // Keep profile monotonic even if input artifacts are noisy or partial.
  v.min_chunks = v.min_chunks.max(1);
  v.bytes_per_core_small = v.bytes_per_core_small.max(1);
  v.bytes_per_core_medium = v.bytes_per_core_medium.max(1);
  v.bytes_per_core_large = v.bytes_per_core_large.max(1);
  v.medium_limit_bytes = v.medium_limit_bytes.max(v.small_limit_bytes.saturating_add(1));
  v
}

fn aggregate_blake3_parallel_values(algos: &[&AlgorithmResult]) -> Blake3ParallelValues {
  if algos.is_empty() {
    return default_blake3_parallel_values();
  }

  let mut min_bytes = Vec::with_capacity(algos.len());
  let mut min_chunks = Vec::with_capacity(algos.len());
  let mut max_threads = Vec::with_capacity(algos.len());
  let mut spawn_cost_bytes = Vec::with_capacity(algos.len());
  let mut merge_cost_bytes = Vec::with_capacity(algos.len());
  let mut bytes_per_core_small = Vec::with_capacity(algos.len());
  let mut bytes_per_core_medium = Vec::with_capacity(algos.len());
  let mut bytes_per_core_large = Vec::with_capacity(algos.len());
  let mut small_limit_bytes = Vec::with_capacity(algos.len());
  let mut medium_limit_bytes = Vec::with_capacity(algos.len());

  for algo in algos {
    let values = blake3_parallel_values(algo);
    min_bytes.push(values.min_bytes);
    min_chunks.push(values.min_chunks);
    max_threads.push(values.max_threads as usize);
    spawn_cost_bytes.push(values.spawn_cost_bytes);
    merge_cost_bytes.push(values.merge_cost_bytes);
    bytes_per_core_small.push(values.bytes_per_core_small);
    bytes_per_core_medium.push(values.bytes_per_core_medium);
    bytes_per_core_large.push(values.bytes_per_core_large);
    small_limit_bytes.push(values.small_limit_bytes);
    medium_limit_bytes.push(values.medium_limit_bytes);
  }

  let defaults = default_blake3_parallel_values();
  let mut merged = Blake3ParallelValues {
    min_bytes: median(&mut min_bytes).unwrap_or(defaults.min_bytes),
    min_chunks: median(&mut min_chunks).unwrap_or(defaults.min_chunks),
    max_threads: median(&mut max_threads)
      .unwrap_or(defaults.max_threads as usize)
      .min(u8::MAX as usize) as u8,
    spawn_cost_bytes: median(&mut spawn_cost_bytes).unwrap_or(defaults.spawn_cost_bytes),
    merge_cost_bytes: median(&mut merge_cost_bytes).unwrap_or(defaults.merge_cost_bytes),
    bytes_per_core_small: median(&mut bytes_per_core_small).unwrap_or(defaults.bytes_per_core_small),
    bytes_per_core_medium: median(&mut bytes_per_core_medium).unwrap_or(defaults.bytes_per_core_medium),
    bytes_per_core_large: median(&mut bytes_per_core_large).unwrap_or(defaults.bytes_per_core_large),
    small_limit_bytes: median(&mut small_limit_bytes).unwrap_or(defaults.small_limit_bytes),
    medium_limit_bytes: median(&mut medium_limit_bytes).unwrap_or(defaults.medium_limit_bytes),
  };
  merged.min_chunks = merged.min_chunks.max(1);
  merged.bytes_per_core_small = merged.bytes_per_core_small.max(1);
  merged.bytes_per_core_medium = merged.bytes_per_core_medium.max(1);
  merged.bytes_per_core_large = merged.bytes_per_core_large.max(1);
  merged.medium_limit_bytes = merged
    .medium_limit_bytes
    .max(merged.small_limit_bytes.saturating_add(1));
  merged
}

fn generate_hash_table(tune_kind: TuneKind, algo: &AlgorithmResult, results: Option<&TuneResults>) -> String {
  // Defaults if size-class data isn't present (older results / partial runs).
  let mut xs = "portable";
  let mut s = "portable";
  let mut m = "portable";
  let mut l = "portable";

  for entry in &algo.size_class_best {
    match entry.class {
      "xs" => xs = entry.kernel.as_str(),
      "s" => s = entry.kernel.as_str(),
      "m" => m = entry.kernel.as_str(),
      "l" => l = entry.kernel.as_str(),
      _ => {}
    }
  }

  let kind_name = format!("{tune_kind:?}");
  let table_ident = tune_kind_table_ident(tune_kind);
  let use_tuned_boundaries = matches!(algo.name, "blake3-chunk" | "blake3");
  let [xs_max, s_max, m_max] = if use_tuned_boundaries {
    if let Some(results) = results {
      blake3_boundaries(results, algo)
    } else {
      [SIZE_CLASS_XS_MAX, SIZE_CLASS_S_MAX, SIZE_CLASS_M_MAX]
    }
  } else {
    [SIZE_CLASS_XS_MAX, SIZE_CLASS_S_MAX, SIZE_CLASS_M_MAX]
  };
  let boundaries_expr = if use_tuned_boundaries {
    format!("[{xs_max}, {s_max}, {m_max}]")
  } else {
    "DEFAULT_BOUNDARIES".to_string()
  };

  // Some hash dispatch tables (notably BLAKE3) use architecture-gated kernel IDs.
  // When we apply tuned results from (say) an x86_64 runner, we must not emit
  // `KernelId::X86*` into builds for other targets, because the enum variants
  // are `#[cfg(target_arch)]`-gated.
  //
  // For now, we handle BLAKE3 explicitly by emitting a cfg-gated table plus a
  // fallback definition for other targets.
  if matches!(algo.name, "blake3-chunk" | "blake3") {
    let kernels = [xs, s, m, l];
    let arch = if kernels.iter().any(|k| k.starts_with("x86_64/")) {
      Some("x86_64")
    } else if kernels.iter().any(|k| k.starts_with("aarch64/")) {
      Some("aarch64")
    } else {
      None
    };

    if let Some(arch) = arch {
      let cfg_expr = match arch {
        "x86_64" => "target_arch = \"x86_64\"",
        "aarch64" => "target_arch = \"aarch64\"",
        _ => unreachable!("unsupported arch tag"),
      };

      return format!(
        "\
// {kind_name} Table
#[cfg({cfg_expr})]
pub static {table_ident}: DispatchTable = DispatchTable {{
  boundaries: {boundaries_expr},
  xs: {xs_id},
  s: {s_id},
  m: {m_id},
  l: {l_id},
}};
#[cfg(not({cfg_expr}))]
pub static {table_ident}: DispatchTable = default_kind_table();
",
        boundaries_expr = boundaries_expr,
        xs_id = hash_kernel_expr(algo.name, xs),
        s_id = hash_kernel_expr(algo.name, s),
        m_id = hash_kernel_expr(algo.name, m),
        l_id = hash_kernel_expr(algo.name, l),
      );
    }
  }

  format!(
    "\
// {kind_name} Table
pub static {table_ident}: DispatchTable = DispatchTable {{
  boundaries: {boundaries_expr},
  xs: {xs_id},
  s: {s_id},
  m: {m_id},
  l: {l_id},
}};
",
    boundaries_expr = boundaries_expr,
    xs_id = hash_kernel_expr(algo.name, xs),
    s_id = hash_kernel_expr(algo.name, s),
    m_id = hash_kernel_expr(algo.name, m),
    l_id = hash_kernel_expr(algo.name, l),
  )
}

fn blake3_split_pair(kernel: &str) -> (&str, &str) {
  if let Some((stream, bulk)) = kernel.split_once('+') {
    (stream, bulk)
  } else {
    (kernel, kernel)
  }
}

fn choose_blake3_pair_component(results: &[&AlgorithmResult], pick_stream: bool) -> Option<String> {
  let mut counts: HashMap<String, (usize, usize)> = HashMap::new();
  for (idx, result) in results.iter().enumerate() {
    let (stream, bulk) = blake3_split_pair(result.best_kernel);
    let chosen = if pick_stream { stream } else { bulk };
    let entry = counts.entry(chosen.to_string()).or_insert((0, idx));
    entry.0 = entry.0.saturating_add(1);
  }
  counts
    .into_iter()
    .max_by(|a, b| a.1.0.cmp(&b.1.0).then_with(|| b.1.1.cmp(&a.1.1)))
    .map(|(kernel, _)| kernel)
}

#[cfg(test)]
fn generate_blake3_streaming_table(
  tune_kind: TuneKind,
  stream64_modes: &[&AlgorithmResult],
  stream4k_modes: &[&AlgorithmResult],
) -> String {
  let kind_name = format!("{tune_kind:?}");
  let table_ident = tune_kind_streaming_table_ident(tune_kind);

  let stream = choose_blake3_pair_component(stream64_modes, true).unwrap_or_else(|| "portable".to_string());
  let bulk = choose_blake3_pair_component(stream4k_modes, false).unwrap_or_else(|| "portable".to_string());

  let arch = if stream.starts_with("x86_64/") || bulk.starts_with("x86_64/") {
    Some("x86_64")
  } else if stream.starts_with("aarch64/") || bulk.starts_with("aarch64/") {
    Some("aarch64")
  } else {
    None
  };

  if let Some(arch) = arch {
    let cfg_expr = match arch {
      "x86_64" => "target_arch = \"x86_64\"",
      "aarch64" => "target_arch = \"aarch64\"",
      _ => unreachable!("unsupported arch tag"),
    };

    return format!(
      "\
// {kind_name} Streaming Table
#[cfg({cfg_expr})]
pub static {table_ident}: StreamingTable = StreamingTable {{
  stream: {stream_id},
  bulk: {bulk_id},
}};
#[cfg(not({cfg_expr}))]
pub static {table_ident}: StreamingTable = default_kind_streaming_table();
",
      stream_id = hash_kernel_expr("blake3-chunk", stream.as_str()),
      bulk_id = hash_kernel_expr("blake3-chunk", bulk.as_str()),
    );
  }

  format!(
    "\
// {kind_name} Streaming Table
pub static {table_ident}: StreamingTable = StreamingTable {{
  stream: {stream_id},
  bulk: {bulk_id},
}};
",
    stream_id = hash_kernel_expr("blake3-chunk", stream.as_str()),
    bulk_id = hash_kernel_expr("blake3-chunk", bulk.as_str()),
  )
}

#[derive(Clone, Copy)]
struct Blake3FamilySpec {
  marker: &'static str,
  profile_ident: &'static str,
  cfg_expr: Option<&'static str>,
}

#[inline]
#[must_use]
fn blake3_family_spec(tune_kind: TuneKind) -> Blake3FamilySpec {
  match tune_kind {
    TuneKind::Custom => Blake3FamilySpec {
      marker: "// Family Profile: CUSTOM",
      profile_ident: "PROFILE_CUSTOM",
      cfg_expr: None,
    },
    TuneKind::Default => Blake3FamilySpec {
      marker: "// Family Profile: DEFAULT_KIND",
      profile_ident: "PROFILE_DEFAULT_KIND",
      cfg_expr: None,
    },
    TuneKind::Portable => Blake3FamilySpec {
      marker: "// Family Profile: PORTABLE",
      profile_ident: "PROFILE_PORTABLE",
      cfg_expr: None,
    },
    TuneKind::Zen4 => Blake3FamilySpec {
      marker: "// Family Profile: X86_ZEN4",
      profile_ident: "PROFILE_X86_ZEN4",
      cfg_expr: Some("target_arch = \"x86_64\""),
    },
    TuneKind::Zen5 => Blake3FamilySpec {
      marker: "// Family Profile: X86_ZEN5",
      profile_ident: "PROFILE_X86_ZEN5",
      cfg_expr: Some("target_arch = \"x86_64\""),
    },
    TuneKind::Zen5c => Blake3FamilySpec {
      marker: "// Family Profile: X86_ZEN5C",
      profile_ident: "PROFILE_X86_ZEN5C",
      cfg_expr: Some("target_arch = \"x86_64\""),
    },
    TuneKind::IntelSpr => Blake3FamilySpec {
      marker: "// Family Profile: X86_INTEL_SPR",
      profile_ident: "PROFILE_X86_INTEL_SPR",
      cfg_expr: Some("target_arch = \"x86_64\""),
    },
    TuneKind::IntelGnr => Blake3FamilySpec {
      marker: "// Family Profile: X86_INTEL_GNR",
      profile_ident: "PROFILE_X86_INTEL_GNR",
      cfg_expr: Some("target_arch = \"x86_64\""),
    },
    TuneKind::IntelIcl => Blake3FamilySpec {
      marker: "// Family Profile: X86_INTEL_ICL",
      profile_ident: "PROFILE_X86_INTEL_ICL",
      cfg_expr: Some("target_arch = \"x86_64\""),
    },
    TuneKind::AppleM1M3 => Blake3FamilySpec {
      marker: "// Family Profile: AARCH64_APPLE_M1M3",
      profile_ident: "PROFILE_AARCH64_APPLE_M1M3",
      cfg_expr: Some("target_arch = \"aarch64\""),
    },
    TuneKind::AppleM4 => Blake3FamilySpec {
      marker: "// Family Profile: AARCH64_APPLE_M4",
      profile_ident: "PROFILE_AARCH64_APPLE_M4",
      cfg_expr: Some("target_arch = \"aarch64\""),
    },
    TuneKind::AppleM5 => Blake3FamilySpec {
      marker: "// Family Profile: AARCH64_APPLE_M5",
      profile_ident: "PROFILE_AARCH64_APPLE_M5",
      cfg_expr: Some("target_arch = \"aarch64\""),
    },
    TuneKind::Graviton2 => Blake3FamilySpec {
      marker: "// Family Profile: AARCH64_GRAVITON2",
      profile_ident: "PROFILE_AARCH64_GRAVITON2",
      cfg_expr: Some("target_arch = \"aarch64\""),
    },
    TuneKind::Graviton3
    | TuneKind::Graviton4
    | TuneKind::Graviton5
    | TuneKind::NeoverseN2
    | TuneKind::NeoverseN3
    | TuneKind::NeoverseV3
    | TuneKind::NvidiaGrace
    | TuneKind::AmpereAltra
    | TuneKind::Aarch64Pmull => Blake3FamilySpec {
      marker: "// Family Profile: AARCH64_SERVER_NEON",
      profile_ident: "PROFILE_AARCH64_SERVER_NEON",
      cfg_expr: Some("target_arch = \"aarch64\""),
    },
    TuneKind::Z13 => Blake3FamilySpec {
      marker: "// Family Profile: Z13",
      profile_ident: "PROFILE_Z13",
      cfg_expr: None,
    },
    TuneKind::Z14 => Blake3FamilySpec {
      marker: "// Family Profile: Z14",
      profile_ident: "PROFILE_Z14",
      cfg_expr: None,
    },
    TuneKind::Z15 => Blake3FamilySpec {
      marker: "// Family Profile: Z15",
      profile_ident: "PROFILE_Z15",
      cfg_expr: None,
    },
    TuneKind::Power7 => Blake3FamilySpec {
      marker: "// Family Profile: POWER7",
      profile_ident: "PROFILE_POWER7",
      cfg_expr: None,
    },
    TuneKind::Power8 => Blake3FamilySpec {
      marker: "// Family Profile: POWER8",
      profile_ident: "PROFILE_POWER8",
      cfg_expr: None,
    },
    TuneKind::Power9 => Blake3FamilySpec {
      marker: "// Family Profile: POWER9",
      profile_ident: "PROFILE_POWER9",
      cfg_expr: None,
    },
    TuneKind::Power10 => Blake3FamilySpec {
      marker: "// Family Profile: POWER10",
      profile_ident: "PROFILE_POWER10",
      cfg_expr: None,
    },
  }
}

fn generate_blake3_family_profile(
  tune_kind: TuneKind,
  algo: &AlgorithmResult,
  policy_source: &AlgorithmResult,
  stream64_modes: &[&AlgorithmResult],
  stream4k_modes: &[&AlgorithmResult],
  results: &TuneResults,
) -> String {
  let spec = blake3_family_spec(tune_kind);

  let mut xs = "portable";
  let mut s = "portable";
  let mut m = "portable";
  let mut l = "portable";
  for entry in &algo.size_class_best {
    match entry.class {
      "xs" => xs = entry.kernel.as_str(),
      "s" => s = entry.kernel.as_str(),
      "m" => m = entry.kernel.as_str(),
      "l" => l = entry.kernel.as_str(),
      _ => {}
    }
  }
  let boundaries = blake3_boundaries(results, algo);
  let stream = choose_blake3_pair_component(stream64_modes, true).unwrap_or_else(|| "portable".to_string());
  let bulk = choose_blake3_pair_component(stream4k_modes, false).unwrap_or_else(|| "portable".to_string());
  let parallel = blake3_parallel_values(policy_source);
  let streaming_parallel = aggregate_blake3_parallel_values(stream4k_modes);

  let core_body = format!(
    "\
pub static {profile_ident}: FamilyProfile = FamilyProfile {{
  dispatch: DispatchTable {{
    boundaries: [{xs_max}, {s_max}, {m_max}],
    xs: {xs_id},
    s: {s_id},
    m: {m_id},
    l: {l_id},
  }},
  streaming: StreamingTable {{
    stream: {stream_id},
    bulk: {bulk_id},
  }},
  parallel: ParallelTable {{
    min_bytes: {par_min_bytes},
    min_chunks: {par_min_chunks},
    max_threads: {par_max_threads},
    spawn_cost_bytes: {par_spawn_cost_bytes},
    merge_cost_bytes: {par_merge_cost_bytes},
    bytes_per_core_small: {par_bpc_small},
    bytes_per_core_medium: {par_bpc_medium},
    bytes_per_core_large: {par_bpc_large},
    small_limit_bytes: {par_small_limit},
    medium_limit_bytes: {par_medium_limit},
  }},
  streaming_parallel: ParallelTable {{
    min_bytes: {stream_par_min_bytes},
    min_chunks: {stream_par_min_chunks},
    max_threads: {stream_par_max_threads},
    spawn_cost_bytes: {stream_par_spawn_cost_bytes},
    merge_cost_bytes: {stream_par_merge_cost_bytes},
    bytes_per_core_small: {stream_par_bpc_small},
    bytes_per_core_medium: {stream_par_bpc_medium},
    bytes_per_core_large: {stream_par_bpc_large},
    small_limit_bytes: {stream_par_small_limit},
    medium_limit_bytes: {stream_par_medium_limit},
  }},
}};\n",
    profile_ident = spec.profile_ident,
    xs_max = boundaries[0],
    s_max = boundaries[1],
    m_max = boundaries[2],
    xs_id = hash_kernel_expr("blake3-chunk", xs),
    s_id = hash_kernel_expr("blake3-chunk", s),
    m_id = hash_kernel_expr("blake3-chunk", m),
    l_id = hash_kernel_expr("blake3-chunk", l),
    stream_id = hash_kernel_expr("blake3-chunk", stream.as_str()),
    bulk_id = hash_kernel_expr("blake3-chunk", bulk.as_str()),
    par_min_bytes = parallel.min_bytes,
    par_min_chunks = parallel.min_chunks,
    par_max_threads = parallel.max_threads,
    par_spawn_cost_bytes = parallel.spawn_cost_bytes,
    par_merge_cost_bytes = parallel.merge_cost_bytes,
    par_bpc_small = parallel.bytes_per_core_small,
    par_bpc_medium = parallel.bytes_per_core_medium,
    par_bpc_large = parallel.bytes_per_core_large,
    par_small_limit = parallel.small_limit_bytes,
    par_medium_limit = parallel.medium_limit_bytes,
    stream_par_min_bytes = streaming_parallel.min_bytes,
    stream_par_min_chunks = streaming_parallel.min_chunks,
    stream_par_max_threads = streaming_parallel.max_threads,
    stream_par_spawn_cost_bytes = streaming_parallel.spawn_cost_bytes,
    stream_par_merge_cost_bytes = streaming_parallel.merge_cost_bytes,
    stream_par_bpc_small = streaming_parallel.bytes_per_core_small,
    stream_par_bpc_medium = streaming_parallel.bytes_per_core_medium,
    stream_par_bpc_large = streaming_parallel.bytes_per_core_large,
    stream_par_small_limit = streaming_parallel.small_limit_bytes,
    stream_par_medium_limit = streaming_parallel.medium_limit_bytes,
  );

  if let Some(cfg_expr) = spec.cfg_expr {
    format!(
      "\
{marker}
#[cfg({cfg_expr})]
{core_body}#[cfg(not({cfg_expr}))]
pub static {profile_ident}: FamilyProfile = default_kind_profile();
",
      marker = spec.marker,
      cfg_expr = cfg_expr,
      core_body = core_body,
      profile_ident = spec.profile_ident,
    )
  } else {
    format!("{marker}\n{core_body}", marker = spec.marker, core_body = core_body)
  }
}

fn apply_hash_dispatch_tables(repo_root: &Path, results: &TuneResults) -> io::Result<Vec<GeneratedArtifact>> {
  let tune_kind = results.platform.tune_kind;
  let mut artifacts = Vec::new();

  for target in HASH_DISPATCH_TARGETS {
    let Some(algo) = find_hash_algo(results, target) else {
      continue;
    };

    if target.algo == "blake3-chunk" {
      let stream64_modes: Vec<&AlgorithmResult> =
        ["blake3-stream64", "blake3-stream64-keyed", "blake3-stream64-derive"]
          .iter()
          .filter_map(|name| results.algorithms.iter().find(|a| a.name == *name))
          .collect();
      let stream4k_modes: Vec<&AlgorithmResult> =
        ["blake3-stream4k", "blake3-stream4k-keyed", "blake3-stream4k-derive"]
          .iter()
          .filter_map(|name| results.algorithms.iter().find(|a| a.name == *name))
          .collect();
      let policy_source = results.algorithms.iter().find(|a| a.name == "blake3").ok_or_else(|| {
        io::Error::new(
          io::ErrorKind::InvalidData,
          "applying blake3 profiles requires `blake3` tuning results (missing algorithm `blake3`)",
        )
      })?;

      if stream64_modes.is_empty() || stream4k_modes.is_empty() {
        return Err(io::Error::new(
          io::ErrorKind::InvalidData,
          "applying blake3 profiles requires stream tuning results; missing one or more of: blake3-stream64, \
           blake3-stream64-keyed, blake3-stream64-derive, blake3-stream4k, blake3-stream4k-keyed, \
           blake3-stream4k-derive",
        ));
      }

      let profile_code = generate_blake3_family_profile(
        tune_kind,
        algo,
        policy_source,
        &stream64_modes,
        &stream4k_modes,
        results,
      );
      let spec = blake3_family_spec(tune_kind);
      let path = generated_apply_path(repo_root, "hashes", spec.profile_ident, tune_kind);
      artifacts.push(GeneratedArtifact {
        path,
        contents: profile_code,
      });
      continue;
    }

    let table_code = generate_hash_table(tune_kind, algo, Some(results));
    let path = generated_apply_path(repo_root, "hashes", target.algo, tune_kind);
    artifacts.push(GeneratedArtifact {
      path,
      contents: table_code,
    });
  }

  Ok(artifacts)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

fn validate_apply_input(results: &TuneResults) -> io::Result<()> {
  if results.platform.arch.is_empty() {
    return Err(io::Error::new(
      io::ErrorKind::InvalidData,
      "invalid tuning artifact: platform.arch must not be empty",
    ));
  }
  if results.platform.os.is_empty() {
    return Err(io::Error::new(
      io::ErrorKind::InvalidData,
      "invalid tuning artifact: platform.os must not be empty",
    ));
  }
  if results.algorithms.is_empty() {
    return Err(io::Error::new(
      io::ErrorKind::InvalidData,
      "invalid tuning artifact: no algorithm results present",
    ));
  }

  let mut names = BTreeSet::new();
  for algo in &results.algorithms {
    if !names.insert(algo.name) {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("invalid tuning artifact: duplicate algorithm result '{}'", algo.name),
      ));
    }

    for class in &algo.size_class_best {
      if !SIZE_CLASSES.contains(&class.class) {
        return Err(io::Error::new(
          io::ErrorKind::InvalidData,
          format!(
            "invalid tuning artifact: algorithm '{}' has unknown size class '{}'",
            algo.name, class.class
          ),
        ));
      }
    }
  }

  Ok(())
}

/// Apply tuning results to generate dispatch table entries.
///
/// This generates a `KernelTable` entry for the current platform and
/// either updates `dispatch.rs` or prints the generated code for manual
/// insertion.
pub fn apply_tuned_defaults(repo_root: &Path, results: &TuneResults) -> io::Result<()> {
  validate_apply_input(results)?;

  let mut artifacts = Vec::new();
  if should_apply_checksum_dispatch(results) {
    artifacts.extend(apply_checksum_dispatch_tables(repo_root, results)?);
  }

  artifacts.extend(apply_hash_dispatch_tables(repo_root, results)?);
  write_artifacts_transactional(&artifacts)?;

  Ok(())
}

/// Validate that tuning results can be safely applied on this host.
///
/// This is intended as a fast, "pre-flight" check for `--apply` that catches:
/// - Unknown/unmappable kernel families
/// - Missing per-size-class winners
/// - Kernel names that don't exist in the checksum bench registry for this host
pub fn self_check(results: &TuneResults) -> io::Result<()> {
  if !should_apply_checksum_dispatch(results) {
    return Ok(());
  }

  let a = algos(results)?;

  // Ensure we can generate the table for this TuneKind.
  let _ = generate_kernel_table(results.platform.tune_kind, &a)?;

  let ccitt = best_by_size_class(a.crc16_ccitt);
  let ibm = best_by_size_class(a.crc16_ibm);
  let openpgp = best_by_size_class(a.crc24_openpgp);
  let crc32 = best_by_size_class(a.crc32_ieee);
  let crc32c = best_by_size_class(a.crc32c);
  let xz = best_by_size_class(a.crc64_xz);
  let nvme = best_by_size_class(a.crc64_nvme);

  let crc16_avail = checksum::bench::available_crc16_kernels();
  let crc24_avail = checksum::bench::available_crc24_kernels();
  let crc32_ieee_avail = checksum::bench::available_crc32_ieee_kernels();
  let crc32c_avail = checksum::bench::available_crc32c_kernels();
  let crc64_avail = checksum::bench::available_crc64_kernels();

  fn ensure_in(avail: &[&'static str], name: &str, variant: CrcVariant) -> io::Result<()> {
    if avail.contains(&name) {
      return Ok(());
    }
    Err(io::Error::new(
      io::ErrorKind::InvalidData,
      format!("kernel not available on this host for {variant:?}: {name}"),
    ))
  }

  for class in SIZE_CLASSES {
    let ccitt_e = kernel_expr(CrcVariant::Crc16Ccitt, &ccitt[class].base, ccitt[class].streams)?;
    let ibm_e = kernel_expr(CrcVariant::Crc16Ibm, &ibm[class].base, ibm[class].streams)?;
    let openpgp_e = kernel_expr(CrcVariant::Crc24OpenPgp, &openpgp[class].base, openpgp[class].streams)?;
    let crc32_e = kernel_expr(CrcVariant::Crc32Ieee, &crc32[class].base, crc32[class].streams)?;
    let crc32c_e = kernel_expr(CrcVariant::Crc32c, &crc32c[class].base, crc32c[class].streams)?;
    let xz_e = kernel_expr(CrcVariant::Crc64Xz, &xz[class].base, xz[class].streams)?;
    let nvme_e = kernel_expr(CrcVariant::Crc64Nvme, &nvme[class].base, nvme[class].streams)?;

    let ccitt_name = ccitt_e.name.trim_matches('"');
    let ibm_name = ibm_e.name.trim_matches('"');
    let openpgp_name = openpgp_e.name.trim_matches('"');
    let crc32_name = crc32_e.name.trim_matches('"');
    let crc32c_name = crc32c_e.name.trim_matches('"');
    let xz_name = xz_e.name.trim_matches('"');
    let nvme_name = nvme_e.name.trim_matches('"');

    ensure_in(&crc16_avail, ccitt_name, CrcVariant::Crc16Ccitt)?;
    ensure_in(&crc16_avail, ibm_name, CrcVariant::Crc16Ibm)?;
    ensure_in(&crc24_avail, openpgp_name, CrcVariant::Crc24OpenPgp)?;
    ensure_in(&crc32_ieee_avail, crc32_name, CrcVariant::Crc32Ieee)?;
    ensure_in(&crc32c_avail, crc32c_name, CrcVariant::Crc32c)?;
    ensure_in(&crc64_avail, xz_name, CrcVariant::Crc64Xz)?;
    ensure_in(&crc64_avail, nvme_name, CrcVariant::Crc64Nvme)?;
  }

  Ok(())
}

#[cfg(test)]
mod tests {
  use std::path::Path;

  use platform::TuneKind;

  use super::{
    CrcVariant, generate_blake3_family_profile, generate_blake3_streaming_table, generate_hash_table, kernel_expr,
  };
  use crate::{AlgorithmResult, PlatformInfo, SizeClassBest, TuneResults, analysis::AnalysisResult};

  #[test]
  fn blake3_hash_tables_are_cfg_gated_for_arch_specific_kernels() {
    let algo = AlgorithmResult {
      name: "blake3-chunk",
      env_prefix: "RSCRYPTO_BENCH_BLAKE3_CHUNK",
      best_kernel: "x86_64/avx2",
      recommended_streams: 1,
      peak_throughput_gib_s: 0.0,
      size_class_best: vec![
        SizeClassBest {
          class: "m",
          kernel: "x86_64/avx2".to_string(),
          streams: 1,
          throughput_gib_s: 0.0,
        },
        SizeClassBest {
          class: "l",
          kernel: "x86_64/avx512".to_string(),
          streams: 1,
          throughput_gib_s: 0.0,
        },
      ],
      thresholds: vec![],
      analysis: AnalysisResult::default(),
    };

    let code = generate_hash_table(TuneKind::Zen4, &algo, None);
    assert!(code.contains("#[cfg(target_arch = \"x86_64\")]"));
    assert!(code.contains("#[cfg(not(target_arch = \"x86_64\"))]"));
    assert!(code.contains("default_kind_table()"));
    assert!(code.contains("KernelId::X86Avx2"));
    assert!(code.contains("KernelId::X86Avx512"));
  }

  #[test]
  fn blake3_family_profile_is_cfg_gated_and_contains_cost_model_terms() {
    let blake3_chunk = AlgorithmResult {
      name: "blake3-chunk",
      env_prefix: "RSCRYPTO_BENCH_BLAKE3_CHUNK",
      best_kernel: "x86_64/avx2",
      recommended_streams: 1,
      peak_throughput_gib_s: 0.0,
      size_class_best: vec![
        SizeClassBest {
          class: "m",
          kernel: "x86_64/avx2".to_string(),
          streams: 1,
          throughput_gib_s: 0.0,
        },
        SizeClassBest {
          class: "l",
          kernel: "x86_64/avx512".to_string(),
          streams: 1,
          throughput_gib_s: 0.0,
        },
      ],
      thresholds: vec![
        ("PARALLEL_MIN_BYTES".to_string(), 96 * 1024),
        ("PARALLEL_MIN_CHUNKS".to_string(), 48),
        ("PARALLEL_SPAWN_COST_BYTES".to_string(), 12 * 1024),
      ],
      analysis: AnalysisResult::default(),
    };
    let blake3 = AlgorithmResult {
      name: "blake3",
      env_prefix: "RSCRYPTO_BENCH_BLAKE3",
      best_kernel: "x86_64/avx2",
      recommended_streams: 1,
      peak_throughput_gib_s: 0.0,
      size_class_best: vec![],
      thresholds: vec![
        ("PARALLEL_MIN_BYTES".to_string(), 96 * 1024),
        ("PARALLEL_MIN_CHUNKS".to_string(), 48),
        ("PARALLEL_SPAWN_COST_BYTES".to_string(), 12 * 1024),
      ],
      analysis: AnalysisResult::default(),
    };
    let stream64 = AlgorithmResult {
      name: "blake3-stream64",
      env_prefix: "RSCRYPTO_BENCH_BLAKE3_STREAM64",
      best_kernel: "x86_64/sse4.1",
      recommended_streams: 1,
      peak_throughput_gib_s: 0.0,
      size_class_best: vec![],
      thresholds: vec![],
      analysis: AnalysisResult::default(),
    };
    let stream4k = AlgorithmResult {
      name: "blake3-stream4k",
      env_prefix: "RSCRYPTO_BENCH_BLAKE3_STREAM4K",
      best_kernel: "x86_64/avx512",
      recommended_streams: 1,
      peak_throughput_gib_s: 0.0,
      size_class_best: vec![],
      thresholds: vec![],
      analysis: AnalysisResult::default(),
    };
    let parent = AlgorithmResult {
      name: "blake3-parent",
      env_prefix: "RSCRYPTO_BENCH_BLAKE3_PARENT",
      best_kernel: "x86_64/avx2",
      recommended_streams: 1,
      peak_throughput_gib_s: 0.0,
      size_class_best: vec![],
      thresholds: vec![("THRESHOLD_PORTABLE_TO_SIMD".to_string(), 256)],
      analysis: AnalysisResult::default(),
    };

    let results = TuneResults {
      platform: PlatformInfo {
        arch: "x86_64",
        os: "linux",
        caps: platform::Caps::NONE,
        tune_kind: TuneKind::Zen4,
        description: String::new(),
      },
      algorithms: vec![
        blake3.clone(),
        blake3_chunk.clone(),
        stream64.clone(),
        stream4k.clone(),
        parent,
      ],
      timestamp: String::new(),
    };

    let code = generate_blake3_family_profile(
      TuneKind::Zen4,
      &blake3_chunk,
      &blake3,
      &[&stream64],
      &[&stream4k],
      &results,
    );
    assert!(code.contains("// Family Profile: X86_ZEN4"));
    assert!(code.contains("#[cfg(target_arch = \"x86_64\")]"));
    assert!(code.contains("pub static PROFILE_X86_ZEN4: FamilyProfile"));
    assert!(code.contains("spawn_cost_bytes: 12288"));
    assert!(code.contains("parallel: ParallelTable"));
    assert!(code.contains("streaming_parallel: ParallelTable"));
  }

  #[test]
  fn blake3_streaming_tables_are_cfg_gated_for_arch_specific_kernels() {
    let stream64 = AlgorithmResult {
      name: "blake3-stream64",
      env_prefix: "RSCRYPTO_BENCH_BLAKE3_STREAM64",
      best_kernel: "x86_64/sse4.1",
      recommended_streams: 1,
      peak_throughput_gib_s: 0.0,
      size_class_best: vec![],
      thresholds: vec![],
      analysis: AnalysisResult::default(),
    };

    let stream4k = AlgorithmResult {
      name: "blake3-stream4k",
      env_prefix: "RSCRYPTO_BENCH_BLAKE3_STREAM4K",
      best_kernel: "x86_64/avx512",
      recommended_streams: 1,
      peak_throughput_gib_s: 0.0,
      size_class_best: vec![],
      thresholds: vec![],
      analysis: AnalysisResult::default(),
    };

    let code = generate_blake3_streaming_table(TuneKind::IntelSpr, &[&stream64], &[&stream4k]);
    assert!(code.contains("#[cfg(target_arch = \"x86_64\")]"));
    assert!(code.contains("#[cfg(not(target_arch = \"x86_64\"))]"));
    assert!(code.contains("default_kind_streaming_table()"));
    assert!(code.contains("KernelId::X86Sse41"));
    assert!(code.contains("KernelId::X86Avx512"));
  }

  #[test]
  fn blake3_hash_tables_use_curve_derived_boundaries_when_available() {
    let blake3_chunk = AlgorithmResult {
      name: "blake3-chunk",
      env_prefix: "RSCRYPTO_BENCH_BLAKE3_CHUNK",
      best_kernel: "x86_64/avx2",
      recommended_streams: 1,
      peak_throughput_gib_s: 0.0,
      size_class_best: vec![],
      thresholds: vec![("THRESHOLD_PORTABLE_TO_SIMD".to_string(), 320)],
      analysis: AnalysisResult::default(),
    };

    let results = TuneResults {
      platform: PlatformInfo {
        arch: "x86_64",
        os: "linux",
        caps: platform::Caps::NONE,
        tune_kind: TuneKind::Zen4,
        description: String::new(),
      },
      algorithms: vec![
        blake3_chunk.clone(),
        AlgorithmResult {
          name: "blake3-parent",
          env_prefix: "RSCRYPTO_BENCH_BLAKE3_PARENT",
          best_kernel: "x86_64/avx2",
          recommended_streams: 1,
          peak_throughput_gib_s: 0.0,
          size_class_best: vec![],
          thresholds: vec![("THRESHOLD_PORTABLE_TO_SIMD".to_string(), 256)],
          analysis: AnalysisResult::default(),
        },
      ],
      timestamp: String::new(),
    };

    let code = generate_hash_table(TuneKind::Zen4, &blake3_chunk, Some(&results));
    // Median(320, 256) = 320 (upper median), then boundary is crossover-1.
    assert!(code.contains("boundaries: [64, 319, 4096]"));
  }

  #[test]
  fn apply_skips_checksum_dispatch_when_no_crc_results_exist() {
    let results = TuneResults {
      platform: PlatformInfo {
        arch: "x86_64",
        os: "linux",
        caps: platform::Caps::NONE,
        tune_kind: TuneKind::Zen4,
        description: String::new(),
      },
      algorithms: vec![AlgorithmResult {
        name: "blake3-chunk",
        env_prefix: "RSCRYPTO_BENCH_BLAKE3_CHUNK",
        best_kernel: "portable",
        recommended_streams: 1,
        peak_throughput_gib_s: 0.0,
        size_class_best: vec![],
        thresholds: vec![],
        analysis: AnalysisResult::default(),
      }],
      timestamp: String::new(),
    };

    assert!(!super::should_apply_checksum_dispatch(&results));
  }

  #[test]
  fn apply_errors_on_partial_crc_results() {
    let results = TuneResults {
      platform: PlatformInfo {
        arch: "x86_64",
        os: "linux",
        caps: platform::Caps::NONE,
        tune_kind: TuneKind::Zen4,
        description: String::new(),
      },
      algorithms: vec![AlgorithmResult {
        name: "crc32c",
        env_prefix: "RSCRYPTO_BENCH_CRC32C",
        best_kernel: "portable",
        recommended_streams: 1,
        peak_throughput_gib_s: 0.0,
        size_class_best: vec![],
        thresholds: vec![],
        analysis: AnalysisResult::default(),
      }],
      timestamp: String::new(),
    };

    assert!(super::should_apply_checksum_dispatch(&results));
    let err = super::apply_checksum_dispatch_tables(Path::new("."), &results).unwrap_err();
    assert!(err.to_string().contains("incomplete CRC tuning results"));
    assert!(err.to_string().contains("crc16-ccitt"));
  }

  #[test]
  fn checksum_kernel_expr_handles_non_x86_stream_kernels() {
    let e = kernel_expr(CrcVariant::Crc64Xz, "power/vpmsum", 8).unwrap();
    assert_eq!(e.func, "crc64_k::XZ_VPMSUM[4]");
    assert_eq!(e.name, "\"power/vpmsum-8way\"");

    let e = kernel_expr(CrcVariant::Crc64Nvme, "s390x/vgfm", 4).unwrap();
    assert_eq!(e.func, "crc64_k::NVME_VGFM[2]");
    assert_eq!(e.name, "\"s390x/vgfm-4way\"");

    let e = kernel_expr(CrcVariant::Crc32Ieee, "riscv64/zvbc", 2).unwrap();
    assert_eq!(e.func, "crc32_k::CRC32_ZVBC[1]");
    assert_eq!(e.name, "\"riscv64/zvbc-2way\"");
  }

  #[test]
  fn checksum_kernel_expr_handles_versioned_kernels_and_small_kernels() {
    let e = kernel_expr(CrcVariant::Crc32c, "x86_64/fusion-avx512-v4s3x3", 8).unwrap();
    assert_eq!(e.func, "crc32_k::CRC32C_FUSION_AVX512[4]");
    assert_eq!(e.name, "\"x86_64/fusion-avx512-v4s3x3-8way\"");

    let e = kernel_expr(CrcVariant::Crc32Ieee, "aarch64/pmull-eor3-v9s3x2e-s3", 3).unwrap();
    assert_eq!(e.func, "crc32_k::CRC32_PMULL_EOR3[2]");
    assert_eq!(e.name, "\"aarch64/pmull-eor3-v9s3x2e-s3-3way\"");

    let e = kernel_expr(CrcVariant::Crc24OpenPgp, "x86_64/pclmul-small", 1).unwrap();
    assert_eq!(e.func, "crc24_k::OPENPGP_PCLMUL_SMALL_KERNEL");
    assert_eq!(e.name, "\"x86_64/pclmul-small\"");
  }
}
