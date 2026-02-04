//! Apply `rscrypto-tune` results into dispatch kernel tables.
//!
//! This is the "pipeline" step: a machine can run `rscrypto-tune --apply`,
//! and the repo's `dispatch.rs` tables get updated for that platform's
//! [`platform::TuneKind`].
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

use crate::{AlgorithmResult, TuneResults};

// ─────────────────────────────────────────────────────────────────────────────
// Size Class Boundaries
// ─────────────────────────────────────────────────────────────────────────────

/// Size class byte thresholds (must match dispatch.rs).
const SIZE_CLASS_XS_MAX: usize = 64;
const SIZE_CLASS_S_MAX: usize = 256;
const SIZE_CLASS_M_MAX: usize = 4096;

/// Size classes used in tuning and dispatch.
const SIZE_CLASSES: [&str; 4] = ["xs", "s", "m", "l"];

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

fn apply_checksum_dispatch_tables(repo_root: &Path, results: &TuneResults) -> io::Result<()> {
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

  let dispatch_path = dispatch_path(repo_root);
  let content = read_file(&dispatch_path)?;
  let updated = update_dispatch_file(&content, table_ident, &table_code)?;
  if updated != content {
    write_file(&dispatch_path, &updated)?;
    eprintln!("Updated: {}", dispatch_path.display());
  }

  Ok(())
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
// File Update Logic
// ─────────────────────────────────────────────────────────────────────────────

// Markers for future auto-update support (not currently used).
#[allow(dead_code)]
const BEGIN_MARKER: &str = "// BEGIN GENERATED (rscrypto-tune)";
#[allow(dead_code)]
const END_MARKER: &str = "// END GENERATED (rscrypto-tune)";

fn dispatch_path(repo_root: &Path) -> PathBuf {
  repo_root.join("crates/checksum/src/dispatch.rs")
}

struct HashDispatchTarget {
  algo: &'static str,
  aliases: &'static [&'static str],
  rel_path: &'static str,
}

const HASH_DISPATCH_TARGETS: &[HashDispatchTarget] = &[
  HashDispatchTarget {
    algo: "sha224-compress",
    aliases: &["sha224"],
    rel_path: "crates/hashes/src/crypto/sha224/dispatch_tables.rs",
  },
  HashDispatchTarget {
    algo: "sha256-compress",
    aliases: &["sha256"],
    rel_path: "crates/hashes/src/crypto/sha256/dispatch_tables.rs",
  },
  HashDispatchTarget {
    algo: "sha384-compress",
    aliases: &["sha384"],
    rel_path: "crates/hashes/src/crypto/sha384/dispatch_tables.rs",
  },
  HashDispatchTarget {
    algo: "sha512-compress",
    aliases: &["sha512"],
    rel_path: "crates/hashes/src/crypto/sha512/dispatch_tables.rs",
  },
  HashDispatchTarget {
    algo: "sha512-224-compress",
    aliases: &["sha512-224"],
    rel_path: "crates/hashes/src/crypto/sha512_224/dispatch_tables.rs",
  },
  HashDispatchTarget {
    algo: "sha512-256-compress",
    aliases: &["sha512-256"],
    rel_path: "crates/hashes/src/crypto/sha512_256/dispatch_tables.rs",
  },
  HashDispatchTarget {
    algo: "blake2b-512-compress",
    aliases: &["blake2b-512"],
    rel_path: "crates/hashes/src/crypto/blake2b/dispatch_tables.rs",
  },
  HashDispatchTarget {
    algo: "blake2s-256-compress",
    aliases: &["blake2s-256"],
    rel_path: "crates/hashes/src/crypto/blake2s/dispatch_tables.rs",
  },
  HashDispatchTarget {
    algo: "blake3-chunk",
    aliases: &["blake3"],
    rel_path: "crates/hashes/src/crypto/blake3/dispatch_tables.rs",
  },
  HashDispatchTarget {
    algo: "xxh3",
    aliases: &[],
    rel_path: "crates/hashes/src/fast/xxh3/dispatch_tables.rs",
  },
  HashDispatchTarget {
    algo: "rapidhash",
    aliases: &[],
    rel_path: "crates/hashes/src/fast/rapidhash/dispatch_tables.rs",
  },
  HashDispatchTarget {
    algo: "siphash",
    aliases: &[],
    rel_path: "crates/hashes/src/fast/siphash/dispatch_tables.rs",
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
    rel_path: "crates/hashes/src/crypto/keccak/dispatch_tables.rs",
  },
  HashDispatchTarget {
    algo: "ascon-hash256",
    aliases: &["ascon-xof128"],
    rel_path: "crates/hashes/src/crypto/ascon/dispatch_tables.rs",
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

fn read_file(path: &Path) -> io::Result<String> {
  fs::read_to_string(path)
}

fn write_file(path: &Path, contents: &str) -> io::Result<()> {
  fs::write(path, contents)
}

fn checksum_arch_module_name() -> Option<&'static str> {
  if cfg!(target_arch = "aarch64") {
    Some("aarch64_tables")
  } else if cfg!(target_arch = "x86_64") {
    Some("x86_64_tables")
  } else if cfg!(target_arch = "s390x") {
    Some("s390x_tables")
  } else if cfg!(target_arch = "powerpc64") {
    Some("power_tables")
  } else if cfg!(target_arch = "riscv64") {
    Some("riscv64_tables")
  } else {
    None
  }
}

fn find_matching_brace(bytes: &[u8], open: usize) -> Option<usize> {
  let mut depth: usize = 0;
  for (i, &b) in bytes.iter().enumerate().skip(open) {
    match b {
      b'{' => depth = depth.strict_add(1),
      b'}' => {
        depth = depth.strict_sub(1);
        if depth == 0 {
          return Some(i);
        }
      }
      _ => {}
    }
  }
  None
}

fn find_kernel_table_section(content: &str, table_ident: &str) -> Option<(usize, usize)> {
  let needle = format!("pub static {table_ident}: KernelTable");
  let start_idx = content.find(&needle)?;

  // Expand backwards to include the header line if present.
  let section_start = content[..start_idx]
    .rfind("\n  // ───────────────────────────────────────────────────────────────────────────")
    .map(|i| i.strict_add(1))
    .unwrap_or(start_idx);

  // Find the opening brace of the KernelTable initializer.
  let init_idx = content[start_idx..].find("= KernelTable {")?.strict_add(start_idx);
  let open_brace = content[init_idx..].find('{')?.strict_add(init_idx);

  let bytes = content.as_bytes();
  let close_brace = find_matching_brace(bytes, open_brace)?;

  // Include trailing `;` and one newline if present.
  let mut end = close_brace.strict_add(1);
  while end < bytes.len() && (bytes[end] == b' ' || bytes[end] == b'\t' || bytes[end] == b'\r') {
    end += 1;
  }
  if end < bytes.len() && bytes[end] == b';' {
    end += 1;
  }
  if end < bytes.len() && bytes[end] == b'\n' {
    end += 1;
  }

  Some((section_start, end))
}

fn insert_into_module(content: &str, module_name: &str, table_code: &str) -> io::Result<String> {
  let needle = format!("mod {module_name} {{");
  let start = content.find(&needle).ok_or_else(|| {
    io::Error::new(
      io::ErrorKind::InvalidData,
      format!("failed to find checksum dispatch module: {module_name}"),
    )
  })?;

  let open_brace = content[start..]
    .find('{')
    .map(|off| start.strict_add(off))
    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "malformed module declaration"))?;

  let bytes = content.as_bytes();
  let close_brace = find_matching_brace(bytes, open_brace).ok_or_else(|| {
    io::Error::new(
      io::ErrorKind::InvalidData,
      format!("failed to parse module body for: {module_name}"),
    )
  })?;

  let before = &content[..close_brace];
  let after = &content[close_brace..];
  Ok(format!("{before}\n\n{table_code}{after}"))
}

/// Insert or update a generated checksum table in `crates/checksum/src/dispatch.rs`.
fn update_dispatch_file(content: &str, table_ident: &str, table_code: &str) -> io::Result<String> {
  if let Some((start, end)) = find_kernel_table_section(content, table_ident) {
    let before = &content[..start];
    let after = &content[end..];
    return Ok(format!("{before}{table_code}{after}"));
  }

  // No existing section - insert into the architecture module for this host.
  let Some(module_name) = checksum_arch_module_name() else {
    return Err(io::Error::new(
      io::ErrorKind::InvalidInput,
      "checksum dispatch apply is not supported on this architecture",
    ));
  };

  insert_into_module(content, module_name, table_code)
}

fn update_hash_dispatch_tables_file(content: &str, tune_kind: TuneKind, table_code: &str) -> io::Result<String> {
  let kind_name = format!("{tune_kind:?}");
  let section_marker = format!("// {kind_name} Table");

  update_hash_dispatch_tables_section(content, &section_marker, table_code)
}

fn update_hash_dispatch_tables_section(content: &str, section_marker: &str, table_code: &str) -> io::Result<String> {
  let Some(start_idx) = content.find(section_marker) else {
    return Err(io::Error::new(
      io::ErrorKind::InvalidData,
      format!("dispatch_tables.rs missing marker: {section_marker}"),
    ));
  };

  let rest = &content[start_idx..];
  let end_offset = rest
    .find("\n// ")
    .or_else(|| rest.find("\n\n#[inline]"))
    .or_else(|| rest.find("\n\npub fn "))
    .unwrap_or(rest.len());

  let before = &content[..start_idx];
  let after = &content[start_idx.strict_add(end_offset)..];
  Ok(format!("{before}{table_code}{after}"))
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

fn tune_kind_parallel_table_ident(tune_kind: TuneKind) -> &'static str {
  match tune_kind {
    TuneKind::Custom => "CUSTOM_PARALLEL_TABLE",
    TuneKind::Default => "DEFAULT_KIND_PARALLEL_TABLE",
    TuneKind::Portable => "PORTABLE_PARALLEL_TABLE",
    TuneKind::Zen4 => "ZEN4_PARALLEL_TABLE",
    TuneKind::Zen5 => "ZEN5_PARALLEL_TABLE",
    TuneKind::Zen5c => "ZEN5C_PARALLEL_TABLE",
    TuneKind::IntelSpr => "INTELSPR_PARALLEL_TABLE",
    TuneKind::IntelGnr => "INTELGNR_PARALLEL_TABLE",
    TuneKind::IntelIcl => "INTELICL_PARALLEL_TABLE",
    TuneKind::AppleM1M3 => "APPLEM1M3_PARALLEL_TABLE",
    TuneKind::AppleM4 => "APPLEM4_PARALLEL_TABLE",
    TuneKind::AppleM5 => "APPLEM5_PARALLEL_TABLE",
    TuneKind::Graviton2 => "GRAVITON2_PARALLEL_TABLE",
    TuneKind::Graviton3 => "GRAVITON3_PARALLEL_TABLE",
    TuneKind::Graviton4 => "GRAVITON4_PARALLEL_TABLE",
    TuneKind::Graviton5 => "GRAVITON5_PARALLEL_TABLE",
    TuneKind::NeoverseN2 => "NEOVERSEN2_PARALLEL_TABLE",
    TuneKind::NeoverseN3 => "NEOVERSEN3_PARALLEL_TABLE",
    TuneKind::NeoverseV3 => "NEOVERSEV3_PARALLEL_TABLE",
    TuneKind::NvidiaGrace => "NVIDIAGRACE_PARALLEL_TABLE",
    TuneKind::AmpereAltra => "AMPEREALTRA_PARALLEL_TABLE",
    TuneKind::Aarch64Pmull => "AARCH64PMULL_PARALLEL_TABLE",
    TuneKind::Z13 => "Z13_PARALLEL_TABLE",
    TuneKind::Z14 => "Z14_PARALLEL_TABLE",
    TuneKind::Z15 => "Z15_PARALLEL_TABLE",
    TuneKind::Power7 => "POWER7_PARALLEL_TABLE",
    TuneKind::Power8 => "POWER8_PARALLEL_TABLE",
    TuneKind::Power9 => "POWER9_PARALLEL_TABLE",
    TuneKind::Power10 => "POWER10_PARALLEL_TABLE",
  }
}

fn hash_kernel_expr(algo: &str, kernel_name: &str) -> &'static str {
  match algo {
    // BLAKE3 dispatch tables select a full kernel bundle (compress + chunk + parent + hash_many).
    "blake3-chunk" | "blake3" => match kernel_name {
      "portable" => "KernelId::Portable",
      "x86_64/ssse3" => "KernelId::X86Ssse3",
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

fn generate_blake3_parallel_table(tune_kind: TuneKind, algo: &AlgorithmResult) -> String {
  let kind_name = format!("{tune_kind:?}");
  let table_ident = tune_kind_parallel_table_ident(tune_kind);

  let mut min_bytes: usize = 512 * 1024;
  let mut min_chunks: usize = 256;
  let mut max_threads: u8 = 0;

  for (suffix, value) in &algo.thresholds {
    match suffix.as_str() {
      "PARALLEL_MIN_BYTES" => min_bytes = *value,
      "PARALLEL_MIN_CHUNKS" => min_chunks = *value,
      "PARALLEL_MAX_THREADS" => {
        let v = (*value).min(u8::MAX as usize) as u8;
        max_threads = v;
      }
      _ => {}
    }
  }

  format!(
    "\
// {kind_name} Parallel Table
pub static {table_ident}: ParallelTable = ParallelTable {{
  min_bytes: {min_bytes},
  min_chunks: {min_chunks},
  max_threads: {max_threads},
}};
"
  )
}

fn generate_hash_table(tune_kind: TuneKind, algo: &AlgorithmResult) -> String {
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
  boundaries: DEFAULT_BOUNDARIES,
  xs: {xs_id},
  s: {s_id},
  m: {m_id},
  l: {l_id},
}};
#[cfg(not({cfg_expr}))]
pub static {table_ident}: DispatchTable = default_kind_table();
",
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
  boundaries: DEFAULT_BOUNDARIES,
  xs: {xs_id},
  s: {s_id},
  m: {m_id},
  l: {l_id},
}};
",
    xs_id = hash_kernel_expr(algo.name, xs),
    s_id = hash_kernel_expr(algo.name, s),
    m_id = hash_kernel_expr(algo.name, m),
    l_id = hash_kernel_expr(algo.name, l),
  )
}

fn generate_blake3_streaming_table(
  tune_kind: TuneKind,
  stream64: &AlgorithmResult,
  stream4k: &AlgorithmResult,
) -> String {
  let kind_name = format!("{tune_kind:?}");
  let table_ident = tune_kind_streaming_table_ident(tune_kind);

  let stream = stream64.best_kernel;
  let bulk = stream4k.best_kernel;

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
      stream_id = hash_kernel_expr("blake3-chunk", stream),
      bulk_id = hash_kernel_expr("blake3-chunk", bulk),
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
    stream_id = hash_kernel_expr("blake3-chunk", stream),
    bulk_id = hash_kernel_expr("blake3-chunk", bulk),
  )
}

fn apply_hash_dispatch_tables(repo_root: &Path, results: &TuneResults) -> io::Result<()> {
  let tune_kind = results.platform.tune_kind;

  for target in HASH_DISPATCH_TARGETS {
    let Some(algo) = find_hash_algo(results, target) else {
      continue;
    };

    let table_code = generate_hash_table(tune_kind, algo);
    let path = repo_root.join(target.rel_path);
    let content = read_file(&path)?;
    let mut updated = update_hash_dispatch_tables_file(&content, tune_kind, &table_code)?;

    // BLAKE3: also apply streaming dispatch preferences (stream kernel vs bulk kernel).
    if target.algo == "blake3-chunk" {
      let stream64 = results.algorithms.iter().find(|a| a.name == "blake3-stream64");
      let stream4k = results.algorithms.iter().find(|a| a.name == "blake3-stream4k");
      if let (Some(stream64), Some(stream4k)) = (stream64, stream4k) {
        let streaming_code = generate_blake3_streaming_table(tune_kind, stream64, stream4k);
        let kind_name = format!("{tune_kind:?}");
        let marker = format!("// {kind_name} Streaming Table");
        updated = update_hash_dispatch_tables_section(&updated, &marker, &streaming_code)?;
      }

      // BLAKE3: apply the tuned multi-core policy (if present) from the `blake3`
      // oneshot result. If it isn't present (partial run), fall back to this
      // target's thresholds or leave defaults unchanged.
      let policy_source = results.algorithms.iter().find(|a| a.name == "blake3").unwrap_or(algo);
      let parallel_code = generate_blake3_parallel_table(tune_kind, policy_source);
      let kind_name = format!("{tune_kind:?}");
      let marker = format!("// {kind_name} Parallel Table");
      updated = update_hash_dispatch_tables_section(&updated, &marker, &parallel_code)?;
    }

    if updated != content {
      write_file(&path, &updated)?;
      eprintln!("Updated: {}", path.display());
    }
  }

  Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Apply tuning results to generate dispatch table entries.
///
/// This generates a `KernelTable` entry for the current platform and
/// either updates `dispatch.rs` or prints the generated code for manual
/// insertion.
pub fn apply_tuned_defaults(repo_root: &Path, results: &TuneResults) -> io::Result<()> {
  if should_apply_checksum_dispatch(results) {
    apply_checksum_dispatch_tables(repo_root, results)?;
  }

  apply_hash_dispatch_tables(repo_root, results)?;

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

  use super::{CrcVariant, generate_blake3_streaming_table, generate_hash_table, kernel_expr};
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

    let code = generate_hash_table(TuneKind::Zen4, &algo);
    assert!(code.contains("#[cfg(target_arch = \"x86_64\")]"));
    assert!(code.contains("#[cfg(not(target_arch = \"x86_64\"))]"));
    assert!(code.contains("default_kind_table()"));
    assert!(code.contains("KernelId::X86Avx2"));
    assert!(code.contains("KernelId::X86Avx512"));
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

    let code = generate_blake3_streaming_table(TuneKind::IntelSpr, &stream64, &stream4k);
    assert!(code.contains("#[cfg(target_arch = \"x86_64\")]"));
    assert!(code.contains("#[cfg(not(target_arch = \"x86_64\"))]"));
    assert!(code.contains("default_kind_streaming_table()"));
    assert!(code.contains("KernelId::X86Sse41"));
    assert!(code.contains("KernelId::X86Avx512"));
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
