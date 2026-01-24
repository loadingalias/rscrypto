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
// Kernel Name Mapping
// ─────────────────────────────────────────────────────────────────────────────

/// Map kernel names from tuning output to dispatch.rs constant references.
///
/// The tuning engine outputs kernel names like "aarch64/pmull-eor3-3way".
/// We need to map these to Rust constant references like `crc64_k::XZ_PMULL_EOR3[2]`.
struct KernelMapper {
  /// Maps (arch, base_kernel, streams) -> Rust constant reference
  mappings: HashMap<(&'static str, &'static str, u8), &'static str>,
}

impl KernelMapper {
  fn new() -> Self {
    let mut m = HashMap::new();

    // ─── aarch64 CRC64 kernels ───
    // PMULL
    m.insert(("aarch64", "pmull", 1), "[0]");
    m.insert(("aarch64", "pmull", 2), "[1]");
    m.insert(("aarch64", "pmull", 3), "[2]");
    // PMULL-EOR3
    m.insert(("aarch64", "pmull-eor3", 1), "[0]");
    m.insert(("aarch64", "pmull-eor3", 2), "[1]");
    m.insert(("aarch64", "pmull-eor3", 3), "[2]");
    // SVE2-PMULL
    m.insert(("aarch64", "sve2-pmull", 1), "[0]");
    m.insert(("aarch64", "sve2-pmull", 2), "[1]");
    m.insert(("aarch64", "sve2-pmull", 3), "[2]");

    // ─── x86_64 CRC64 kernels ───
    // PCLMUL
    m.insert(("x86_64", "pclmul", 1), "[0]");
    m.insert(("x86_64", "pclmul", 2), "[1]");
    m.insert(("x86_64", "pclmul", 4), "[2]");
    m.insert(("x86_64", "pclmul", 7), "[3]");
    m.insert(("x86_64", "pclmul", 8), "[4]");
    // VPCLMUL
    m.insert(("x86_64", "vpclmul", 1), "[0]");
    m.insert(("x86_64", "vpclmul", 2), "[1]");
    m.insert(("x86_64", "vpclmul", 4), "[2]");
    m.insert(("x86_64", "vpclmul", 7), "[3]");
    m.insert(("x86_64", "vpclmul", 8), "[4]");

    Self { mappings: m }
  }

  /// Parse a kernel name like "aarch64/pmull-eor3-3way" into (arch, base, streams).
  fn parse_kernel_name(name: &str) -> Option<(&str, &str, u8)> {
    let parts: Vec<&str> = name.split('/').collect();
    if parts.len() != 2 {
      return None;
    }

    let arch = parts[0];
    let kernel = parts[1];

    // Extract stream count from suffix
    let (base, streams) = if kernel.ends_with("-2way") {
      (kernel.strip_suffix("-2way")?, 2)
    } else if kernel.ends_with("-3way") {
      (kernel.strip_suffix("-3way")?, 3)
    } else if kernel.ends_with("-4way") {
      (kernel.strip_suffix("-4way")?, 4)
    } else if kernel.ends_with("-7way") {
      (kernel.strip_suffix("-7way")?, 7)
    } else if kernel.ends_with("-8way") {
      (kernel.strip_suffix("-8way")?, 8)
    } else {
      (kernel, 1)
    };

    Some((arch, base, streams))
  }

  /// Get the Rust constant reference for a kernel name and variant.
  fn get_constant(&self, kernel_name: &str, variant: &str) -> String {
    // Handle portable kernels
    if kernel_name.starts_with("portable/") || kernel_name == "portable" {
      return self.portable_constant(variant);
    }

    // Handle small kernels (no stream suffix, special constant)
    if kernel_name.ends_with("-small") {
      return self.small_constant(kernel_name, variant);
    }

    // Parse the kernel name
    let Some((arch, base, streams)) = Self::parse_kernel_name(kernel_name) else {
      return self.fallback_constant(variant);
    };

    // Get the array index suffix
    let idx_suffix = self.mappings.get(&(arch, base, streams)).copied().unwrap_or("[0]");

    // Build the full constant reference
    self.build_constant(arch, base, variant, idx_suffix)
  }

  fn portable_constant(&self, variant: &str) -> String {
    match variant {
      "crc16-ccitt" => "crate::crc16::portable::crc16_ccitt_slice8".to_string(),
      "crc16-ibm" => "crate::crc16::portable::crc16_ibm_slice8".to_string(),
      "crc24-openpgp" => "crate::crc24::portable::crc24_openpgp_slice8".to_string(),
      "crc32-ieee" => "crate::crc32::portable::crc32_slice16_ieee".to_string(),
      "crc32c" => "crate::crc32::portable::crc32c_slice16".to_string(),
      "crc64-xz" => "crate::crc64::portable::crc64_slice16_xz".to_string(),
      "crc64-nvme" => "crate::crc64::portable::crc64_slice16_nvme".to_string(),
      _ => format!("/* unknown portable variant: {variant} */"),
    }
  }

  fn small_constant(&self, kernel_name: &str, variant: &str) -> String {
    let parts: Vec<&str> = kernel_name.split('/').collect();
    if parts.len() != 2 {
      return self.fallback_constant(variant);
    }

    let arch = parts[0];
    let kernel_module = self.arch_to_module(arch);
    let variant_prefix = self.variant_to_prefix(variant);

    // e.g., crc64_k::XZ_PMULL_SMALL
    format!("{kernel_module}::{variant_prefix}_SMALL")
  }

  fn build_constant(&self, arch: &str, base: &str, variant: &str, idx_suffix: &str) -> String {
    let kernel_module = self.arch_to_module(arch);
    let variant_prefix = self.variant_to_prefix(variant);
    let kernel_suffix = self.base_to_suffix(base);

    // e.g., crc64_k::XZ_PMULL_EOR3[2]
    format!("{kernel_module}::{variant_prefix}_{kernel_suffix}{idx_suffix}")
  }

  fn arch_to_module(&self, arch: &str) -> &'static str {
    match arch {
      "aarch64" => "crc64_k",
      "x86_64" => "crc64_k",
      "power" | "powerpc64" => "crc64_k::power",
      "s390x" => "crc64_k::s390x",
      "riscv64" => "crc64_k::riscv64",
      _ => "crc64_k",
    }
  }

  fn variant_to_prefix(&self, variant: &str) -> &'static str {
    match variant {
      "crc64-xz" => "XZ",
      "crc64-nvme" => "NVME",
      "crc32-ieee" => "CRC32",
      "crc32c" => "CRC32C",
      "crc16-ccitt" => "CCITT",
      "crc16-ibm" => "IBM",
      "crc24-openpgp" => "OPENPGP",
      _ => "UNKNOWN",
    }
  }

  fn base_to_suffix(&self, base: &str) -> &'static str {
    match base {
      "pmull" => "PMULL",
      "pmull-eor3" => "PMULL_EOR3",
      "sve2-pmull" => "SVE2_PMULL",
      "pclmul" => "PCLMUL",
      "vpclmul" => "VPCLMUL",
      "vpmsum" => "VPMSUM",
      "vgfm" => "VGFM",
      "zbc" => "ZBC",
      "zvbc" => "ZVBC",
      "hwcrc" => "HWCRC",
      "fusion-sse" => "FUSION_SSE",
      "fusion-vpclmul" => "FUSION_VPCLMUL",
      _ => "UNKNOWN",
    }
  }

  fn fallback_constant(&self, variant: &str) -> String {
    self.portable_constant(variant)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Size Class Best Kernel Selection
// ─────────────────────────────────────────────────────────────────────────────

/// Find the best kernel for each size class from the analysis results.
fn best_kernels_per_size_class(algo: &AlgorithmResult) -> HashMap<String, String> {
  let mut best = HashMap::new();

  // Use the best large kernel as default for all size classes
  let default_kernel = algo.best_kernel;

  for class in SIZE_CLASSES {
    // For now, use the best large kernel for all size classes.
    // A more sophisticated version would analyze crossover points
    // to determine the best kernel for each size class.
    //
    // TODO: Extend analysis to track per-size-class winners.
    best.insert(class.to_string(), default_kernel.to_string());
  }

  // Apply crossover logic: if we have a portable_to_clmul threshold,
  // use portable/slice for sizes below the threshold.
  if let Some((_, threshold)) = algo.thresholds.iter().find(|(k, _)| k.contains("PORTABLE_TO_CLMUL")) {
    let class = size_to_class(*threshold);
    if class == "xs" || class == "s" {
      // Use small kernel variant for tiny buffers
      // Construct proper small kernel name
      let parts: Vec<&str> = algo.best_kernel.split('/').collect();
      if parts.len() == 2 {
        let arch = parts[0];
        let base = parts[1]
          .split('-')
          .take_while(|p| !["2way", "3way", "4way", "7way", "8way", "small"].contains(p))
          .collect::<Vec<_>>()
          .join("-");
        best.insert("xs".to_string(), format!("{arch}/{base}-small"));
      }
    }
  }

  best
}

/// Map a size threshold to a size class.
fn size_to_class(size: usize) -> &'static str {
  if size <= SIZE_CLASS_XS_MAX {
    "xs"
  } else if size <= SIZE_CLASS_S_MAX {
    "s"
  } else if size <= SIZE_CLASS_M_MAX {
    "m"
  } else {
    "l"
  }
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
  let table_code = generate_kernel_table(tune_kind, &a);

  let dispatch_path = dispatch_path(repo_root);
  let content = read_file(&dispatch_path)?;
  let updated = update_dispatch_file(&content, tune_kind, &table_code)?;
  if updated != content {
    write_file(&dispatch_path, &updated)?;
    eprintln!("Updated: {}", dispatch_path.display());
  }

  Ok(())
}

/// Generate a KernelSet entry for one size class.
fn generate_kernel_set(size_class: &str, algos: &AlgoSet<'_>, mapper: &KernelMapper, indent: &str) -> String {
  let crc16_ccitt_best = best_kernels_per_size_class(algos.crc16_ccitt);
  let crc16_ibm_best = best_kernels_per_size_class(algos.crc16_ibm);
  let crc24_openpgp_best = best_kernels_per_size_class(algos.crc24_openpgp);
  let crc32_ieee_best = best_kernels_per_size_class(algos.crc32_ieee);
  let crc32c_best = best_kernels_per_size_class(algos.crc32c);
  let crc64_xz_best = best_kernels_per_size_class(algos.crc64_xz);
  let crc64_nvme_best = best_kernels_per_size_class(algos.crc64_nvme);

  let crc16_ccitt_kernel = crc16_ccitt_best
    .get(size_class)
    .map(|s| s.as_str())
    .unwrap_or("portable");
  let crc16_ibm_kernel = crc16_ibm_best.get(size_class).map(|s| s.as_str()).unwrap_or("portable");
  let crc24_openpgp_kernel = crc24_openpgp_best
    .get(size_class)
    .map(|s| s.as_str())
    .unwrap_or("portable");
  let crc32_ieee_kernel = crc32_ieee_best
    .get(size_class)
    .map(|s| s.as_str())
    .unwrap_or("portable");
  let crc32c_kernel = crc32c_best.get(size_class).map(|s| s.as_str()).unwrap_or("portable");
  let crc64_xz_kernel = crc64_xz_best.get(size_class).map(|s| s.as_str()).unwrap_or("portable");
  let crc64_nvme_kernel = crc64_nvme_best
    .get(size_class)
    .map(|s| s.as_str())
    .unwrap_or("portable");

  format!(
    "{indent}KernelSet {{\n{indent}  crc16_ccitt: {crc16_ccitt},\n{indent}  crc16_ibm: {crc16_ibm},\n{indent}  \
     crc24_openpgp: {crc24_openpgp},\n{indent}  crc32_ieee: {crc32_ieee},\n{indent}  crc32c: {crc32c},\n{indent}  \
     crc64_xz: {crc64_xz},\n{indent}  crc64_nvme: {crc64_nvme},\n{indent}}}",
    crc16_ccitt = mapper.get_constant(crc16_ccitt_kernel, "crc16-ccitt"),
    crc16_ibm = mapper.get_constant(crc16_ibm_kernel, "crc16-ibm"),
    crc24_openpgp = mapper.get_constant(crc24_openpgp_kernel, "crc24-openpgp"),
    crc32_ieee = mapper.get_constant(crc32_ieee_kernel, "crc32-ieee"),
    crc32c = mapper.get_constant(crc32c_kernel, "crc32c"),
    crc64_xz = mapper.get_constant(crc64_xz_kernel, "crc64-xz"),
    crc64_nvme = mapper.get_constant(crc64_nvme_kernel, "crc64-nvme"),
  )
}

/// Generate a complete KernelTable entry.
fn generate_kernel_table(tune_kind: TuneKind, algos: &AlgoSet<'_>) -> String {
  let mapper = KernelMapper::new();
  let kind_name = format!("{tune_kind:?}");
  let table_name = format!("{}_TABLE", kind_name.to_uppercase());

  let xs_set = generate_kernel_set("xs", algos, &mapper, "    ");
  let s_set = generate_kernel_set("s", algos, &mapper, "    ");
  let m_set = generate_kernel_set("m", algos, &mapper, "    ");
  let l_set = generate_kernel_set("l", algos, &mapper, "    ");

  format!(
    "// ───────────────────────────────────────────────────────────────────────────
// {kind_name} Table
//
// Generated by rscrypto-tune. Do not edit manually.
// ───────────────────────────────────────────────────────────────────────────
pub static {table_name}: KernelTable = KernelTable {{
  boundaries: [{SIZE_CLASS_XS_MAX}, {SIZE_CLASS_S_MAX}, {SIZE_CLASS_M_MAX}],

  xs: {xs_set},

  s: {s_set},

  m: {m_set},

  l: {l_set},
}};
"
  )
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
      "cshake128",
      "cshake256",
      "kmac128",
      "kmac256",
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

/// Insert or update a generated table in dispatch.rs.
///
/// The generated table is placed between BEGIN/END markers in the
/// appropriate architecture section.
fn update_dispatch_file(content: &str, tune_kind: TuneKind, table_code: &str) -> io::Result<String> {
  let kind_name = format!("{tune_kind:?}");

  // Look for existing marker for this TuneKind
  let section_marker = format!("// {kind_name} Table");

  // If we have an existing section for this TuneKind, replace it
  if let Some(start_idx) = content.find(&section_marker) {
    // Find the end of this table (next "pub static" or end of module)
    let rest = &content[start_idx..];
    let end_offset = rest
      .find("\n\npub static ")
      .or_else(|| rest.find("\n\n// ───"))
      .or_else(|| rest.find("\n\n#[cfg"))
      .unwrap_or(rest.len());

    let before = &content[..start_idx];
    let after = &content[start_idx.strict_add(end_offset)..];

    return Ok(format!("{before}{table_code}{after}"));
  }

  // No existing section - append to the appropriate architecture module
  // For now, just print the generated code for manual insertion
  eprintln!("\n=== Generated KernelTable for {kind_name} ===");
  eprintln!("{table_code}");
  eprintln!("=== End Generated Code ===\n");
  eprintln!(
    "Please manually insert the above code into the appropriate section of:\n  crates/checksum/src/dispatch.rs"
  );

  Ok(content.to_string())
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

#[cfg(test)]
mod tests {
  use std::path::Path;

  use platform::TuneKind;

  use super::{generate_blake3_streaming_table, generate_hash_table};
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
}
