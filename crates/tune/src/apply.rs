//! Apply `rscrypto-tune` results into baked-in tuned defaults.
//!
//! This is the "pipeline" step: a machine can run `rscrypto-tune --apply`,
//! and the repo's `tuned_defaults.rs` tables get updated for that platform's
//! [`platform::TuneKind`].

use std::{
  fs, io,
  path::{Path, PathBuf},
};

use platform::TuneKind;

use crate::{AlgorithmResult, TuneResults};

const BEGIN: &str = "// BEGIN GENERATED (rscrypto-tune)";
const END: &str = "// END GENERATED (rscrypto-tune)";

#[derive(Clone, Copy, Debug)]
struct Thresholds<'a> {
  algo: &'a AlgorithmResult,
}

impl<'a> Thresholds<'a> {
  #[inline]
  #[must_use]
  fn new(algo: &'a AlgorithmResult) -> Self {
    Self { algo }
  }

  #[must_use]
  fn usize_value(self, env_suffix: &str) -> Option<usize> {
    self
      .algo
      .thresholds
      .iter()
      .find_map(|(k, v)| if k == env_suffix { Some(*v) } else { None })
  }

  #[must_use]
  fn option_usize(self, env_suffix: &str) -> Option<usize> {
    let v = self.usize_value(env_suffix)?;
    if v == usize::MAX { None } else { Some(v) }
  }
}

#[derive(Clone, Debug)]
struct AlgoSet<'a> {
  crc16_ccitt: &'a AlgorithmResult,
  crc16_ibm: &'a AlgorithmResult,
  crc24_openpgp: &'a AlgorithmResult,
  crc32_ieee: &'a AlgorithmResult,
  crc32c: &'a AlgorithmResult,
  crc64_xz: &'a AlgorithmResult,
  crc64_nvme: &'a AlgorithmResult,
}

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

#[must_use]
fn tune_kind_ident(kind: TuneKind) -> String {
  format!("{kind:?}")
}

fn tuned_defaults_path(repo_root: &Path, rel: &str) -> PathBuf {
  repo_root.join(rel)
}

fn read(path: &Path) -> io::Result<String> {
  fs::read_to_string(path)
}

fn write(path: &Path, contents: &str) -> io::Result<()> {
  fs::write(path, contents)
}

fn find_section_bounds(haystack: &str, section_anchor: &str) -> io::Result<(usize, usize)> {
  let Some(anchor_idx) = haystack.find(section_anchor) else {
    return Err(io::Error::new(
      io::ErrorKind::InvalidData,
      format!("missing section anchor: {section_anchor}"),
    ));
  };

  let Some(begin_rel) = haystack[anchor_idx..].find(BEGIN) else {
    return Err(io::Error::new(io::ErrorKind::InvalidData, "missing BEGIN marker"));
  };
  let begin_idx = anchor_idx.strict_add(begin_rel);

  let Some(end_rel) = haystack[begin_idx..].find(END) else {
    return Err(io::Error::new(io::ErrorKind::InvalidData, "missing END marker"));
  };
  let end_idx = begin_idx.strict_add(end_rel);

  Ok((begin_idx, end_idx))
}

fn line_start(s: &str, idx: usize) -> usize {
  s[..idx].rfind('\n').map(|i| i + 1).unwrap_or(0)
}

fn line_end(s: &str, idx: usize) -> usize {
  s[idx..]
    .find('\n')
    .map(|i| idx.strict_add(i))
    .unwrap_or_else(|| s.len())
}

fn trim_block(s: &str) -> &str {
  s.trim_matches('\n')
}

fn parse_entries(block: &str) -> Vec<(String, core::ops::Range<usize>)> {
  let mut entries = Vec::new();
  let mut i: usize = 0;

  while let Some(rel) = block[i..].find("(TuneKind::") {
    let start = i.strict_add(rel);

    // Scan to the matching ')' of the tuple.
    let mut depth: i32 = 0;
    let mut end: Option<usize> = None;

    for (off, ch) in block[start..].char_indices() {
      match ch {
        '(' => depth = depth.strict_add(1),
        ')' => {
          depth = depth.strict_sub(1);
          if depth == 0 {
            end = Some(start.strict_add(off + 1));
            break;
          }
        }
        _ => {}
      }
    }

    let Some(tuple_end) = end else { break };
    let mut entry_end = tuple_end;

    // Include the trailing comma (and one newline) if present.
    if block[entry_end..].starts_with(',') {
      entry_end = entry_end.strict_add(1);
    }
    if block[entry_end..].starts_with('\n') {
      entry_end = entry_end.strict_add(1);
    }

    // Extract kind identifier.
    let tuple = &block[start..tuple_end];
    let kind = tuple
      .split("TuneKind::")
      .nth(1)
      .and_then(|rest| rest.split(|c: char| !c.is_ascii_alphanumeric()).next())
      .unwrap_or("Unknown");

    entries.push((kind.to_string(), start..entry_end));
    i = entry_end;
  }

  entries
}

fn extend_entry_start_to_comments(block: &str, entry_start: usize) -> usize {
  let mut start = entry_start;
  loop {
    let ls = line_start(block, start);
    if ls == 0 {
      break;
    }
    let prev_end = ls.strict_sub(1);
    let prev_ls = line_start(block, prev_end);
    let prev_line = &block[prev_ls..line_end(block, prev_ls)];
    let prev_line = prev_line.trim_end_matches('\n');

    if prev_line.trim().is_empty() {
      break;
    }
    if prev_line.trim_start().starts_with("//") {
      if prev_line.contains(BEGIN) {
        break;
      }
      start = prev_ls;
      continue;
    }
    break;
  }
  start
}

fn upsert_tunekind_entry(block: &str, kind_ident: &str, entry_with_comment: &str) -> String {
  let mut out = String::new();
  let raw = trim_block(block);
  let entries = parse_entries(raw);

  // Find and replace if present.
  for (kind, range) in &entries {
    if kind == kind_ident {
      let start = extend_entry_start_to_comments(raw, range.start);
      out.push_str(&raw[..start]);
      out.push_str(entry_with_comment);
      out.push_str(&raw[range.end..]);
      out.push('\n');
      return out;
    }
  }

  // Not present: insert before END, after the last entry (or after BEGIN).
  if let Some((_, last)) = entries.last() {
    out.push_str(&raw[..last.end]);
    out.push_str(entry_with_comment);
    out.push_str(&raw[last.end..]);
    out.push('\n');
    return out;
  }

  // No entries at all.
  out.push_str(raw);
  if !raw.ends_with('\n') {
    out.push('\n');
  }
  out.push_str(entry_with_comment);
  out.push('\n');
  out
}

fn update_section(haystack: &str, section_anchor: &str, kind_ident: &str, entry: &str) -> io::Result<String> {
  let (begin_idx, end_idx) = find_section_bounds(haystack, section_anchor)?;

  let block_start = line_end(haystack, begin_idx);
  let block_end = line_start(haystack, end_idx);
  let before = &haystack[..block_start];
  let block = &haystack[block_start..block_end];
  let after = &haystack[block_end..];

  let updated_block = upsert_tunekind_entry(block, kind_ident, entry);
  Ok(format!("{before}{updated_block}{after}"))
}

fn fmt_opt(v: Option<usize>) -> String {
  match v {
    Some(v) => format!("Some({v})"),
    None => "None".to_string(),
  }
}

fn fmt_usize(v: usize) -> String {
  if v == usize::MAX {
    "usize::MAX".to_string()
  } else {
    v.to_string()
  }
}

fn crc16_entry(kind_ident: &str, algo: &AlgorithmResult) -> String {
  let t = Thresholds::new(algo);
  let slice4_to_slice8 = t.usize_value("THRESHOLD_SLICE4_TO_SLICE8").unwrap_or(64);
  let portable_to_clmul = t.usize_value("THRESHOLD_PORTABLE_TO_CLMUL").unwrap_or(64);
  let pclmul_to_vpclmul = t.option_usize("THRESHOLD_PCLMUL_TO_VPCLMUL");
  let min_bytes_per_lane = t.option_usize("MIN_BYTES_PER_LANE");
  let streams = algo.recommended_streams;

  format!(
    "  // {kind_ident}: generated by rscrypto-tune\n  (TuneKind::{kind_ident}, Crc16TunedDefaults {{ \
     slice4_to_slice8: {slice4_to_slice8}, portable_to_clmul: {portable_to_clmul}, pclmul_to_vpclmul: {}, streams: \
     {streams}, min_bytes_per_lane: {} }}),\n",
    fmt_opt(pclmul_to_vpclmul),
    fmt_opt(min_bytes_per_lane),
  )
}

fn crc24_entry(kind_ident: &str, algo: &AlgorithmResult) -> String {
  let t = Thresholds::new(algo);
  let slice4_to_slice8 = t.usize_value("THRESHOLD_SLICE4_TO_SLICE8").unwrap_or(64);
  let portable_to_clmul = t.usize_value("THRESHOLD_PORTABLE_TO_CLMUL").unwrap_or(64);
  let pclmul_to_vpclmul = t.option_usize("THRESHOLD_PCLMUL_TO_VPCLMUL");
  let min_bytes_per_lane = t.option_usize("MIN_BYTES_PER_LANE");
  let streams = algo.recommended_streams;

  format!(
    "  // {kind_ident}: generated by rscrypto-tune\n  (TuneKind::{kind_ident}, Crc24TunedDefaults {{ \
     slice4_to_slice8: {slice4_to_slice8}, portable_to_clmul: {portable_to_clmul}, pclmul_to_vpclmul: {}, streams: \
     {streams}, min_bytes_per_lane: {} }}),\n",
    fmt_opt(pclmul_to_vpclmul),
    fmt_opt(min_bytes_per_lane),
  )
}

fn crc32_entry(kind_ident: &str, ieee: &AlgorithmResult, crc32c: &AlgorithmResult) -> String {
  let ieee_t = Thresholds::new(ieee);
  let crc32c_t = Thresholds::new(crc32c);

  let crc32_streams = ieee.recommended_streams;
  let crc32c_streams = crc32c.recommended_streams;

  let crc32_portable_bytewise_to_slice16 = ieee_t
    .usize_value("THRESHOLD_PORTABLE_BYTEWISE_TO_SLICE16")
    .unwrap_or(64);
  let crc32_portable_to_hwcrc = ieee_t.usize_value("THRESHOLD_PORTABLE_TO_HWCRC").unwrap_or(64);
  let crc32_hwcrc_to_fusion = ieee_t.usize_value("THRESHOLD_HWCRC_TO_FUSION").unwrap_or(usize::MAX);
  let crc32_fusion_to_avx512 = ieee_t.usize_value("THRESHOLD_FUSION_TO_AVX512").unwrap_or(usize::MAX);
  let crc32_fusion_to_vpclmul = ieee_t.usize_value("THRESHOLD_FUSION_TO_VPCLMUL").unwrap_or(usize::MAX);
  let crc32_min_bpl = ieee_t.option_usize("MIN_BYTES_PER_LANE");

  let crc32c_portable_bytewise_to_slice16 = crc32c_t
    .usize_value("THRESHOLD_PORTABLE_BYTEWISE_TO_SLICE16")
    .unwrap_or(64);
  let crc32c_portable_to_hwcrc = crc32c_t.usize_value("THRESHOLD_PORTABLE_TO_HWCRC").unwrap_or(64);
  let crc32c_hwcrc_to_fusion = crc32c_t.usize_value("THRESHOLD_HWCRC_TO_FUSION").unwrap_or(usize::MAX);
  let crc32c_fusion_to_avx512 = crc32c_t.usize_value("THRESHOLD_FUSION_TO_AVX512").unwrap_or(usize::MAX);
  let crc32c_fusion_to_vpclmul = crc32c_t
    .usize_value("THRESHOLD_FUSION_TO_VPCLMUL")
    .unwrap_or(usize::MAX);
  let crc32c_min_bpl = crc32c_t.option_usize("MIN_BYTES_PER_LANE");

  format!(
    "  // {kind_ident}: generated by rscrypto-tune\n  (TuneKind::{kind_ident}, Crc32TunedDefaults {{\n    crc32:  \
     Crc32VariantTunedDefaults {{ streams: {crc32_streams}, portable_bytewise_to_slice16: {}, portable_to_hwcrc: {}, \
     hwcrc_to_fusion: {}, fusion_to_avx512: {}, fusion_to_vpclmul: {}, min_bytes_per_lane: {} }},\n    crc32c: \
     Crc32VariantTunedDefaults {{ streams: {crc32c_streams}, portable_bytewise_to_slice16: {}, portable_to_hwcrc: {}, \
     hwcrc_to_fusion: {}, fusion_to_avx512: {}, fusion_to_vpclmul: {}, min_bytes_per_lane: {} }},\n  }}),\n",
    fmt_usize(crc32_portable_bytewise_to_slice16),
    fmt_usize(crc32_portable_to_hwcrc),
    fmt_usize(crc32_hwcrc_to_fusion),
    fmt_usize(crc32_fusion_to_avx512),
    fmt_usize(crc32_fusion_to_vpclmul),
    fmt_opt(crc32_min_bpl),
    fmt_usize(crc32c_portable_bytewise_to_slice16),
    fmt_usize(crc32c_portable_to_hwcrc),
    fmt_usize(crc32c_hwcrc_to_fusion),
    fmt_usize(crc32c_fusion_to_avx512),
    fmt_usize(crc32c_fusion_to_vpclmul),
    fmt_opt(crc32c_min_bpl),
  )
}

pub fn apply_tuned_defaults(repo_root: &Path, results: &TuneResults) -> io::Result<()> {
  let kind_ident = tune_kind_ident(results.platform.tune_kind);
  let a = algos(results)?;

  // CRC-16
  let crc16_path = tuned_defaults_path(repo_root, "crates/checksum/src/crc16/tuned_defaults.rs");
  let mut crc16 = read(&crc16_path)?;
  crc16 = update_section(
    &crc16,
    "pub const CRC16_CCITT_TUNED_DEFAULTS",
    &kind_ident,
    &crc16_entry(&kind_ident, a.crc16_ccitt),
  )?;
  crc16 = update_section(
    &crc16,
    "pub const CRC16_IBM_TUNED_DEFAULTS",
    &kind_ident,
    &crc16_entry(&kind_ident, a.crc16_ibm),
  )?;
  write(&crc16_path, &crc16)?;

  // CRC-24
  let crc24_path = tuned_defaults_path(repo_root, "crates/checksum/src/crc24/tuned_defaults.rs");
  let crc24 = read(&crc24_path)?;
  let crc24 = update_section(
    &crc24,
    "pub const CRC24_TUNED_DEFAULTS",
    &kind_ident,
    &crc24_entry(&kind_ident, a.crc24_openpgp),
  )?;
  write(&crc24_path, &crc24)?;

  // CRC-32
  let crc32_path = tuned_defaults_path(repo_root, "crates/checksum/src/crc32/tuned_defaults.rs");
  let crc32 = read(&crc32_path)?;
  let entry = crc32_entry(&kind_ident, a.crc32_ieee, a.crc32c);
  let crc32 = update_section(&crc32, "pub const CRC32_TUNED_DEFAULTS", &kind_ident, &entry)?;
  write(&crc32_path, &crc32)?;

  // CRC-64
  let crc64_path = tuned_defaults_path(repo_root, "crates/checksum/src/crc64/tuned_defaults.rs");
  let crc64 = read(&crc64_path)?;
  let entry = {
    let xz = Thresholds::new(a.crc64_xz);
    let nvme = Thresholds::new(a.crc64_nvme);

    let xz_streams = a.crc64_xz.recommended_streams;
    let nvme_streams = a.crc64_nvme.recommended_streams;

    let xz_portable_to_clmul = xz.usize_value("THRESHOLD_PORTABLE_TO_CLMUL").unwrap_or(64);
    let nvme_portable_to_clmul = nvme.usize_value("THRESHOLD_PORTABLE_TO_CLMUL").unwrap_or(64);

    let xz_pclmul_to_vpclmul = xz.usize_value("THRESHOLD_PCLMUL_TO_VPCLMUL").unwrap_or(usize::MAX);
    let nvme_pclmul_to_vpclmul = nvme.usize_value("THRESHOLD_PCLMUL_TO_VPCLMUL").unwrap_or(usize::MAX);

    let xz_small_kernel_max_bytes = xz.usize_value("THRESHOLD_SMALL_KERNEL_MAX_BYTES").unwrap_or(512);
    let nvme_small_kernel_max_bytes = nvme.usize_value("THRESHOLD_SMALL_KERNEL_MAX_BYTES").unwrap_or(512);

    let xz_min_bpl = xz.option_usize("MIN_BYTES_PER_LANE");
    let nvme_min_bpl = nvme.option_usize("MIN_BYTES_PER_LANE");

    format!(
      "  // {kind_ident}: generated by rscrypto-tune\n  (TuneKind::{kind_ident}, Crc64TunedDefaults {{\n    xz:   \
       Crc64VariantTunedDefaults {{ streams: {xz_streams}, portable_to_clmul: {}, pclmul_to_vpclmul: {}, \
       small_kernel_max_bytes: {}, min_bytes_per_lane: {} }},\n    nvme: Crc64VariantTunedDefaults {{ streams: \
       {nvme_streams}, portable_to_clmul: {}, pclmul_to_vpclmul: {}, small_kernel_max_bytes: {}, min_bytes_per_lane: \
       {} }},\n  }}),\n",
      fmt_usize(xz_portable_to_clmul),
      fmt_usize(xz_pclmul_to_vpclmul),
      fmt_usize(xz_small_kernel_max_bytes),
      fmt_opt(xz_min_bpl),
      fmt_usize(nvme_portable_to_clmul),
      fmt_usize(nvme_pclmul_to_vpclmul),
      fmt_usize(nvme_small_kernel_max_bytes),
      fmt_opt(nvme_min_bpl),
    )
  };
  let crc64 = update_section(&crc64, "pub const CRC64_TUNED_DEFAULTS", &kind_ident, &entry)?;
  write(&crc64_path, &crc64)?;

  Ok(())
}
