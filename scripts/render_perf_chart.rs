//! Render the README performance scorecard from `benchmark_results/OVERVIEW.md`.
//!
//! Reads the Linux CI headline, README category geomeans, and macOS local
//! snapshot from `benchmark_results/OVERVIEW.md`, then writes the SVG used by
//! the README at `assets/readme/perf.svg`.
//!
//! Run via `just chart` from the repository root.
//!
//! No external dependencies: pure `std`.

#![deny(warnings)]

use std::{fs, path::PathBuf, process::ExitCode};

const OVERVIEW_PATH: &str = "benchmark_results/OVERVIEW.md";
const OUT_PATH: &str = "assets/readme/perf.svg";
const README_PATH: &str = "README.md";

const WIDTH: u32 = 900;
const HEIGHT: u32 = 670;

const BG: &str = "#000000";
const TITLE: &str = "#e6edf3";
const TEXT: &str = "#c9d1d9";
const MUTED: &str = "#7d8590";
const RULE: &str = "#30363d";
const BLUE: &str = "#0090FF";
const TRACK: &str = "#21262d";

const CHART_TITLE: &str = "rscrypto";
const CHART_SUBTITLE: &str = "Geomean speedups vs fastest matched competitors. Higher is better.";
const SUMMARY_LINUX_LABEL: &str = "Linux";
const SUMMARY_APPLE_LABEL: &str = "Apple Silicon";
const CHECKSUM_TITLE: &str = "Checksums";
const CHECKSUM_COMPETITORS: &str = "- Competitor Crates/Libs: crc-fast, crc, crc32fast, crc32c, crc64fast";
const GROUP_TITLE: &str = "Primitive Geomeans";
const FOOTER_LINUX_RUNNERS: &str = concat!(
  "- Linux Runners: AMD Zen 4/5; Intel Sapphire Rapids/Ice Lake; ",
  "AWS Graviton 3/4; IBM POWER 10 and IBM Z16 (s390x); Rise RISC-V",
);
const FOOTER_MACOS: &str = "- macOS: MBP M1 10-Core, 16GB RAM - Local Dev Box";
const FOOTER_FASTEST_EXTERNAL: &str =
  "- Fastest External: aws-lc-rs, ring, RustCrypto, BLAKE3, libcrux, crc-fast, etc.";

const STYLES: &str = r#"<style>
text.sans { font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
text.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace; }
text.hero { letter-spacing: -2px; paint-order: stroke fill; stroke: #0090FF; stroke-width: 0.7px; }
</style>"#;

struct ReadmeRow {
  label: &'static str,
  line_starts_with: &'static str,
}

const GROUP_ROWS: &[ReadmeRow] = &[
  ReadmeRow {
    label: "AEAD",
    line_starts_with: "- **AEAD:**",
  },
  ReadmeRow {
    label: "RSA",
    line_starts_with: "- **RSA:**",
  },
  ReadmeRow {
    label: "ECDSA",
    line_starts_with: "- **ECDSA P-256/P-384:**",
  },
  ReadmeRow {
    label: "Hashes/MAC/XOF",
    line_starts_with: "- **Hashes/MACs/XOFs:**",
  },
  ReadmeRow {
    label: "Auth/KDF",
    line_starts_with: "- **Auth/KDF:**",
  },
  ReadmeRow {
    label: "Password Hashing",
    line_starts_with: "- **Password hashing:**",
  },
  ReadmeRow {
    label: "ML-KEM",
    line_starts_with: "- **ML-KEM:**",
  },
];

#[derive(Clone, Copy)]
struct ScopeMetric {
  pairs: u64,
  wins: u64,
  ties: u64,
  geomean: f64,
}

struct ChartData {
  linux_fastest: ScopeMetric,
  macos_fastest: ScopeMetric,
  checksum_geomean: f64,
  rows: Vec<(&'static str, f64)>,
}

fn main() -> ExitCode {
  match run() {
    Ok(()) => ExitCode::SUCCESS,
    Err(e) => {
      eprintln!("render_perf_chart: {e}");
      ExitCode::FAILURE
    }
  }
}

fn run() -> Result<(), String> {
  let overview =
    fs::read_to_string(OVERVIEW_PATH).map_err(|e| format!("read {OVERVIEW_PATH}: {e}\n(run from repo root)"))?;
  let data = parse_chart_data(&overview)?;
  let svg = render_svg(&data);
  validate_svg_contract(&svg, &data)?;
  let readme_update = readme_with_updated_alt(&data)?;
  write_file(OUT_PATH, &svg)?;
  if let Some(readme) = readme_update {
    write_file(README_PATH, &readme)?;
    eprintln!("updated {README_PATH} benchmark chart alt text");
  }
  eprintln!("wrote {OUT_PATH} ({} bytes)", svg.len());
  Ok(())
}

fn parse_chart_data(content: &str) -> Result<ChartData, String> {
  let linux_fastest = extract_table_metric(content, "Linux CI: fastest external per case")?;
  let macos_fastest = extract_table_metric(content, "macOS local: fastest external per case")?;
  let checksum_geomean = extract_readme_geomean(content, "- **Checksums:**")?;

  let rows = GROUP_ROWS
    .iter()
    .map(|row| extract_readme_geomean(content, row.line_starts_with).map(|value| (row.label, value)))
    .collect::<Result<Vec<_>, _>>()?;

  let data = ChartData {
    linux_fastest,
    macos_fastest,
    checksum_geomean,
    rows,
  };
  validate_chart_data(&data)?;
  Ok(data)
}

fn write_file(path: &str, contents: &str) -> Result<(), String> {
  let path_buf = PathBuf::from(path);
  if let Some(parent) = path_buf.parent() {
    fs::create_dir_all(parent).map_err(|e| format!("create {}: {}", parent.display(), e))?;
  }
  fs::write(&path_buf, contents).map_err(|e| format!("write {path}: {e}"))
}

fn readme_with_updated_alt(data: &ChartData) -> Result<Option<String>, String> {
  let readme = fs::read_to_string(README_PATH).map_err(|e| format!("read {README_PATH}: {e}"))?;
  let src_marker = "\n       src=\"assets/readme/perf.svg\"";
  let src_idx = readme
    .find(src_marker)
    .ok_or_else(|| format!("{README_PATH}: missing benchmark chart image src"))?;

  let alt_marker = "  <img alt=\"";
  let alt_start = readme[..src_idx]
    .rfind(alt_marker)
    .ok_or_else(|| format!("{README_PATH}: missing benchmark chart alt text"))?;
  let alt_value_start = alt_start + alt_marker.len();
  let alt_value_end = src_idx
    .checked_sub(1)
    .ok_or_else(|| format!("{README_PATH}: malformed benchmark chart alt text"))?;
  if !readme[alt_value_end..src_idx].starts_with('"') {
    return Err(format!("{README_PATH}: malformed benchmark chart alt text"));
  }

  let alt = escape_xml(&render_readme_alt(data));
  if readme[alt_value_start..alt_value_end] == alt {
    return Ok(None);
  }

  let mut updated = String::with_capacity(readme.len() + alt.len());
  updated.push_str(&readme[..alt_value_start]);
  updated.push_str(&alt);
  updated.push_str(&readme[alt_value_end..]);
  Ok(Some(updated))
}

fn extract_readme_geomean(content: &str, line_starts_with: &str) -> Result<f64, String> {
  let line = content
    .lines()
    .find(|line| line.trim_start().starts_with(line_starts_with))
    .ok_or_else(|| format!("OVERVIEW.md: missing README Numbers bullet `{line_starts_with}`"))?;

  first_speedup(line).ok_or_else(|| format!("OVERVIEW.md: no speedup token in `{line_starts_with}`"))
}

fn extract_table_metric(content: &str, label: &str) -> Result<ScopeMetric, String> {
  let needle = format!("| {label} |");
  let line = content
    .lines()
    .find(|line| line.starts_with(&needle))
    .ok_or_else(|| format!("OVERVIEW.md: missing table row `{label}`"))?;
  let cells = markdown_cells(line);
  if cells.len() < 6 {
    return Err(format!(
      "OVERVIEW.md: table row `{label}` has {} cells, expected at least 6",
      cells.len()
    ));
  }

  let pairs = first_integer(cells[1]).ok_or_else(|| format!("OVERVIEW.md: `{label}` missing pair count"))?;
  let (wins, ties, losses) = parse_wtl(cells[2]).ok_or_else(|| format!("OVERVIEW.md: `{label}` missing W/T/L"))?;
  let wtl_total = wins
    .checked_add(ties)
    .and_then(|n| n.checked_add(losses))
    .ok_or_else(|| format!("OVERVIEW.md: `{label}` W/T/L count overflow"))?;
  if wtl_total != pairs {
    return Err(format!(
      "OVERVIEW.md: `{label}` has {pairs} pairs but W/T/L totals {wtl_total}"
    ));
  }
  let geomean = first_speedup(cells[4]).ok_or_else(|| format!("OVERVIEW.md: `{label}` missing geomean"))?;

  Ok(ScopeMetric {
    pairs,
    wins,
    ties,
    geomean,
  })
}

fn markdown_cells(line: &str) -> Vec<&str> {
  line.trim().trim_matches('|').split('|').map(str::trim).collect()
}

fn parse_wtl(cell: &str) -> Option<(u64, u64, u64)> {
  let mut parts = cell.split('/');
  let wins = parse_u64(parts.next()?)?;
  let ties = parse_u64(parts.next()?)?;
  let losses = parse_u64(parts.next()?)?;
  Some((wins, ties, losses))
}

fn first_speedup(line: &str) -> Option<f64> {
  find_speedups(line).into_iter().next()
}

fn first_integer(line: &str) -> Option<u64> {
  find_integers(line).into_iter().next()
}

fn parse_u64(s: &str) -> Option<u64> {
  s.chars().filter(|ch| *ch != ',').collect::<String>().parse().ok()
}

fn find_speedups(line: &str) -> Vec<f64> {
  let bytes = line.as_bytes();
  let mut out = Vec::new();
  let mut i = 0;
  while i < bytes.len() {
    if !bytes[i].is_ascii_digit() {
      i = i.saturating_add(1);
      continue;
    }
    let start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
      i = i.saturating_add(1);
    }
    if i >= bytes.len() || bytes[i] != b'.' {
      continue;
    }
    i = i.saturating_add(1);
    let frac_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
      i = i.saturating_add(1);
    }
    if i == frac_start || i >= bytes.len() || bytes[i] != b'x' {
      continue;
    }
    let s = std::str::from_utf8(&bytes[start..i]).expect("ASCII speedup token");
    if let Ok(v) = s.parse::<f64>() {
      out.push(v);
    }
    i = i.saturating_add(1);
  }
  out
}

fn find_integers(line: &str) -> Vec<u64> {
  let bytes = line.as_bytes();
  let mut out = Vec::new();
  let mut i = 0;
  while i < bytes.len() {
    if !bytes[i].is_ascii_digit() {
      i = i.saturating_add(1);
      continue;
    }
    let mut value = 0_u64;
    let mut saw_digit = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
      value = value
        .saturating_mul(10)
        .saturating_add(u64::from(bytes[i].saturating_sub(b'0')));
      saw_digit = true;
      i = i.saturating_add(1);
      while i + 1 < bytes.len() && bytes[i] == b',' && bytes[i + 1].is_ascii_digit() {
        i = i.saturating_add(1);
        value = value
          .saturating_mul(10)
          .saturating_add(u64::from(bytes[i].saturating_sub(b'0')));
        i = i.saturating_add(1);
      }
    }
    if i < bytes.len() && (bytes[i] == b'.' || bytes[i] == b'x') {
      continue;
    }
    if saw_digit {
      out.push(value);
    }
  }
  out
}

fn format_thousands(n: u64) -> String {
  let s = n.to_string();
  let mut buf = Vec::with_capacity(s.len() + s.len() / 3);
  for (idx, ch) in s.chars().rev().enumerate() {
    if idx > 0 && idx % 3 == 0 {
      buf.push(',');
    }
    buf.push(ch);
  }
  buf.into_iter().rev().collect()
}

fn format_summary_detail(metric: ScopeMetric) -> String {
  let wins_or_ties = metric
    .wins
    .checked_add(metric.ties)
    .expect("metric wins + ties validated");
  format!(
    "{} wins | {} wins/ties | {} cases",
    format_thousands(metric.wins),
    format_thousands(wins_or_ties),
    format_thousands(metric.pairs),
  )
}

fn render_readme_alt(data: &ChartData) -> String {
  format!(
    "rscrypto benchmark chart: {:.2}x Linux and {:.2}x Apple Silicon fastest-matched geomeans, checksums at {:.2}x \
     against crc-fast, crc, crc32fast, crc32c, and crc64fast, plus primitive geomean bars and M1 MBP Apple Silicon \
     notes.",
    data.linux_fastest.geomean, data.macos_fastest.geomean, data.checksum_geomean,
  )
}

fn escape_xml(s: &str) -> String {
  let mut out = String::with_capacity(s.len());
  for c in s.chars() {
    match c {
      '<' => out.push_str("&lt;"),
      '>' => out.push_str("&gt;"),
      '&' => out.push_str("&amp;"),
      '"' => out.push_str("&quot;"),
      '\'' => out.push_str("&apos;"),
      _ => out.push(c),
    }
  }
  out
}

fn validate_chart_data(data: &ChartData) -> Result<(), String> {
  validate_metric("Linux fastest external", data.linux_fastest)?;
  validate_metric("macOS fastest external", data.macos_fastest)?;
  validate_speedup("Checksums", data.checksum_geomean)?;

  if data.rows.len() != GROUP_ROWS.len() {
    return Err(format!(
      "chart data has {} primitive rows, expected {}",
      data.rows.len(),
      GROUP_ROWS.len()
    ));
  }

  for (label, value) in &data.rows {
    validate_speedup(label, *value)?;
  }

  Ok(())
}

fn validate_metric(label: &str, metric: ScopeMetric) -> Result<(), String> {
  if metric.pairs == 0 {
    return Err(format!("{label}: pair count must be nonzero"));
  }
  let wins_or_ties = metric
    .wins
    .checked_add(metric.ties)
    .ok_or_else(|| format!("{label}: wins + ties overflow"))?;
  if wins_or_ties > metric.pairs {
    return Err(format!(
      "{label}: wins + ties ({wins_or_ties}) exceeds pairs ({})",
      metric.pairs
    ));
  }
  validate_speedup(label, metric.geomean)
}

fn validate_speedup(label: &str, value: f64) -> Result<(), String> {
  if value.is_finite() && value > 0.0 {
    Ok(())
  } else {
    Err(format!("{label}: speedup must be a positive finite value, got {value}"))
  }
}

fn validate_svg_contract(svg: &str, data: &ChartData) -> Result<(), String> {
  require_svg_token(svg, &format!("viewBox=\"0 0 {WIDTH} {HEIGHT}\""))?;
  require_svg_token(svg, &format!("fill=\"{BG}\""))?;
  require_svg_token(svg, &format!("fill=\"{BLUE}\""))?;
  require_svg_token(svg, "text.hero")?;
  require_svg_token(svg, "font-size=\"76\" font-weight=\"900\"")?;

  for token in [
    CHART_TITLE,
    CHART_SUBTITLE,
    SUMMARY_LINUX_LABEL,
    SUMMARY_APPLE_LABEL,
    CHECKSUM_TITLE,
    CHECKSUM_COMPETITORS,
    GROUP_TITLE,
    FOOTER_LINUX_RUNNERS,
    FOOTER_MACOS,
    FOOTER_FASTEST_EXTERNAL,
  ] {
    require_svg_text(svg, token)?;
  }

  require_svg_text(svg, &format!("{:.2}x", data.linux_fastest.geomean))?;
  require_svg_text(svg, &format_summary_detail(data.linux_fastest))?;
  require_svg_text(svg, &format!("{:.2}x", data.macos_fastest.geomean))?;
  require_svg_text(svg, &format_summary_detail(data.macos_fastest))?;
  require_svg_text(svg, &format!("{:.2}x", data.checksum_geomean))?;

  for (label, value) in &data.rows {
    require_svg_text(svg, label)?;
    require_svg_text(svg, &format!("{value:.2}x"))?;
  }

  if svg.contains("OpenSSL/BoringSSL") {
    return Err("SVG contract contains stale direct OpenSSL/BoringSSL benchmark copy".to_string());
  }

  Ok(())
}

fn require_svg_text(svg: &str, value: &str) -> Result<(), String> {
  require_svg_token(svg, &escape_xml(value))
}

fn require_svg_token(svg: &str, token: &str) -> Result<(), String> {
  if svg.contains(token) {
    Ok(())
  } else {
    Err(format!("SVG contract missing `{token}`"))
  }
}

fn render_svg(data: &ChartData) -> String {
  let mut svg = String::new();
  svg.push_str(&format!(
    "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {WIDTH} {HEIGHT}\">"
  ));
  svg.push_str(STYLES);
  svg.push_str(&format!(
    "<rect x=\"0\" y=\"0\" width=\"{WIDTH}\" height=\"{HEIGHT}\" fill=\"{BG}\" rx=\"8\"/>"
  ));

  render_summary(&mut svg, data);
  render_checksum_bar(&mut svg, data);
  render_group_bars(&mut svg, data);
  render_evidence_bullets(&mut svg);

  svg.push_str("</svg>");
  svg
}

fn render_summary(svg: &mut String, data: &ChartData) {
  text(svg, 48.0, 42.0, 28, 850, TITLE, CHART_TITLE);
  text(svg, 48.0, 68.0, 13, 500, MUTED, CHART_SUBTITLE);

  svg.push_str(&format!(
    "<line x1=\"48\" y1=\"88\" x2=\"852\" y2=\"88\" stroke=\"{RULE}\" stroke-width=\"1\"/>"
  ));

  render_geomean(
    svg,
    48.0,
    SUMMARY_LINUX_LABEL,
    &format!("{:.2}x", data.linux_fastest.geomean),
    &format_summary_detail(data.linux_fastest),
  );

  render_geomean(
    svg,
    520.0,
    SUMMARY_APPLE_LABEL,
    &format!("{:.2}x", data.macos_fastest.geomean),
    &format_summary_detail(data.macos_fastest),
  );

  svg.push_str(&format!(
    "<line x1=\"450\" y1=\"116\" x2=\"450\" y2=\"222\" stroke=\"{RULE}\" stroke-width=\"1\"/>"
  ));
  svg.push_str(&format!(
    "<line x1=\"48\" y1=\"246\" x2=\"852\" y2=\"246\" stroke=\"{RULE}\" stroke-width=\"1\"/>"
  ));
}

fn render_geomean(svg: &mut String, x: f64, label: &str, value: &str, detail: &str) {
  text(svg, x, 120.0, 15, 750, TITLE, label);
  mono_class(svg, x, 188.0, 76, 900, BLUE, "mono hero", value);
  mono(svg, x + 4.0, 216.0, 13, 550, MUTED, detail);
}

fn render_checksum_bar(svg: &mut String, data: &ChartData) {
  let bar_x = 48.0;
  let bar_w = 804.0;
  let checksum_max = 6.0_f64;
  let width = (data.checksum_geomean / checksum_max).clamp(0.0, 1.0) * bar_w;

  text(svg, 48.0, 276.0, 19, 800, TITLE, CHECKSUM_TITLE);
  mono(
    svg,
    790.0,
    276.0,
    18,
    900,
    TITLE,
    &format!("{:.2}x", data.checksum_geomean),
  );
  svg.push_str(&format!(
    "<rect x=\"{bar_x:.1}\" y=\"296\" width=\"{bar_w:.1}\" height=\"8\" fill=\"{TRACK}\" rx=\"4\"/>"
  ));
  svg.push_str(&format!(
    "<rect x=\"{bar_x:.1}\" y=\"296\" width=\"{width:.1}\" height=\"8\" fill=\"{BLUE}\" rx=\"4\"/>"
  ));
  mono(svg, 48.0, 324.0, 11, 550, MUTED, CHECKSUM_COMPETITORS);
  svg.push_str(&format!(
    "<line x1=\"48\" y1=\"344\" x2=\"852\" y2=\"344\" stroke=\"{RULE}\" stroke-width=\"1\"/>"
  ));
}

fn render_group_bars(svg: &mut String, data: &ChartData) {
  text(svg, 48.0, 370.0, 19, 800, TITLE, GROUP_TITLE);

  let label_x = 48.0;
  let bar_x = 260.0;
  let bar_w = 430.0;
  let value_x = 724.0;
  let row_y = 398.0;
  let row_gap = 21.0;
  let bar_h = 11.0;
  let max = 1.60_f64;

  for (idx, (label, value)) in data.rows.iter().enumerate() {
    let y = row_y + (idx as f64) * row_gap;
    let clamped = value.clamp(1.0, max);
    let width = ((clamped - 1.0) / (max - 1.0)) * bar_w;

    text(svg, label_x, y + 10.0, 13, 700, TEXT, label);
    svg.push_str(&format!(
      "<rect x=\"{bar_x:.1}\" y=\"{y:.1}\" width=\"{bar_w:.1}\" height=\"{bar_h:.1}\" fill=\"{TRACK}\" rx=\"3\"/>"
    ));
    svg.push_str(&format!(
      "<rect x=\"{bar_x:.1}\" y=\"{y:.1}\" width=\"{width:.1}\" height=\"{bar_h:.1}\" fill=\"{BLUE}\" rx=\"3\"/>"
    ));
    mono(svg, value_x, y + 10.0, 15, 850, TITLE, &format!("{value:.2}x"));
  }
}

fn render_evidence_bullets(svg: &mut String) {
  svg.push_str(&format!(
    "<line x1=\"48\" y1=\"558\" x2=\"852\" y2=\"558\" stroke=\"{RULE}\" stroke-width=\"1\"/>"
  ));
  text(svg, 48.0, 578.0, 10, 500, MUTED, FOOTER_LINUX_RUNNERS);
  text(svg, 48.0, 600.0, 10, 500, MUTED, FOOTER_MACOS);
  text(svg, 48.0, 622.0, 10, 500, MUTED, FOOTER_FASTEST_EXTERNAL);
}

fn text(svg: &mut String, x: f64, y: f64, size: u32, weight: u32, fill: &str, value: &str) {
  text_anchor(svg, x, y, size, weight, fill, "start", value);
}

fn mono(svg: &mut String, x: f64, y: f64, size: u32, weight: u32, fill: &str, value: &str) {
  text_anchor_class(svg, x, y, size, weight, fill, "start", "mono", value);
}

fn mono_class(svg: &mut String, x: f64, y: f64, size: u32, weight: u32, fill: &str, class: &str, value: &str) {
  text_anchor_class(svg, x, y, size, weight, fill, "start", class, value);
}

fn text_anchor(svg: &mut String, x: f64, y: f64, size: u32, weight: u32, fill: &str, anchor: &str, value: &str) {
  text_anchor_class(svg, x, y, size, weight, fill, anchor, "sans", value);
}

fn text_anchor_class(
  svg: &mut String,
  x: f64,
  y: f64,
  size: u32,
  weight: u32,
  fill: &str,
  anchor: &str,
  class: &str,
  value: &str,
) {
  svg.push_str(&format!(
    "<text class=\"{class}\" x=\"{x:.1}\" y=\"{y:.1}\" text-anchor=\"{anchor}\" font-size=\"{size}\" \
     font-weight=\"{weight}\" fill=\"{fill}\">{}</text>",
    escape_xml(value)
  ));
}
