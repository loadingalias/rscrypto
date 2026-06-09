//! Render the README performance chart from `benchmark_results/OVERVIEW.md`.
//!
//! Reads the headline geomean numbers from the "README Numbers" section of
//! `benchmark_results/OVERVIEW.md` and writes a single SVG bar chart to
//! `assets/readme/perf.svg`. The SVG bakes in a single dark hero-card theme
//! (solid background, light text, rust-orange bars) so it renders identically
//! across any embed context — no dependency on `prefers-color-scheme`, which
//! is unreliable for SVGs loaded via `<img src>` (the SVG is its own
//! document and many renderers don't propagate the parent page's theme).
//!
//! Run via `just chart`. Must be invoked from the repo root.
//!
//! No external dependencies — pure `std`.

use std::{fs, path::PathBuf, process::ExitCode};

const OVERVIEW_PATH: &str = "benchmark_results/OVERVIEW.md";
const OUT_PATH: &str = "assets/readme/perf.svg";

/// One row of the bar chart. The geomean value is extracted from the
/// "README Numbers" bullet identified by `line_starts_with`, taking the
/// `skip`-th `X.XXx` token on that line.
struct Row {
  label: &'static str,
  line_starts_with: &'static str,
  skip: usize,
}

/// Sorted descending by expected speedup so the strongest result lands at
/// the top of the chart. If a category drops below another, reorder here.
const ROWS: &[Row] = &[
  Row {
    label: "Checksums",
    line_starts_with: "- **Checksums:**",
    skip: 0,
  },
  Row {
    label: "AEAD",
    line_starts_with: "- **AEAD:**",
    skip: 0,
  },
  Row {
    label: "Hashes/MACs/XOFs",
    line_starts_with: "- **Hashes/MACs/XOFs:**",
    skip: 0,
  },
  Row {
    label: "RSA",
    line_starts_with: "- **RSA:**",
    skip: 0,
  },
  Row {
    label: "Auth/KDF",
    line_starts_with: "- **Auth/KDF:**",
    skip: 0,
  },
  Row {
    label: "Public-key",
    line_starts_with: "- **Public-key:**",
    skip: 0,
  },
  Row {
    label: "Password hashing",
    line_starts_with: "- **Password hashing:**",
    skip: 0,
  },
];

/// Hero-card palette. Solid dark background, light text, brighter rust
/// orange bars. Keep colors aligned with GitHub's dark surface so the chart
/// blends on dark backgrounds and reads as a designed panel on light ones.
const BG: &str = "#0d1117";
const TITLE: &str = "#e6edf3";
const SUBTITLE: &str = "#9ba1a8";
const LABEL: &str = "#e6edf3";
const AXIS: &str = "#7d8590";
const GRID: &str = "#30363d";
const PARITY: &str = "#7d8590";
const BAR: &str = "#f4a460";
const BAR_LABEL: &str = "#e6edf3";
const CAPTION: &str = "#7d8590";

const STYLES: &str =
  r#"<style>text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }</style>"#;

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
    fs::read_to_string(OVERVIEW_PATH).map_err(|e| format!("read {OVERVIEW_PATH}: {e}\n(run from the repo root)"))?;

  let bars: Vec<(String, f64)> = ROWS
    .iter()
    .map(|row| extract_row(&overview, row).map(|v| (row.label.to_string(), v)))
    .collect::<Result<_, _>>()?;

  let (wins, total) = extract_headline(&overview)?;
  let subtitle = format!(
    "Linux CI \u{00B7} {wins} of {total} matched comparisons",
    wins = format_thousands(wins),
    total = format_thousands(total),
  );

  let svg = render_svg(&bars, &subtitle, 3.0_f64);
  write_file(OUT_PATH, &svg)?;

  eprintln!("wrote {OUT_PATH} ({} bytes)", svg.len());
  Ok(())
}

fn write_file(path: &str, contents: &str) -> Result<(), String> {
  let path_buf = PathBuf::from(path);
  if let Some(parent) = path_buf.parent() {
    fs::create_dir_all(parent).map_err(|e| format!("create {}: {}", parent.display(), e))?;
  }
  fs::write(&path_buf, contents).map_err(|e| format!("write {path}: {e}"))
}

fn extract_row(content: &str, row: &Row) -> Result<f64, String> {
  let line = content
    .lines()
    .find(|l| l.trim_start().starts_with(row.line_starts_with))
    .ok_or_else(|| {
      format!(
        "OVERVIEW.md: no bullet starting with `{}` (label `{}`)",
        row.line_starts_with, row.label
      )
    })?;
  let nums = find_speedups(line);
  nums.get(row.skip).copied().ok_or_else(|| {
    format!(
      "OVERVIEW.md: bullet `{}` has only {} `X.XXx` token(s); needed at least {}",
      row.line_starts_with,
      nums.len(),
      row.skip + 1
    )
  })
}

fn extract_headline(content: &str) -> Result<(u64, u64), String> {
  let line = content
    .lines()
    .find(|l| l.contains("matched Linux CI"))
    .ok_or_else(|| "OVERVIEW.md: no `matched Linux CI` headline line".to_string())?;
  let ints = find_integers(line);
  if ints.len() < 2 {
    return Err(format!(
      "OVERVIEW.md: headline line has only {} integers, need 2",
      ints.len()
    ));
  }
  Ok((ints[0], ints[1]))
}

/// Find every `X.XXx` token in `line` and return the parsed numeric values.
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
    if i == frac_start {
      continue;
    }
    if i >= bytes.len() || bytes[i] != b'x' {
      continue;
    }
    let s = std::str::from_utf8(&bytes[start..i]).expect("ASCII digits");
    if let Ok(v) = s.parse::<f64>() {
      out.push(v);
    }
    i = i.saturating_add(1);
  }
  out
}

/// Find every plain integer in `line`, accepting comma group separators
/// (skips digits that are part of a decimal `X.YY` or `X.YYx` token).
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
  let mut buf: Vec<char> = Vec::with_capacity(s.len() + s.len() / 3);
  for (idx, ch) in s.chars().rev().enumerate() {
    if idx > 0 && idx % 3 == 0 {
      buf.push(',');
    }
    buf.push(ch);
  }
  buf.into_iter().rev().collect()
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

fn render_svg(bars: &[(String, f64)], subtitle: &str, max_x: f64) -> String {
  let width: u32 = 700;
  let height: u32 = 370;
  let plot_x_start = 190.0_f64;
  let plot_x_end = 640.0_f64;
  let plot_y_start = 68.0_f64;
  let plot_y_end = 322.0_f64;
  let plot_w = plot_x_end - plot_x_start;

  let bar_count = bars.len() as f64;
  let bar_height = 22.0_f64;
  let bar_gap = 12.0_f64;
  let bars_total = bar_height * bar_count + bar_gap * (bar_count - 1.0);
  let top_pad = ((plot_y_end - plot_y_start) - bars_total) / 2.0;

  let mut svg = String::new();
  svg.push_str(&format!(
    "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\">"
  ));
  svg.push_str(STYLES);

  // Background hero card. Inline fill keeps colors stable across renderers
  // even if `<style>` blocks are stripped.
  svg.push_str(&format!(
    "<rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"{BG}\" rx=\"6\"/>"
  ));

  // Title.
  svg.push_str(&format!(
    "<text x=\"{cx}\" y=\"24\" text-anchor=\"middle\" font-size=\"15\" font-weight=\"600\" fill=\"{TITLE}\">Speedup \
     vs the strongest Rust baseline (geomean)</text>",
    cx = width / 2,
  ));

  // Subtitle.
  svg.push_str(&format!(
    "<text x=\"{cx}\" y=\"46\" text-anchor=\"middle\" font-size=\"12\" fill=\"{SUBTITLE}\">{s}</text>",
    cx = width / 2,
    s = escape_xml(subtitle),
  ));

  // Gridlines + x-axis tick labels.
  let max_tick = max_x.ceil() as u32;
  for tick in 0..=max_tick {
    let x = plot_x_start + (f64::from(tick) / max_x) * plot_w;
    let (stroke, dash) = if tick == 1 {
      (PARITY, " stroke-dasharray=\"4,4\"")
    } else {
      (GRID, "")
    };
    svg.push_str(&format!(
      "<line x1=\"{x:.1}\" y1=\"{y1:.1}\" x2=\"{x:.1}\" y2=\"{y2:.1}\" stroke=\"{stroke}\" stroke-width=\"1\"{dash}/>",
      y1 = plot_y_start,
      y2 = plot_y_end,
    ));
    svg.push_str(&format!(
      "<text x=\"{x:.1}\" y=\"338\" text-anchor=\"middle\" font-size=\"10\" fill=\"{AXIS}\">{tick}x</text>",
    ));
  }

  // Bars.
  for (i, (label, value)) in bars.iter().enumerate() {
    let by = plot_y_start + top_pad + (i as f64) * (bar_height + bar_gap);
    let bw = (value / max_x) * plot_w;
    let text_y = by + bar_height / 2.0 + 4.0;

    // Category label (right-aligned, just left of the plot area).
    svg.push_str(&format!(
      "<text x=\"{x:.1}\" y=\"{y:.1}\" text-anchor=\"end\" font-size=\"12\" fill=\"{LABEL}\">{label}</text>",
      x = plot_x_start - 12.0,
      y = text_y,
      label = escape_xml(label),
    ));

    // Bar.
    svg.push_str(&format!(
      "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" fill=\"{BAR}\" rx=\"2\"/>",
      x = plot_x_start,
      y = by,
      w = bw,
      h = bar_height,
    ));

    // Value label past the right end of each bar (sits on dark bg).
    svg.push_str(&format!(
      "<text x=\"{x:.1}\" y=\"{y:.1}\" font-size=\"11\" font-weight=\"600\" fill=\"{BAR_LABEL}\">{v:.2}x</text>",
      x = plot_x_start + bw + 6.0,
      y = text_y,
      v = value,
    ));
  }

  // Source caption.
  svg.push_str(&format!(
    "<text x=\"{cx}\" y=\"362\" text-anchor=\"middle\" font-size=\"9\" fill=\"{CAPTION}\">Source: \
     benchmark_results/OVERVIEW.md</text>",
    cx = width / 2,
  ));

  svg.push_str("</svg>");
  svg
}
