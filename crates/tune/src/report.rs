//! Output formatters for tuning results.
//!
//! This module provides multiple output formats for tuning results:
//!
//! - [`OutputFormat::Summary`]: Human-readable summary (default)
//! - [`OutputFormat::Env`]: Shell environment variable exports
//! - [`OutputFormat::Json`]: JSON for programmatic use
//! - [`OutputFormat::Tsv`]: Tab-separated values for spreadsheets
//! - [`OutputFormat::Contribute`]: Markdown for GitHub issue submission
//!
//! Use the [`Report`] struct for custom output destinations, or the
//! convenience functions like [`print_summary`] for stdout output.

use std::io::{self, Write};

use crate::{AlgorithmResult, TuneResults, targets};

/// Output format for tuning results.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OutputFormat {
  /// Human-readable summary (default).
  #[default]
  Summary,

  /// Shell environment variable exports.
  Env,

  /// JSON format for programmatic use.
  Json,

  /// Tab-separated values.
  Tsv,

  /// Markdown formatted for GitHub issue submission.
  Contribute,
}

impl OutputFormat {
  /// Parse format from string.
  #[must_use]
  pub fn parse(s: &str) -> Option<Self> {
    match s.to_lowercase().as_str() {
      "summary" | "text" | "human" => Some(Self::Summary),
      "env" | "shell" | "export" => Some(Self::Env),
      "json" => Some(Self::Json),
      "tsv" | "tab" => Some(Self::Tsv),
      "contribute" | "pr" | "issue" | "markdown" | "md" => Some(Self::Contribute),
      _ => None,
    }
  }
}

/// Report generator for tuning results.
pub struct Report<W: Write> {
  writer: W,
  format: OutputFormat,
}

impl<W: Write> Report<W> {
  /// Create a new report generator.
  pub fn new(writer: W, format: OutputFormat) -> Self {
    Self { writer, format }
  }

  /// Write the complete tuning results.
  pub fn write(&mut self, results: &TuneResults) -> io::Result<()> {
    match self.format {
      OutputFormat::Summary => self.write_summary(results),
      OutputFormat::Env => self.write_env(results),
      OutputFormat::Json => self.write_json(results),
      OutputFormat::Tsv => self.write_tsv(results),
      OutputFormat::Contribute => self.write_contribute(results),
    }
  }

  /// Write human-readable summary.
  fn write_summary(&mut self, results: &TuneResults) -> io::Result<()> {
    writeln!(self.writer, "rscrypto Tuning Results")?;
    writeln!(self.writer, "========================")?;
    writeln!(self.writer)?;

    // Platform info
    writeln!(self.writer, "Platform: {}", results.platform.description)?;
    writeln!(self.writer, "Tune preset: {:?}", results.platform.tune_kind)?;
    writeln!(self.writer, "Timestamp: {}", results.timestamp)?;
    writeln!(self.writer)?;

    // Algorithm results
    for algo in &results.algorithms {
      self.write_algorithm_summary(algo, results.platform.arch, results.platform.tune_kind)?;
      writeln!(self.writer)?;
    }

    Ok(())
  }

  /// Write summary for a single algorithm.
  fn write_algorithm_summary(
    &mut self,
    algo: &AlgorithmResult,
    arch: &str,
    tune_kind: platform::TuneKind,
  ) -> io::Result<()> {
    writeln!(self.writer, "=== {} ===", algo.name)?;
    writeln!(self.writer, "Best kernel: {}", algo.best_kernel)?;
    writeln!(self.writer, "Recommended streams: {}", algo.recommended_streams)?;
    writeln!(self.writer, "Peak throughput: {:.2} GiB/s", algo.peak_throughput_gib_s)?;

    let mut wrote_perf_targets = false;
    for class_best in &algo.size_class_best {
      let Some(target_gib_s) = targets::class_target_gib_s(algo.name, arch, tune_kind, class_best.class) else {
        continue;
      };
      if !wrote_perf_targets {
        writeln!(self.writer, "Perf targets ({arch}):")?;
        wrote_perf_targets = true;
      }
      let status = if class_best.throughput_gib_s >= target_gib_s {
        "ok"
      } else {
        "MISS"
      };
      writeln!(
        self.writer,
        "  {}: {:.2} GiB/s (target >= {:.2}) [{status}]",
        class_best.class, class_best.throughput_gib_s, target_gib_s
      )?;
    }

    if !algo.thresholds.is_empty() {
      writeln!(self.writer, "Recommended thresholds:")?;
      for (name, value) in &algo.thresholds {
        if *value == usize::MAX {
          writeln!(self.writer, "  {name}: usize::MAX")?;
        } else {
          writeln!(self.writer, "  {name}: {value}")?;
        }
      }
    }

    // Crossovers
    if !algo.analysis.crossovers.is_empty() {
      writeln!(self.writer, "Crossovers:")?;
      for crossover in &algo.analysis.crossovers {
        writeln!(
          self.writer,
          "  {} -> {} at {} bytes (margin: {:.1}%)",
          crossover.from_kernel, crossover.to_kernel, crossover.crossover_size, crossover.margin_percent
        )?;
      }
    }

    Ok(())
  }

  /// Write shell environment variable exports.
  fn write_env(&mut self, results: &TuneResults) -> io::Result<()> {
    writeln!(
      self.writer,
      "# rscrypto tuning results - generated {}",
      results.timestamp
    )?;
    writeln!(self.writer, "# Platform: {}", results.platform.description)?;
    writeln!(self.writer)?;

    for algo in &results.algorithms {
      self.write_algorithm_env(algo)?;
      writeln!(self.writer)?;
    }

    Ok(())
  }

  /// Write env exports for a single algorithm.
  fn write_algorithm_env(&mut self, algo: &AlgorithmResult) -> io::Result<()> {
    writeln!(self.writer, "# {}", algo.name)?;

    // Use the stored env prefix (e.g., "RSCRYPTO_CRC64")
    let prefix = algo.env_prefix;

    // Stream count
    writeln!(self.writer, "export {prefix}_STREAMS={}", algo.recommended_streams)?;

    // Thresholds - the suffix already includes proper naming (e.g., "THRESHOLD_PORTABLE_TO_CLMUL")
    for (env_suffix, value) in &algo.thresholds {
      if *value == usize::MAX {
        writeln!(self.writer, "export {prefix}_{env_suffix}=usize::MAX")?;
      } else {
        writeln!(self.writer, "export {prefix}_{env_suffix}={value}")?;
      }
    }

    Ok(())
  }

  /// Write JSON format.
  fn write_json(&mut self, results: &TuneResults) -> io::Result<()> {
    // Simple JSON serialization without serde
    writeln!(self.writer, "{{")?;
    writeln!(self.writer, "  \"timestamp\": \"{}\",", results.timestamp)?;
    writeln!(self.writer, "  \"platform\": {{")?;
    writeln!(self.writer, "    \"arch\": \"{}\",", results.platform.arch)?;
    writeln!(self.writer, "    \"os\": \"{}\",", results.platform.os)?;
    writeln!(
      self.writer,
      "    \"description\": \"{}\",",
      escape_json(&results.platform.description)
    )?;
    writeln!(self.writer, "    \"tune_kind\": \"{:?}\"", results.platform.tune_kind)?;
    writeln!(self.writer, "  }},")?;

    writeln!(self.writer, "  \"algorithms\": [")?;
    for (i, algo) in results.algorithms.iter().enumerate() {
      let comma = if i < results.algorithms.len().strict_sub(1) {
        ","
      } else {
        ""
      };
      self.write_algorithm_json(algo, comma)?;
    }
    writeln!(self.writer, "  ]")?;
    writeln!(self.writer, "}}")?;

    Ok(())
  }

  /// Write JSON for a single algorithm.
  fn write_algorithm_json(&mut self, algo: &AlgorithmResult, trailing_comma: &str) -> io::Result<()> {
    writeln!(self.writer, "    {{")?;
    writeln!(self.writer, "      \"name\": \"{}\",", algo.name)?;
    writeln!(self.writer, "      \"env_prefix\": \"{}\",", algo.env_prefix)?;
    writeln!(self.writer, "      \"best_kernel\": \"{}\",", algo.best_kernel)?;
    writeln!(
      self.writer,
      "      \"recommended_streams\": {},",
      algo.recommended_streams
    )?;
    writeln!(
      self.writer,
      "      \"peak_throughput_gib_s\": {:.6},",
      algo.peak_throughput_gib_s
    )?;

    writeln!(self.writer, "      \"thresholds\": {{")?;
    for (i, (env_suffix, value)) in algo.thresholds.iter().enumerate() {
      let comma = if i < algo.thresholds.len().strict_sub(1) {
        ","
      } else {
        ""
      };
      // Use the full env var name in JSON for clarity
      let env_var = format!("{}_{}", algo.env_prefix, env_suffix);
      if *value == usize::MAX {
        writeln!(self.writer, "        \"{env_var}\": null{comma}")?;
      } else {
        writeln!(self.writer, "        \"{env_var}\": {value}{comma}")?;
      }
    }
    writeln!(self.writer, "      }}")?;

    writeln!(self.writer, "    }}{trailing_comma}")?;

    Ok(())
  }

  /// Write TSV format.
  fn write_tsv(&mut self, results: &TuneResults) -> io::Result<()> {
    // Header
    writeln!(self.writer, "algorithm\tbest_kernel\tstreams\tpeak_gib_s\tthresholds")?;

    for algo in &results.algorithms {
      let thresholds: Vec<String> = algo
        .thresholds
        .iter()
        .map(|(k, v)| {
          if *v == usize::MAX {
            format!("{k}=MAX")
          } else {
            format!("{k}={v}")
          }
        })
        .collect();

      writeln!(
        self.writer,
        "{}\t{}\t{}\t{:.6}\t{}",
        algo.name,
        algo.best_kernel,
        algo.recommended_streams,
        algo.peak_throughput_gib_s,
        thresholds.join(",")
      )?;
    }

    Ok(())
  }

  /// Write markdown format for GitHub issue contribution.
  fn write_contribute(&mut self, results: &TuneResults) -> io::Result<()> {
    writeln!(self.writer)?;
    writeln!(self.writer, "## Tuning Results")?;
    writeln!(self.writer)?;
    writeln!(self.writer, "**Platform:** `{}`", results.platform.description)?;
    writeln!(self.writer, "**Tune preset:** `{:?}`", results.platform.tune_kind)?;
    writeln!(self.writer, "**Timestamp:** {}", results.timestamp)?;
    writeln!(self.writer)?;

    // Compact table
    writeln!(self.writer, "| Algorithm | Best Kernel | Streams | Peak GiB/s |")?;
    writeln!(self.writer, "|-----------|-------------|---------|------------|")?;
    for algo in &results.algorithms {
      writeln!(
        self.writer,
        "| {} | `{}` | {} | {:.1} |",
        algo.name, algo.best_kernel, algo.recommended_streams, algo.peak_throughput_gib_s
      )?;
    }
    writeln!(self.writer)?;

    // Thresholds in collapsible section
    writeln!(self.writer, "<details>")?;
    writeln!(self.writer, "<summary>Detailed thresholds (click to expand)</summary>")?;
    writeln!(self.writer)?;
    writeln!(self.writer, "```")?;
    for algo in &results.algorithms {
      writeln!(self.writer, "# {}", algo.name)?;
      writeln!(self.writer, "{}_STREAMS={}", algo.env_prefix, algo.recommended_streams)?;
      for (suffix, value) in &algo.thresholds {
        if *value == usize::MAX {
          writeln!(self.writer, "{}=usize::MAX", suffix)?;
        } else {
          writeln!(self.writer, "{}={}", suffix, value)?;
        }
      }
      writeln!(self.writer)?;
    }
    writeln!(self.writer, "```")?;
    writeln!(self.writer)?;
    writeln!(self.writer, "</details>")?;
    writeln!(self.writer)?;

    // Instructions
    writeln!(self.writer, "---")?;
    writeln!(self.writer)?;
    writeln!(self.writer, "Copy everything above this line into a GitHub issue at:")?;
    writeln!(
      self.writer,
      "https://github.com/loadingalias/rscrypto/issues/new?template=tuning-results.md"
    )?;

    Ok(())
  }
}

/// Escape a string for JSON output.
fn escape_json(s: &str) -> String {
  s.replace('\\', "\\\\")
    .replace('"', "\\\"")
    .replace('\n', "\\n")
    .replace('\r', "\\r")
    .replace('\t', "\\t")
}

/// Write results to stdout in the specified format.
fn print_with_format(results: &TuneResults, format: OutputFormat) -> io::Result<()> {
  let stdout = io::stdout();
  Report::new(stdout.lock(), format).write(results)
}

/// Print results to stdout as human-readable summary.
pub fn print_summary(results: &TuneResults) -> io::Result<()> {
  print_with_format(results, OutputFormat::Summary)
}

/// Print results to stdout as shell environment variable exports.
pub fn print_env(results: &TuneResults) -> io::Result<()> {
  print_with_format(results, OutputFormat::Env)
}

/// Print results to stdout as JSON.
pub fn print_json(results: &TuneResults) -> io::Result<()> {
  print_with_format(results, OutputFormat::Json)
}

/// Print results to stdout as tab-separated values.
pub fn print_tsv(results: &TuneResults) -> io::Result<()> {
  print_with_format(results, OutputFormat::Tsv)
}

/// Print results to stdout as contribution-ready markdown.
pub fn print_contribute(results: &TuneResults) -> io::Result<()> {
  print_with_format(results, OutputFormat::Contribute)
}
