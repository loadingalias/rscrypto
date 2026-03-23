#!/usr/bin/env bash
set -euo pipefail

# extract-results.sh — Download CI bench artifacts and produce a W/T/L report.
#
# Usage:
#   scripts/bench/extract-results.sh <run-id> [--filter <pattern>]
#
# Examples:
#   scripts/bench/extract-results.sh 23415302173
#   scripts/bench/extract-results.sh 23415302173 --filter crc
#
# Requires: gh (GitHub CLI)

usage() {
  echo "Usage: $0 <run-id> [--filter <pattern>]" >&2
  exit 1
}

RUN_ID="${1:-}"
shift || true
FILTER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --filter) FILTER="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -z "$RUN_ID" ]] && usage

REPO="loadingalias/rscrypto"
WORK="/tmp/bench-extract-$$"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

# ── 1. Download artifacts ─────────────────────────────────────────────────────

echo "Fetching artifact list for run $RUN_ID..."
gh api "repos/$REPO/actions/runs/$RUN_ID/artifacts" --jq \
  '.artifacts[] | "\(.id) \(.name)"' | awk '!seen[$2]++' > "$WORK/artifacts.txt"

ARTIFACT_COUNT=$(wc -l < "$WORK/artifacts.txt" | tr -d ' ')
echo "Found $ARTIFACT_COUNT artifacts."

while read -r id name; do
  platform="${name#benchmark-}"
  echo "  Downloading $platform..."
  gh api "repos/$REPO/actions/artifacts/$id/zip" > "$WORK/${platform}.zip"
  mkdir -p "$WORK/$platform"
  unzip -o "$WORK/${platform}.zip" -d "$WORK/$platform" > /dev/null
done < "$WORK/artifacts.txt"

# ── 2. Parse Criterion stdout ─────────────────────────────────────────────────
#
# Extract lines matching: group/impl/size  time:   [lo mid hi unit]
# Produce TSV: platform  group  impl  size  mid_ns

parse_output() {
  local platform="$1"
  local file="$2"

  # Criterion output has two formats:
  #   1. bench_id        time:   [lo mid hi unit]     (single line)
  #   2. bench_id\n                        time:   [lo mid hi unit]  (split across lines)
  # We use awk to handle both: track the last bench_id seen, then parse time lines.
  awk -v platform="$platform" -v filter="$FILTER" '
  /^[A-Za-z0-9_-]+\/[A-Za-z0-9_-]+\/[0-9]+/ {
    # Extract bench_id (first non-whitespace token)
    bench_id = $1
    # If this line also has time:, fall through to the time block
  }
  /time:[ \t]+\[/ && bench_id != "" {
    # Parse: time:   [lo lo_unit mid mid_unit hi hi_unit]
    line = $0
    # Extract content between [ and ]
    start = index(line, "[")
    end = index(line, "]")
    if (start > 0 && end > start) {
      inner = substr(line, start + 1, end - start - 1)
      m = split(inner, vals, " ")
      # vals: lo lo_unit mid mid_unit hi hi_unit
      if (m >= 4) {
        mid = vals[3] + 0.0
        unit = vals[4]
        # Convert to nanoseconds
        if (unit == "ps") mid_ns = mid / 1000.0
        else if (unit == "ns") mid_ns = mid
        else if (unit == "µs" || unit == "us") mid_ns = mid * 1000.0
        else if (unit == "ms") mid_ns = mid * 1000000.0
        else if (unit == "s") mid_ns = mid * 1000000000.0
        else { bench_id = ""; next }

        # Split bench_id: group/impl/size
        n2 = split(bench_id, bid, "/")
        if (n2 >= 3) {
          group = bid[1]
          impl = bid[2]
          size = bid[3]
          if (filter != "" && index(group, filter) == 0) {
            bench_id = ""
            next
          }
          printf "%s\t%s\t%s\t%s\t%.4f\n", platform, group, impl, size, mid_ns
        }
      }
    }
    bench_id = ""
  }
  ' "$file"
}

echo ""
echo "Parsing results..."
RESULTS="$WORK/results.tsv"
: > "$RESULTS"

for dir in "$WORK"/*/; do
  [[ -f "$dir/output.txt" ]] || continue
  platform=$(basename "$dir")
  parse_output "$platform" "$dir/output.txt" >> "$RESULTS"
done

TOTAL_LINES=$(wc -l < "$RESULTS" | tr -d ' ')
echo "Parsed $TOTAL_LINES data points."

# ── 3. Compute W/T/L ─────────────────────────────────────────────────────────

echo ""
echo "================================================================="
echo "  CRC Benchmark Results -- Run $RUN_ID"
echo "================================================================="

awk -F'\t' '
BEGIN {
  TIE_MARGIN = 0.03
  wins = 0; ties = 0; losses = 0
}

{
  platform = $1; group = $2; impl = $3; size = $4; ns = $5 + 0.0
  key = platform SUBSEP group SUBSEP size
  data[key, impl] = ns
  # Track which impls exist per key
  n = ++impl_count[key]
  impl_list[key, n] = impl
  # Track unique keys
  if (!(key in seen_keys)) {
    seen_keys[key] = 1
    nkeys++
    all_keys[nkeys] = key
    key_platform[key] = platform
    key_group[key] = group
    key_size[key] = size + 0
  }
}

END {
  # Process each key
  for (ki = 1; ki <= nkeys; ki++) {
    key = all_keys[ki]
    platform = key_platform[key]
    group = key_group[key]
    size = key_size[key]

    # Find rscrypto time
    rscrypto_ns = data[key, "rscrypto"] + 0.0
    if (rscrypto_ns == 0) continue

    # Compare against each competitor
    ni = impl_count[key]
    for (ii = 1; ii <= ni; ii++) {
      impl = impl_list[key, ii]
      if (impl == "rscrypto") continue
      comp_ns = data[key, impl] + 0.0
      if (comp_ns == 0) continue

      ratio = rscrypto_ns / comp_ns
      if (ratio <= 1.0 + TIE_MARGIN && ratio >= 1.0 - TIE_MARGIN) {
        result = "TIE"
        ties++
      } else if (rscrypto_ns < comp_ns) {
        result = "WIN"
        wins++
      } else {
        result = "LOSS"
        losses++
      }

      # Track per-size bucket
      bucket = (size <= 64) ? "small" : "large"
      if (result == "WIN") bucket_wins[bucket]++
      if (result == "TIE") bucket_ties[bucket]++
      if (result == "LOSS") bucket_losses[bucket]++

      # Track per-group
      if (result == "WIN") group_wins[group]++
      if (result == "TIE") group_ties[group]++
      if (result == "LOSS") group_losses[group]++

      # Store losses for detail report
      if (result == "LOSS") {
        nloss++
        loss_platform[nloss] = platform
        loss_group[nloss] = group
        loss_size[nloss] = size
        loss_rscrypto[nloss] = rscrypto_ns
        loss_impl[nloss] = impl
        loss_comp[nloss] = comp_ns
        loss_ratio[nloss] = ratio
      }
    }
  }

  total = wins + ties + losses
  win_pct = (total > 0) ? (wins * 100.0 / total) : 0

  printf "\n"
  printf "=================================================================\n"
  printf "  OVERALL: %d W / %d T / %d L  (%.0f%% win rate, %d comparisons)\n", \
    wins, ties, losses, win_pct, total
  printf "=================================================================\n"

  # Per-group summary
  printf "\n  Per-group breakdown:\n"
  for (g in group_wins) {
    gt = group_wins[g] + group_ties[g] + group_losses[g]
    gp = (gt > 0) ? (group_wins[g] * 100.0 / gt) : 0
    printf "    %-20s %3dW / %2dT / %2dL  (%5.1f%%)\n", \
      g, group_wins[g] + 0, group_ties[g] + 0, group_losses[g] + 0, gp
  }
  # Print groups that only have ties/losses (no wins entry)
  for (g in group_ties) {
    if (!(g in group_wins)) {
      gt = 0 + group_ties[g] + group_losses[g]
      printf "    %-20s %3dW / %2dT / %2dL  (%5.1f%%)\n", \
        g, 0, group_ties[g] + 0, group_losses[g] + 0, 0.0
    }
  }
  for (g in group_losses) {
    if (!(g in group_wins) && !(g in group_ties)) {
      gt = 0 + group_losses[g]
      printf "    %-20s %3dW / %2dT / %2dL  (%5.1f%%)\n", \
        g, 0, 0, group_losses[g] + 0, 0.0
    }
  }

  # Per-size-bucket summary
  printf "\n  Fast-path impact (0-64 B vs 256+ B):\n"
  for (b in bucket_wins) {
    bt = bucket_wins[b] + bucket_ties[b] + bucket_losses[b]
    bp = (bt > 0) ? (bucket_wins[b] * 100.0 / bt) : 0
    label = (b == "small") ? "0-64 B (fast-path)" : "256+ B (dispatch)"
    printf "    %-25s %3dW / %2dT / %2dL  (%5.1f%%)\n", \
      label, bucket_wins[b] + 0, bucket_ties[b] + 0, bucket_losses[b] + 0, bp
  }

  # All losses
  printf "\n  ALL LOSSES (rscrypto slower, sorted by severity):\n"
  printf "  %-14s %-16s %7s  %10s  %-12s %10s  %s\n", \
    "PLATFORM", "GROUP", "SIZE", "OURS (ns)", "COMPETITOR", "THEIRS", "RATIO"
  printf "  %-14s %-16s %7s  %10s  %-12s %10s  %s\n", \
    "--------------", "----------------", "-------", "----------", "------------", "----------", "-------"
  for (i = 1; i <= nloss; i++) {
    printf "  %-14s %-16s %7d  %10.2f  %-12s %10.2f  %.2fx\n", \
      loss_platform[i], loss_group[i], loss_size[i], \
      loss_rscrypto[i], loss_impl[i], loss_comp[i], loss_ratio[i]
  }
  printf "\n"
}
' "$RESULTS"
