#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Coverage Reporting for rscrypto
#
# Generates Rust source coverage from two deterministic sources:
#   1. Unit/integration/property tests via cargo-nextest.
#   2. Committed fuzz corpora via corpus replay integration tests.
#
# Usage:
#   ./scripts/test/test-coverage.sh                # Total LCOV: nextest + fuzz replay
#   ./scripts/test/test-coverage.sh --nextest      # Test suite LCOV only
#   ./scripts/test/test-coverage.sh --fuzz         # Fuzz corpus replay LCOV only
#   ./scripts/test/test-coverage.sh --html         # Total LCOV + HTML report
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

maybe_disable_sccache

COV_DIR="$REPO_ROOT/coverage"
FUZZ_COVERAGE_SCOPE=${RSCRYPTO_FUZZ_COVERAGE_SCOPE:-all}
LLVM_COV_TARGET_DIR="${RSCRYPTO_COV_TARGET_DIR:-$REPO_ROOT/target/llvm-cov-target}"

RUN_NEXTEST=false
RUN_FUZZ=false
GEN_HTML=false

case "${1:-}" in
  --nextest)  RUN_NEXTEST=true ;;
  --fuzz)     RUN_FUZZ=true ;;
  --html)     RUN_NEXTEST=true; RUN_FUZZ=true; GEN_HTML=true ;;
  -h|--help)
    echo "Usage: $0 [--nextest|--fuzz|--html]"
    echo "  (no args)    Both nextest + fuzz coverage"
    echo "  --nextest    Test suite coverage only"
    echo "  --fuzz       Fuzz corpus coverage only"
    echo "  --html       Both + HTML report"
    exit 0
    ;;
  *)          RUN_NEXTEST=true; RUN_FUZZ=true ;;
esac

mkdir -p "$COV_DIR"

clean_coverage_outputs() {
  rm -f "$COV_DIR"/nextest.lcov "$COV_DIR"/fuzz.lcov "$COV_DIR"/total.lcov "$COV_DIR"/merged.lcov
  rm -f "$COV_DIR"/fuzz-*.lcov "$COV_DIR"/fuzz-profile-manifest.txt "$COV_DIR"/SUMMARY.md
  rm -rf "$COV_DIR"/html-nextest "$COV_DIR"/html-fuzz "$COV_DIR"/html-total
  rm -rf "$LLVM_COV_TARGET_DIR"
}

clean_coverage_outputs

# ── Coverage capture ─────────────────────────────────────────────────────────

require_coverage_tools() {
  if ! command -v cargo-llvm-cov &>/dev/null; then
    echo "Error: cargo-llvm-cov not found"
    echo "Install with: cargo install cargo-llvm-cov"
    exit 1
  fi

  if [ "$RUN_NEXTEST" = true ] && ! cargo nextest --version &>/dev/null; then
    echo "Error: cargo-nextest not found"
    echo "Install with: cargo install cargo-nextest"
    exit 1
  fi
}

start_coverage_capture() {
  export CARGO_INCREMENTAL=0
  export CARGO_TARGET_DIR="$LLVM_COV_TARGET_DIR"

  cargo llvm-cov clean --workspace

  # shellcheck disable=SC2046
  eval "$(cargo llvm-cov show-env --sh)"
}

run_nextest_capture() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Test Suite Coverage Capture (cargo-nextest)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  cargo nextest run --all-features --workspace
}

run_fuzz_replay_capture() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Fuzz Corpus Coverage Capture (deterministic replay)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if [ ! -f "$REPO_ROOT/tests/fuzz_corpus_replay.rs" ]; then
    echo "Error: tests/fuzz_corpus_replay.rs not found"
    return 1
  fi

  local test_filter=()
  case "$FUZZ_COVERAGE_SCOPE" in
    all) ;;
    full) test_filter=("replay_full_") ;;
    scoped) test_filter=("replay_scoped_") ;;
    *)
      echo "Error: unknown fuzz coverage scope: $FUZZ_COVERAGE_SCOPE"
      return 1
      ;;
  esac

  echo "Fuzz coverage scope: $FUZZ_COVERAGE_SCOPE"
  cargo test --all-features --test fuzz_corpus_replay "${test_filter[@]}" -- --ignored --nocapture

  echo ""
  echo "Fuzz corpus replay captured"
}

coverage_output_path() {
  if [ "$RUN_NEXTEST" = true ] && [ "$RUN_FUZZ" = true ]; then
    echo "$COV_DIR/total.lcov"
  elif [ "$RUN_NEXTEST" = true ]; then
    echo "$COV_DIR/nextest.lcov"
  else
    echo "$COV_DIR/fuzz.lcov"
  fi
}

coverage_html_dir() {
  if [ "$RUN_NEXTEST" = true ] && [ "$RUN_FUZZ" = true ]; then
    echo "$COV_DIR/html-total"
  elif [ "$RUN_NEXTEST" = true ]; then
    echo "$COV_DIR/html-nextest"
  else
    echo "$COV_DIR/html-fuzz"
  fi
}

generate_report() {
  local lcov_path=$1
  local html_dir

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Generating coverage report"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  cargo llvm-cov report --lcov --output-path "$lcov_path"
  echo "LCOV report: $lcov_path"

  if [ "$GEN_HTML" = true ]; then
    html_dir="$(coverage_html_dir)"
    cargo llvm-cov report --html --output-dir "$html_dir"
    echo "HTML report: $html_dir/index.html"
  fi
}

generate_summary() {
  local lcov_path=$1

  local lines_found lines_hit fn_found fn_hit
  lines_found=$(awk -F: '/^LF:/{s+=$2} END{print s+0}' "$lcov_path")
  lines_hit=$(awk -F: '/^LH:/{s+=$2} END{print s+0}' "$lcov_path")
  fn_found=$(awk -F: '/^FNF:/{s+=$2} END{print s+0}' "$lcov_path")
  fn_hit=$(awk -F: '/^FNH:/{s+=$2} END{print s+0}' "$lcov_path")

  local line_pct="0.0" fn_pct="0.0"
  if [ "$lines_found" -gt 0 ]; then
    line_pct=$(awk "BEGIN{printf \"%.1f\", 100*$lines_hit/$lines_found}")
  fi
  if [ "$fn_found" -gt 0 ]; then
    fn_pct=$(awk "BEGIN{printf \"%.1f\", 100*$fn_hit/$fn_found}")
  fi

  local sources_list=""
  if [ "$RUN_NEXTEST" = true ]; then
    sources_list="${sources_list}
- \`nextest\`"
  fi
  if [ "$RUN_FUZZ" = true ]; then
    sources_list="${sources_list}
- \`fuzz corpus replay (${FUZZ_COVERAGE_SCOPE})\`"
  fi

  echo ""
  echo "  Lines:     ${lines_hit}/${lines_found} (${line_pct}%)"
  echo "  Functions: ${fn_hit}/${fn_found} (${fn_pct}%)"
  echo "  Report:    $lcov_path"

  cat > "$COV_DIR/SUMMARY.md" <<SUMMARY_EOF
# Coverage Summary

| Metric    | Hit | Total | Coverage |
|-----------|-----|-------|----------|
| Lines     | ${lines_hit} | ${lines_found} | ${line_pct}% |
| Functions | ${fn_hit} | ${fn_found} | ${fn_pct}% |

Generated: $(date -u +"%Y-%m-%d %H:%M UTC")

## Sources
${sources_list}
SUMMARY_EOF

  echo "  Summary:   $COV_DIR/SUMMARY.md"
}

# ── Main ──────────────────────────────────────────────────────────────────────

require_coverage_tools
start_coverage_capture

if [ "$RUN_NEXTEST" = true ]; then
  run_nextest_capture
fi

if [ "$RUN_FUZZ" = true ]; then
  run_fuzz_replay_capture
fi

LCOV_PATH="$(coverage_output_path)"
generate_report "$LCOV_PATH"
generate_summary "$LCOV_PATH"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Coverage reports in: $COV_DIR/"
if find "$COV_DIR" -maxdepth 1 \( -name "*.lcov" -o -name "SUMMARY.md" \) -type f -print -quit | grep -q .; then
  find "$COV_DIR" -maxdepth 1 \( -name "*.lcov" -o -name "SUMMARY.md" \) -type f -exec ls -la {} +
else
  echo "(no coverage files generated)"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
