#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Coverage Reporting for rscrypto
#
# Generates LCOV coverage reports from two sources:
#   1. Test suite via cargo-llvm-cov + nextest  → coverage/nextest.lcov
#   2. Fuzz corpus via cargo fuzz coverage      → coverage/fuzz.lcov
#
# Usage:
#   ./scripts/test/test-coverage.sh                # Both nextest + fuzz
#   ./scripts/test/test-coverage.sh --nextest      # Test suite only
#   ./scripts/test/test-coverage.sh --fuzz         # Fuzz corpus only
#   ./scripts/test/test-coverage.sh --html         # Both + HTML report
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/fuzz-packages.sh
source "$SCRIPT_DIR/../lib/fuzz-packages.sh"

maybe_disable_sccache

COV_DIR="$REPO_ROOT/coverage"
FUZZ_CORPUS_SECS=${RSCRYPTO_COV_FUZZ_SECS:-30}
FUZZ_COVERAGE_SCOPE=${RSCRYPTO_FUZZ_COVERAGE_SCOPE:-full}

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

# ── Nextest coverage ─────────────────────────────────────────────────────────

run_nextest_coverage() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Test Suite Coverage (cargo-llvm-cov + nextest)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if ! command -v cargo-llvm-cov &>/dev/null; then
    echo "Error: cargo-llvm-cov not found"
    echo "Install with: cargo install cargo-llvm-cov"
    exit 1
  fi

  local lcov_path="$COV_DIR/nextest.lcov"

  cargo llvm-cov nextest \
    --all-features \
    --workspace \
    --lcov \
    --output-path "$lcov_path"

  echo ""
  echo "Nextest coverage: $lcov_path"

  if [ "$GEN_HTML" = true ]; then
    cargo llvm-cov nextest \
      --all-features \
      --workspace \
      --html \
      --output-dir "$COV_DIR/html-nextest"
    echo "HTML report: $COV_DIR/html-nextest/index.html"
  fi
}

# ── Fuzz corpus coverage ─────────────────────────────────────────────────────

run_fuzz_coverage() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Fuzz Corpus Coverage"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if ! command -v cargo-fuzz &>/dev/null; then
    echo "Error: cargo-fuzz not found"
    exit 1
  fi

  if [ ! -d "$FUZZ_ROOT" ]; then
    echo "No fuzz/ directory found — skipping"
    return 0
  fi

  discover_fuzz_packages
  fuzz_select_packages "$FUZZ_COVERAGE_SCOPE"
  if [ ${#SELECTED_FUZZ_PACKAGES[@]} -eq 0 ]; then
    echo "No fuzz packages selected for coverage — skipping"
    return 0
  fi

  local host_target
  host_target="$(rustc -vV | sed -n 's|host: ||p')"

  # Locate llvm-profdata and llvm-cov from the toolchain.
  local llvm_profdata llvm_cov
  local toolchain_lib
  toolchain_lib="$(rustc --print sysroot)/lib/rustlib/$host_target/bin"
  if [ -x "$toolchain_lib/llvm-profdata" ]; then
    llvm_profdata="$toolchain_lib/llvm-profdata"
    llvm_cov="$toolchain_lib/llvm-cov"
  elif command -v llvm-profdata &>/dev/null; then
    llvm_profdata="llvm-profdata"
    llvm_cov="llvm-cov"
  else
    echo "Error: llvm-profdata not found"
    echo "Install with: rustup component add llvm-tools-preview"
    return 1
  fi

  local all_lcov_files=()
  local package_dir package_label package_slug target

  echo ""
  echo "Phase 1: Seeding corpus (${FUZZ_CORPUS_SECS}s per target without corpus)"
  for package_dir in "${SELECTED_FUZZ_PACKAGES[@]}"; do
    package_label="$(fuzz_package_label "$package_dir")"
    while IFS= read -r target; do
      [ -z "$target" ] && continue
      local corpus_dir="$package_dir/corpus/$target"
      if [ -d "$corpus_dir" ] && [ "$(ls -A "$corpus_dir" 2>/dev/null)" ]; then
        echo "  ${package_label}/${target}: corpus exists ($(ls "$corpus_dir" | wc -l | tr -d ' ') entries)"
      else
        echo "  ${package_label}/${target}: building corpus (${FUZZ_CORPUS_SECS}s)..."
        mkdir -p "$corpus_dir"
        local fuzz_exit=0
        fuzz_in_package "$package_dir" run "$target" \
          --target "$host_target" \
          -- -max_total_time="$FUZZ_CORPUS_SECS" -max_len=65536 \
          2>&1 | { grep -v "^INFO:" || true; } || fuzz_exit=$?
        if [ "$fuzz_exit" -ne 0 ]; then
          echo "    warning: corpus seeding exited $fuzz_exit (crash?), continuing"
        fi
      fi
    done < <(fuzz_list_targets "$package_dir")
  done

  echo ""
  echo "Phase 2: Generating coverage reports"
  for package_dir in "${SELECTED_FUZZ_PACKAGES[@]}"; do
    package_label="$(fuzz_package_label "$package_dir")"
    package_slug="${package_label//\//-}"
    while IFS= read -r target; do
      [ -z "$target" ] && continue
      echo "  ${package_label}/${target}: generating coverage..."
      local cov_exit=0
      fuzz_in_package "$package_dir" coverage "$target" --target "$host_target" \
        2>&1 | { grep -v "^INFO:" || true; } || cov_exit=$?
      if [ "$cov_exit" -ne 0 ]; then
        echo "    warning: coverage generation exited $cov_exit, skipping"
        continue
      fi

      local cov_dir="$package_dir/coverage/$target"
      if [ ! -d "$cov_dir" ]; then
        echo "    warning: no coverage output for ${package_label}/${target}"
        continue
      fi

      local profraw_files
      profraw_files=$(find "$cov_dir" -name "*.profraw" 2>/dev/null)
      if [ -z "$profraw_files" ]; then
        echo "    warning: no profraw files for ${package_label}/${target}"
        continue
      fi

      "$llvm_profdata" merge -sparse $profraw_files -o "$cov_dir/merged.profdata"

      local binary
      binary=$(find "$FUZZ_SHARED_TARGET_DIR/$host_target/coverage" -name "$target" -type f -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null | head -1)
      if [ -z "$binary" ]; then
        echo "    warning: coverage binary not found for ${package_label}/${target}"
        continue
      fi

      local target_lcov="$REPO_ROOT/coverage/fuzz-${package_slug}--${target}.lcov"
      "$llvm_cov" export "$binary" \
        -instr-profile="$cov_dir/merged.profdata" \
        -format=lcov \
        > "$target_lcov" 2>/dev/null
      all_lcov_files+=("$target_lcov")
      echo "    $target_lcov"
    done < <(fuzz_list_targets "$package_dir")
  done

  # Phase 3: Merge all fuzz LCOV files into one.
  if [ ${#all_lcov_files[@]} -gt 0 ]; then
    local merged="$COV_DIR/fuzz.lcov"
    cat "${all_lcov_files[@]}" > "$merged"
    echo ""
    echo "Merged fuzz coverage: $merged (${#all_lcov_files[@]} targets)"
  else
    echo ""
    echo "No fuzz coverage data generated"
  fi
}

# ── Merge + Summary ──────────────────────────────────────────────────────────

generate_summary() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Merging coverage reports"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  local lcov_files=()
  for f in "$COV_DIR"/nextest.lcov "$COV_DIR"/fuzz.lcov "$COV_DIR"/fuzz-*.lcov; do
    [ -f "$f" ] && lcov_files+=("$f")
  done

  if [ ${#lcov_files[@]} -eq 0 ]; then
    echo "No LCOV files to merge"
    return
  fi

  local merged="$COV_DIR/merged.lcov"
  cat "${lcov_files[@]}" > "$merged"

  local lines_found lines_hit fn_found fn_hit
  lines_found=$(awk -F: '/^LF:/{s+=$2} END{print s+0}' "$merged")
  lines_hit=$(awk -F: '/^LH:/{s+=$2} END{print s+0}' "$merged")
  fn_found=$(awk -F: '/^FNF:/{s+=$2} END{print s+0}' "$merged")
  fn_hit=$(awk -F: '/^FNH:/{s+=$2} END{print s+0}' "$merged")

  local line_pct="0.0" fn_pct="0.0"
  if [ "$lines_found" -gt 0 ]; then
    line_pct=$(awk "BEGIN{printf \"%.1f\", 100*$lines_hit/$lines_found}")
  fi
  if [ "$fn_found" -gt 0 ]; then
    fn_pct=$(awk "BEGIN{printf \"%.1f\", 100*$fn_hit/$fn_found}")
  fi

  echo ""
  echo "  Lines:     ${lines_hit}/${lines_found} (${line_pct}%)"
  echo "  Functions: ${fn_hit}/${fn_found} (${fn_pct}%)"
  echo "  Sources:   ${#lcov_files[@]} LCOV file(s) merged"
  echo "  Merged:    $merged"

  local sources_list=""
  for f in "${lcov_files[@]}"; do
    sources_list="${sources_list}
- \`$(basename "$f")\`"
  done

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

if [ "$RUN_NEXTEST" = true ]; then
  run_nextest_coverage
fi

if [ "$RUN_FUZZ" = true ]; then
  run_fuzz_coverage
fi

if [ "$RUN_NEXTEST" = true ] && [ "$RUN_FUZZ" = true ]; then
  generate_summary
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Coverage reports in: $COV_DIR/"
ls -la "$COV_DIR"/*.lcov 2>/dev/null || echo "(no LCOV files generated)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
