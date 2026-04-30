#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Coverage Reporting for rscrypto
#
# Generates coverage reports from two sources:
#   1. Test suite via cargo-llvm-cov + nextest  → coverage/nextest.lcov
#   2. Fuzz corpus via cargo fuzz coverage      → coverage/fuzz-profile-manifest.txt
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

clean_coverage_outputs() {
  if [ "$RUN_NEXTEST" = true ]; then
    rm -f "$COV_DIR/nextest.lcov"
    rm -rf "$COV_DIR/html-nextest"
  fi

  if [ "$RUN_FUZZ" = true ]; then
    local host_target
    host_target="$(rustc -vV | sed -n 's|host: ||p')"

    rm -f "$COV_DIR"/fuzz*.lcov "$COV_DIR/fuzz-profile-manifest.txt"
    rm -rf "$FUZZ_ROOT/coverage"
    if [ -d "$FUZZ_SCOPED_ROOT" ]; then
      find "$FUZZ_SCOPED_ROOT" -mindepth 2 -maxdepth 2 -type d -name coverage -prune -exec rm -rf {} +
    fi
    rm -rf "$REPO_ROOT/target/$host_target/coverage"
    rm -rf "$FUZZ_SHARED_TARGET_DIR/$host_target/coverage"
  fi

  if [ "$RUN_NEXTEST" = true ] && [ "$RUN_FUZZ" = true ]; then
    rm -f "$COV_DIR/merged.lcov" "$COV_DIR/SUMMARY.md"
  fi
}

clean_coverage_outputs

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

  cargo llvm-cov clean --workspace
  cargo llvm-cov nextest \
    --all-features \
    --workspace \
    --lcov \
    --output-path "$lcov_path"

  echo ""
  echo "Nextest coverage: $lcov_path"

  if [ "$GEN_HTML" = true ]; then
    cargo llvm-cov clean --workspace
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

  # Locate llvm-profdata from the toolchain. cargo-fuzz coverage emits
  # libFuzzer profile data, not Rust source coverage maps, so this lane
  # validates and preserves profiles instead of exporting LCOV.
  local llvm_profdata
  local toolchain_lib
  toolchain_lib="$(rustc --print sysroot)/lib/rustlib/$host_target/bin"
  if [ -x "$toolchain_lib/llvm-profdata" ]; then
    llvm_profdata="$toolchain_lib/llvm-profdata"
  elif command -v llvm-profdata &>/dev/null; then
    llvm_profdata="llvm-profdata"
  else
    echo "Error: llvm-profdata not found"
    echo "Install with: rustup component add llvm-tools-preview"
    return 1
  fi

  local profile_manifest="$COV_DIR/fuzz-profile-manifest.txt"
  : > "$profile_manifest"

  local generated_count=0
  local failed_count=0
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
  echo "Phase 2: Generating coverage profiles"
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
        echo "    error: coverage generation exited $cov_exit"
        failed_count=$((failed_count + 1))
        continue
      fi

      local cov_dir="$package_dir/coverage/$target"
      if [ ! -d "$cov_dir" ]; then
        echo "    error: no coverage output for ${package_label}/${target}"
        failed_count=$((failed_count + 1))
        continue
      fi

      local profile_path="$cov_dir/coverage.profdata"
      if [ ! -f "$profile_path" ]; then
        local profraw_files
        local merged_profile="$cov_dir/merged.profdata"
        profraw_files=$(find "$cov_dir" -name "*.profraw" -type f 2>/dev/null)
        if [ -z "$profraw_files" ]; then
          echo "    error: no profile data for ${package_label}/${target}"
          failed_count=$((failed_count + 1))
          continue
        fi

        if ! "$llvm_profdata" merge -sparse $profraw_files -o "$merged_profile"; then
          echo "    error: llvm-profdata merge failed for ${package_label}/${target}"
          failed_count=$((failed_count + 1))
          continue
        fi
        profile_path="$merged_profile"
      fi

      if ! "$llvm_profdata" show "$profile_path" >/dev/null; then
        echo "    error: invalid profile data for ${package_label}/${target}"
        failed_count=$((failed_count + 1))
        continue
      fi

      generated_count=$((generated_count + 1))
      printf '%s\t%s\t%s\n' "$package_label" "$target" "${profile_path#$REPO_ROOT/}" >> "$profile_manifest"
      echo "    profile: ${profile_path#$REPO_ROOT/}"

      # Keep optional LCOV files generated by older cargo-fuzz/toolchain
      # combinations, but do not synthesize a source coverage report from
      # libFuzzer edge profiles.
      local target_lcov="$cov_dir/lcov.info"
      if [ -f "$target_lcov" ]; then
        local output_lcov="$REPO_ROOT/coverage/fuzz-${package_slug}--${target}.lcov"
        cp "$target_lcov" "$output_lcov"
        echo "    lcov: ${output_lcov#$REPO_ROOT/}"
      fi
    done < <(fuzz_list_targets "$package_dir")
  done

  if [ "$failed_count" -ne 0 ]; then
    echo ""
    echo "Error: fuzz coverage failed for $failed_count target(s)"
    return 1
  fi

  local all_lcov_files=()
  while IFS= read -r -d '' lcov_file; do
    all_lcov_files+=("$lcov_file")
  done < <(find "$COV_DIR" -maxdepth 1 -name "fuzz-*.lcov" -type f -print0 2>/dev/null)

  if [ ${#all_lcov_files[@]} -gt 0 ]; then
    local merged="$COV_DIR/fuzz.lcov"
    cat "${all_lcov_files[@]}" > "$merged"
    echo ""
    echo "Merged fuzz LCOV: $merged (${#all_lcov_files[@]} targets)"
  fi

  if [ "$generated_count" -gt 0 ]; then
    echo ""
    echo "Fuzz coverage profiles: $profile_manifest ($generated_count targets)"
  else
    echo ""
    echo "No fuzz coverage profiles generated"
  fi
}

# ── Merge + Summary ──────────────────────────────────────────────────────────

generate_summary() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Merging coverage reports"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  local lcov_files=()
  for f in "$COV_DIR"/nextest.lcov "$COV_DIR"/fuzz.lcov; do
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
if find "$COV_DIR" -maxdepth 1 \( -name "*.lcov" -o -name "fuzz-profile-manifest.txt" \) -type f -print -quit | grep -q .; then
  find "$COV_DIR" -maxdepth 1 \( -name "*.lcov" -o -name "fuzz-profile-manifest.txt" \) -type f -exec ls -la {} +
else
  echo "(no coverage files generated)"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
