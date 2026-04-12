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

maybe_disable_sccache

COV_DIR="$REPO_ROOT/coverage"
FUZZ_DIR="$REPO_ROOT/fuzz"
FUZZ_CORPUS_SECS=${RSCRYPTO_COV_FUZZ_SECS:-30}

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

  if [ ! -d "$FUZZ_DIR" ]; then
    echo "No fuzz/ directory found — skipping"
    return 0
  fi

  pushd "$FUZZ_DIR" > /dev/null

  local targets
  targets=$(cargo fuzz list 2>/dev/null)
  if [ -z "$targets" ]; then
    echo "No fuzz targets found — skipping"
    popd > /dev/null
    return 0
  fi

  local host_target
  host_target="$(rustc -vV | sed -n 's|host: ||p')"

  # Phase 1: Seed corpus for any targets that don't have one yet.
  echo ""
  echo "Phase 1: Seeding corpus (${FUZZ_CORPUS_SECS}s per target without corpus)"
  for target in $targets; do
    local corpus_dir="corpus/$target"
    if [ -d "$corpus_dir" ] && [ "$(ls -A "$corpus_dir" 2>/dev/null)" ]; then
      echo "  $target: corpus exists ($(ls "$corpus_dir" | wc -l | tr -d ' ') entries)"
    else
      echo "  $target: building corpus (${FUZZ_CORPUS_SECS}s)..."
      mkdir -p "$corpus_dir"
      # Run briefly to seed corpus; suppress INFO noise; tolerate exit codes.
      local fuzz_exit=0
      cargo fuzz run "$target" \
        --target "$host_target" \
        -- -max_total_time="$FUZZ_CORPUS_SECS" -max_len=65536 \
        2>&1 | { grep -v "^INFO:" || true; } || fuzz_exit=$?
      if [ "$fuzz_exit" -ne 0 ]; then
        echo "    warning: corpus seeding exited $fuzz_exit (crash?), continuing"
      fi
    fi
  done

  # Phase 2: Generate coverage from corpus.
  echo ""
  echo "Phase 2: Generating coverage reports"

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
    popd > /dev/null
    return 1
  fi

  local all_lcov_files=()
  for target in $targets; do
    echo "  $target: generating coverage..."
    local cov_exit=0
    cargo fuzz coverage "$target" --target "$host_target" 2>&1 | { grep -v "^INFO:" || true; } || cov_exit=$?
    if [ "$cov_exit" -ne 0 ]; then
      echo "    warning: coverage generation exited $cov_exit, skipping"
      continue
    fi

    local cov_dir="coverage/$target"
    if [ ! -d "$cov_dir" ]; then
      echo "    warning: no coverage output for $target"
      continue
    fi

    # Merge profraw files into profdata.
    local profraw_files
    profraw_files=$(find "$cov_dir" -name "*.profraw" 2>/dev/null)
    if [ -z "$profraw_files" ]; then
      echo "    warning: no profraw files for $target"
      continue
    fi

    "$llvm_profdata" merge -sparse $profraw_files -o "$cov_dir/merged.profdata"

    # Find the coverage-instrumented binary.
    local binary
    binary=$(find "target/$host_target/coverage" -name "$target" -type f 2>/dev/null | head -1)
    if [ -z "$binary" ]; then
      echo "    warning: coverage binary not found for $target"
      continue
    fi

    # Export as LCOV.
    local target_lcov="$REPO_ROOT/coverage/fuzz-$target.lcov"
    "$llvm_cov" export "$binary" \
      -instr-profile="$cov_dir/merged.profdata" \
      -format=lcov \
      > "$target_lcov" 2>/dev/null
    all_lcov_files+=("$target_lcov")
    echo "    $target_lcov"
  done

  popd > /dev/null

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

# ── Main ──────────────────────────────────────────────────────────────────────

if [ "$RUN_NEXTEST" = true ]; then
  run_nextest_coverage
fi

if [ "$RUN_FUZZ" = true ]; then
  run_fuzz_coverage
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Coverage reports in: $COV_DIR/"
ls -la "$COV_DIR"/*.lcov 2>/dev/null || echo "(no LCOV files generated)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
