#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fuzz Testing for rscrypto
#
# Per-crate fuzzing: each crate with a fuzz/ directory is tested independently.
# This matches the rail codebase pattern for better isolation and modularity.
#
# Usage:
#   ./scripts/test/test-fuzz.sh                    # Run smoke test (60s each)
#   ./scripts/test/test-fuzz.sh --all              # Run all targets (60s each)
#   ./scripts/test/test-fuzz.sh <crate>            # Run specific crate's fuzz targets
#   ./scripts/test/test-fuzz.sh --build            # Build without running
#   ./scripts/test/test-fuzz.sh --list             # List available targets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

export RSCRYPTO_TEST_MODE=${RSCRYPTO_TEST_MODE:-local}

maybe_disable_sccache() {
  if [[ -n "${RUSTC_WRAPPER:-}" && "${RUSTC_WRAPPER##*/}" == "sccache" ]]; then
    if ! "$RUSTC_WRAPPER" rustc -vV >/dev/null 2>&1; then
      echo "⚠️  WARNING: sccache is configured but not usable; disabling RUSTC_WRAPPER for this run."
      export RUSTC_WRAPPER=
    fi
  fi
}

maybe_disable_sccache

# Configuration (can be overridden via environment)
DURATION_SECS=${RSCRYPTO_FUZZ_DURATION_SECS:-60}
TIMEOUT=${RSCRYPTO_FUZZ_TIMEOUT_SECS:-10}
RSS_LIMIT=${RSCRYPTO_FUZZ_RSS_LIMIT_MB:-2048}
MAX_LEN=${RSCRYPTO_FUZZ_MAX_LEN:-65536}
JOBS=${RSCRYPTO_FUZZ_JOBS:-1}

# Skip if commit mode (fuzzing takes too long)
if [ "$RSCRYPTO_TEST_MODE" = "commit" ]; then
  echo "Fuzzing skipped in commit mode"
  exit 0
fi

show_help() {
  echo "Fuzz testing for rscrypto (per-crate)"
  echo ""
  echo "Usage:"
  echo "  $0                        Run smoke test (${DURATION_SECS}s per target)"
  echo "  $0 --all                  Run all targets (${DURATION_SECS}s each)"
  echo "  $0 <crate>                Run specific crate's fuzz targets"
  echo "  $0 --build                Build fuzz targets without running"
  echo "  $0 --list                 List available targets"
  echo "  $0 --clean                Clean fuzz artifacts"
  echo ""
  echo "Environment variables:"
  echo "  RSCRYPTO_FUZZ_DURATION_SECS  Duration per target (default: 60)"
  echo "  RSCRYPTO_FUZZ_TIMEOUT_SECS   Timeout per test case (default: 10)"
  echo "  RSCRYPTO_FUZZ_RSS_LIMIT_MB   Memory limit in MB (default: 2048)"
  echo "  RSCRYPTO_FUZZ_MAX_LEN        Max input length (default: 65536)"
  echo "  RSCRYPTO_FUZZ_JOBS           Parallel jobs (default: 1)"
}

check_requirements() {
  if ! cargo --version &>/dev/null; then
    echo "Error: Rust toolchain not found"
    exit 1
  fi

  if ! cargo fuzz --version &>/dev/null; then
    echo "Error: cargo-fuzz not found"
    echo "Install with: cargo install cargo-fuzz"
    exit 1
  fi
}

# Find all crates with fuzz/ directories
find_fuzz_dirs() {
  find crates -type d -name fuzz 2>/dev/null || true
}

list_targets() {
  echo "Available fuzz targets:"
  echo ""

  local fuzz_dirs
  fuzz_dirs=$(find_fuzz_dirs)

  if [ -z "$fuzz_dirs" ]; then
    echo "  No fuzz directories found"
    return
  fi

  for fuzz_dir in $fuzz_dirs; do
    local crate_name
    crate_name=$(basename "$(dirname "$fuzz_dir")")
    echo "━━━ $crate_name ━━━"

    pushd "$fuzz_dir" > /dev/null
    local targets
    targets=$(cargo fuzz list 2>/dev/null || true)
    if [ -z "$targets" ]; then
      echo "  (no targets)"
    else
      for target in $targets; do
        echo "  $target"
      done
    fi
    popd > /dev/null
    echo ""
  done
}

build_targets() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🔨 Building fuzz targets..."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  local fuzz_dirs
  fuzz_dirs=$(find_fuzz_dirs)

  if [ -z "$fuzz_dirs" ]; then
    echo "No fuzz directories found"
    exit 0
  fi

  for fuzz_dir in $fuzz_dirs; do
    local crate_name
    crate_name=$(basename "$(dirname "$fuzz_dir")")
    echo "Building: $crate_name"

    pushd "$fuzz_dir" > /dev/null
    cargo fuzz build --target "$(rustc -vV | sed -n 's|host: ||p')"
    popd > /dev/null
  done

  echo "✅ Fuzz targets built successfully!"
}

clean_artifacts() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🧹 Cleaning fuzz artifacts..."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  local fuzz_dirs
  fuzz_dirs=$(find_fuzz_dirs)

  for fuzz_dir in $fuzz_dirs; do
    pushd "$fuzz_dir" > /dev/null
    cargo fuzz clean 2>/dev/null || true
    rm -rf artifacts/ corpus/ coverage/
    popd > /dev/null
  done

  echo "✅ Fuzz artifacts cleaned!"
}

run_fuzz() {
  local fuzz_dirs="$1"
  local duration="$2"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🚀 Fuzz Testing"
  echo "⏱️  Duration: ${duration}s per target"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  local total_targets=0
  local failed_targets=0
  local crashed_targets=""

  for fuzz_dir in $fuzz_dirs; do
    local crate_name
    crate_name=$(basename "$(dirname "$fuzz_dir")")
    echo "━━━ Crate: $crate_name ━━━"

    pushd "$fuzz_dir" > /dev/null

    local targets
    targets=$(cargo fuzz list 2>/dev/null || true)
    if [ -z "$targets" ]; then
      echo "  No fuzz targets found"
      popd > /dev/null
      continue
    fi

    for target in $targets; do
      total_targets=$((total_targets + 1))
      echo "  Running: $target"

      mkdir -p "artifacts/$target"

      local fuzz_args="-max_total_time=$duration -timeout=$TIMEOUT -rss_limit_mb=$RSS_LIMIT -max_len=$MAX_LEN"
      fuzz_args="$fuzz_args -artifact_prefix=artifacts/$target/"

      # shellcheck disable=SC2086
      if cargo fuzz run "$target" \
          --jobs="$JOBS" \
          --target "$(rustc -vV | sed -n 's|host: ||p')" \
          -- $fuzz_args 2>&1 | grep -v "^INFO:"; then
        echo "    ✅ Completed"
      else
        echo "    ❌ Failed or found crash"
        failed_targets=$((failed_targets + 1))
        crashed_targets="${crashed_targets}${crate_name}/${target}\n"
      fi
    done

    popd > /dev/null
    echo ""
  done

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Fuzzing Summary: $total_targets targets, $failed_targets failed"
  if [ $failed_targets -gt 0 ]; then
    echo -e "Crashed: $crashed_targets"
    exit 1
  else
    echo "✅ All fuzz targets passed!"
  fi
}

# Main
case "${1:-}" in
  -h|--help)
    show_help
    ;;
  --list)
    check_requirements
    list_targets
    ;;
  --build)
    check_requirements
    build_targets
    ;;
  --clean)
    check_requirements
    clean_artifacts
    ;;
  --all)
    check_requirements
    fuzz_dirs=$(find_fuzz_dirs)
    if [ -z "$fuzz_dirs" ]; then
      echo "No fuzz directories found in crates/"
      exit 0
    fi
    run_fuzz "$fuzz_dirs" "${2:-$DURATION_SECS}"
    ;;
  "")
    # Default: run all (same as --all for per-crate fuzzing)
    check_requirements
    fuzz_dirs=$(find_fuzz_dirs)
    if [ -z "$fuzz_dirs" ]; then
      echo "No fuzz directories found in crates/"
      exit 0
    fi
    run_fuzz "$fuzz_dirs" "$DURATION_SECS"
    ;;
  *)
    # Specific crate
    check_requirements
    crate_fuzz_dir="crates/$1/fuzz"
    if [ ! -d "$crate_fuzz_dir" ]; then
      echo "Error: No fuzz directory found for crate: $1"
      echo "Expected: $crate_fuzz_dir"
      exit 1
    fi
    run_fuzz "$crate_fuzz_dir" "${2:-$DURATION_SECS}"
    ;;
esac
