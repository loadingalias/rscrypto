#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fuzz Testing for rscrypto
#
# Root-level fuzzing: targets live in fuzz/fuzz_targets/*.rs and are built via
# cargo-fuzz from the fuzz/ workspace.
#
# Usage:
#   ./scripts/test/test-fuzz.sh                    # Run all targets (60s each)
#   ./scripts/test/test-fuzz.sh --all              # Run all targets (60s each)
#   ./scripts/test/test-fuzz.sh <target>           # Run specific target
#   ./scripts/test/test-fuzz.sh --build            # Build without running
#   ./scripts/test/test-fuzz.sh --list             # List available targets
#   ./scripts/test/test-fuzz.sh --clean            # Clean fuzz artifacts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

export RSCRYPTO_TEST_MODE=${RSCRYPTO_TEST_MODE:-local}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FUZZ_DIR="$REPO_ROOT/fuzz"

# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

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
  echo "Fuzz testing for rscrypto"
  echo ""
  echo "Usage:"
  echo "  $0                        Run all targets (${DURATION_SECS}s per target)"
  echo "  $0 --all                  Run all targets (${DURATION_SECS}s each)"
  echo "  $0 <target>               Run specific target"
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
  if [ ! -d "$FUZZ_DIR" ]; then
    echo "No fuzz/ directory found at repo root"
    exit 0
  fi
}

list_targets() {
  echo "Available fuzz targets:"
  echo ""
  pushd "$FUZZ_DIR" > /dev/null
  cargo fuzz list 2>/dev/null
  popd > /dev/null
}

build_targets() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Building fuzz targets..."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  pushd "$FUZZ_DIR" > /dev/null
  cargo fuzz build --target "$(rustc -vV | sed -n 's|host: ||p')"
  popd > /dev/null

  echo "Fuzz targets built successfully"
}

clean_artifacts() {
  echo "Cleaning fuzz artifacts..."

  pushd "$FUZZ_DIR" > /dev/null
  cargo fuzz clean 2>/dev/null
  rm -rf artifacts/ corpus/ coverage/
  popd > /dev/null

  echo "Fuzz artifacts cleaned"
}

run_target() {
  local target="$1"
  local duration="$2"

  echo "  Running: $target (${duration}s)"

  pushd "$FUZZ_DIR" > /dev/null

  mkdir -p "artifacts/$target"

  local fuzz_args="-max_total_time=$duration -timeout=$TIMEOUT -rss_limit_mb=$RSS_LIMIT -max_len=$MAX_LEN"
  fuzz_args="$fuzz_args -artifact_prefix=artifacts/$target/"

  # shellcheck disable=SC2086
  # Capture exit code from cargo-fuzz, not from the grep filter.
  # With pipefail, `grep -v` returning 1 (all lines filtered) would mask a
  # successful fuzz run.  Use a variable to isolate the two exit codes.
  local fuzz_exit=0
  cargo fuzz run "$target" \
      --jobs="$JOBS" \
      --target "$(rustc -vV | sed -n 's|host: ||p')" \
      -- $fuzz_args 2>&1 | { grep -v "^INFO:" || true; } || fuzz_exit=$?

  if [ "$fuzz_exit" -eq 0 ]; then
    echo "    Completed: $target"
    popd > /dev/null
    return 0
  else
    echo "    Failed or found crash: $target"
    popd > /dev/null
    return 1
  fi
}

run_all() {
  local duration="$1"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Fuzz Testing"
  echo "Duration: ${duration}s per target"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  pushd "$FUZZ_DIR" > /dev/null
  local targets
  targets=$(cargo fuzz list 2>/dev/null)
  popd > /dev/null

  if [ -z "$targets" ]; then
    echo "No fuzz targets found"
    exit 0
  fi

  local total=0
  local failed=0
  local crashed=""

  for target in $targets; do
    total=$((total + 1))
    if ! run_target "$target" "$duration"; then
      failed=$((failed + 1))
      crashed="${crashed}  ${target}\n"
    fi
  done

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Summary: $total targets, $failed failed"
  if [ $failed -gt 0 ]; then
    echo -e "Crashed:\n$crashed"
    exit 1
  else
    echo "All fuzz targets passed"
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
  --all|"")
    check_requirements
    run_all "${2:-$DURATION_SECS}"
    ;;
  *)
    # Specific target
    check_requirements
    run_target "$1" "${2:-$DURATION_SECS}"
    ;;
esac
