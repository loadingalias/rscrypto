#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fuzz Testing for rscrypto
#
# This project uses a centralized fuzz crate at ./fuzz/ that tests all
# workspace crates. Unlike per-crate fuzzing, all targets live in one place.
#
# Usage:
#   ./scripts/test/test-fuzz.sh                    # Run smoke test (30s each)
#   ./scripts/test/test-fuzz.sh <target>           # Run specific target
#   ./scripts/test/test-fuzz.sh <target> <secs>    # Run target for duration
#   ./scripts/test/test-fuzz.sh --all              # Run all targets
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
# Default: 60s local, override with RSCRYPTO_FUZZ_DURATION_SECS=300 for CI
DURATION_SECS=${RSCRYPTO_FUZZ_DURATION_SECS:-60}
TIMEOUT=${RSCRYPTO_FUZZ_TIMEOUT_SECS:-10}
RSS_LIMIT=${RSCRYPTO_FUZZ_RSS_LIMIT_MB:-2048}
MAX_LEN=${RSCRYPTO_FUZZ_MAX_LEN:-65536}
JOBS=${RSCRYPTO_FUZZ_JOBS:-1}

FUZZ_DIR="fuzz"

# Smoke targets (default): keep runtime reasonable while still exercising the
# most important invariants.
SMOKE_TARGETS=(
  # Platform crate
  "fuzz_caps"
  "fuzz_caps_ops"
  # Checksum crate (CRC64 focus)
  "fuzz_crc64"
  "fuzz_streaming"
  "fuzz_combine"
  "fuzz_differential"
)

# All targets (invoked via --all).
ALL_TARGETS=(
  # Platform crate
  "fuzz_caps"
  "fuzz_caps_ops"
  # Checksum crate
  "fuzz_crc16"
  "fuzz_crc32"
  "fuzz_crc32c"
  "fuzz_crc64"
  "fuzz_streaming"
  "fuzz_combine"
  "fuzz_differential"
)

# Skip if commit mode (fuzzing takes too long)
if [ "$RSCRYPTO_TEST_MODE" = "commit" ]; then
  echo "Fuzzing skipped in commit mode"
  exit 0
fi

show_help() {
  echo "Fuzz testing for rscrypto"
  echo ""
  echo "Usage:"
  echo "  $0                        Run smoke test (${DURATION_SECS}s per target)"
  echo "  $0 <target>               Run specific target indefinitely"
  echo "  $0 <target> <seconds>     Run target for specified duration"
  echo "  $0 --all                  Run all targets (${DURATION_SECS}s each)"
  echo "  $0 --smoke [seconds]      Run smoke test with custom duration"
  echo "  $0 --build                Build fuzz targets without running"
  echo "  $0 --list                 List available targets"
  echo "  $0 --clean                Clean fuzz artifacts"
  echo "  $0 --coverage <target>    Generate coverage report for target"
  echo ""
  echo "Environment variables:"
  echo "  RSCRYPTO_FUZZ_DURATION_SECS  Duration per target (default: 30)"
  echo "  RSCRYPTO_FUZZ_TIMEOUT_SECS   Timeout per test case (default: 10)"
  echo "  RSCRYPTO_FUZZ_RSS_LIMIT_MB   Memory limit in MB (default: 2048)"
  echo "  RSCRYPTO_FUZZ_MAX_LEN        Max input length (default: 65536)"
  echo "  RSCRYPTO_FUZZ_JOBS           Parallel jobs (default: 1)"
}

list_targets() {
  echo "Available fuzz targets:"
  echo ""
  echo "Platform crate:"
  echo "  fuzz_caps         Caps bitset invariants and has_bit consistency"
  echo "  fuzz_caps_ops     Caps union/intersection algebraic properties"
  echo ""
  echo "Checksum crate:"
  echo "  fuzz_crc32c       CRC32-C (Castagnoli) - iSCSI, SCTP, Btrfs"
  echo "  fuzz_crc32        CRC32 (ISO-HDLC) - Ethernet, gzip, PNG"
  echo "  fuzz_crc64        CRC64/XZ and CRC64/NVME variants"
  echo "  fuzz_crc16        CRC16/IBM and CRC16/CCITT-FALSE"
  echo "  fuzz_combine      CRC combine operations (parallel checksumming)"
  echo "  fuzz_streaming    Streaming API with arbitrary chunk sizes"
  echo "  fuzz_differential Differential testing vs reference implementations"
}

check_requirements() {
  # Note: We rely on rust-toolchain.toml for nightly version, not +nightly
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
    echo "Error: Fuzz directory not found at $FUZZ_DIR"
    exit 1
  fi
}

run_target() {
  local target="$1"
  local duration="${2:-}"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🔬 Fuzzing: $target"
  if [ -n "$duration" ]; then
    echo "⏱️  Duration: ${duration}s"
  else
    echo "⏱️  Duration: indefinite (Ctrl+C to stop)"
  fi
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  pushd "$FUZZ_DIR" > /dev/null

  FUZZ_ARGS="-timeout=$TIMEOUT -rss_limit_mb=$RSS_LIMIT -max_len=$MAX_LEN"
  FUZZ_ARGS="$FUZZ_ARGS -artifact_prefix=artifacts/$target/"

  mkdir -p "artifacts/$target"

  if [ -n "$duration" ]; then
    FUZZ_ARGS="$FUZZ_ARGS -max_total_time=$duration"
  fi

  # Use gnu target on Linux CI to avoid needing musl toolchain
  # cargo-fuzz defaults to musl on Linux which requires musl-gcc/g++
  TARGET_FLAG=""
  if [ "$(uname)" = "Linux" ] && [ "${RSCRYPTO_TEST_MODE:-local}" != "local" ]; then
    TARGET_FLAG="--target x86_64-unknown-linux-gnu"
  fi

  local result=0
  # shellcheck disable=SC2086
  if cargo fuzz run "$target" $TARGET_FLAG --jobs="$JOBS" -- $FUZZ_ARGS 2>&1 | grep -v "^INFO:"; then
    echo "✅ $target completed"
  else
    echo "❌ $target failed or found crash"
    result=1
  fi

  popd > /dev/null
  return $result
}

run_smoke() {
  local duration="${1:-$DURATION_SECS}"
  shift || true
  local targets=("$@")

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🚀 Fuzz Smoke Test"
  echo "⏱️  Duration: ${duration}s per target"
  echo "🎯 Targets: ${#targets[@]}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  local failed=()
  local passed=0

  for target in "${targets[@]}"; do
    if run_target "$target" "$duration"; then
      passed=$((passed + 1))
    else
      failed+=("$target")
    fi
    echo ""
  done

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Fuzz Summary: $passed/${#targets[@]} targets passed"

  if [ ${#failed[@]} -gt 0 ]; then
    echo "❌ Failed targets: ${failed[*]}"
    exit 1
  else
    echo "✅ All fuzz targets passed!"
  fi
}

build_targets() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🔨 Building fuzz targets..."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  cd "$FUZZ_DIR"

  # Use gnu target on Linux CI to avoid needing musl toolchain
  TARGET_FLAG=""
  if [ "$(uname)" = "Linux" ] && [ "${RSCRYPTO_TEST_MODE:-local}" != "local" ]; then
    TARGET_FLAG="--target x86_64-unknown-linux-gnu"
  fi

  # shellcheck disable=SC2086
  cargo fuzz build $TARGET_FLAG

  echo "✅ Fuzz targets built successfully!"
}

clean_artifacts() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🧹 Cleaning fuzz artifacts..."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  cd "$FUZZ_DIR"
  cargo fuzz clean
  rm -rf artifacts/

  echo "✅ Fuzz artifacts cleaned!"
}

run_coverage() {
  local target="$1"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "📊 Generating coverage for: $target"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  cd "$FUZZ_DIR"

  # Use gnu target on Linux CI to avoid needing musl toolchain
  TARGET_FLAG=""
  if [ "$(uname)" = "Linux" ] && [ "${RSCRYPTO_TEST_MODE:-local}" != "local" ]; then
    TARGET_FLAG="--target x86_64-unknown-linux-gnu"
  fi

  # shellcheck disable=SC2086
  cargo fuzz coverage "$target" $TARGET_FLAG

  echo "✅ Coverage report generated in: fuzz/coverage/$target/"
}

# Main
case "${1:-}" in
  -h|--help)
    show_help
    ;;
  --list)
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
  --coverage)
    check_requirements
    if [ -z "${2:-}" ]; then
      echo "Error: --coverage requires a target name"
      echo "Usage: $0 --coverage <target>"
      exit 1
    fi
    run_coverage "$2"
    ;;
  --all)
    check_requirements
    run_smoke "${2:-$DURATION_SECS}" "${ALL_TARGETS[@]}"
    ;;
  --smoke)
    check_requirements
    run_smoke "${2:-$DURATION_SECS}" "${SMOKE_TARGETS[@]}"
    ;;
  "")
    # Default: run smoke test
    check_requirements
    run_smoke "$DURATION_SECS" "${SMOKE_TARGETS[@]}"
    ;;
  *)
    # Specific target
    check_requirements
    run_target "$1" "${2:-}"
    ;;
esac
