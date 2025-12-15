#!/usr/bin/env bash
set -euo pipefail

# Fuzz testing script for rscrypto
#
# Usage:
#   ./scripts/test/fuzz.sh                    # List available targets
#   ./scripts/test/fuzz.sh <target>           # Run target indefinitely
#   ./scripts/test/fuzz.sh <target> <seconds> # Run target for duration
#   ./scripts/test/fuzz.sh --smoke [seconds]  # Run all targets briefly
#   ./scripts/test/fuzz.sh --build            # Build without running
#   ./scripts/test/fuzz.sh --clean            # Clean fuzz artifacts

FUZZ_DIR="fuzz"
DEFAULT_SMOKE_DURATION=30

# All available fuzz targets
TARGETS=(
  "fuzz_crc32c"
  "fuzz_crc32"
  "fuzz_crc64"
  "fuzz_crc16"
  "fuzz_combine"
  "fuzz_streaming"
  "fuzz_differential"
)

show_help() {
  echo "Fuzz testing for rscrypto"
  echo ""
  echo "Usage:"
  echo "  $0                        List available targets"
  echo "  $0 <target>               Run target indefinitely"
  echo "  $0 <target> <seconds>     Run target for specified duration"
  echo "  $0 --smoke [seconds]      Run all targets briefly (default: ${DEFAULT_SMOKE_DURATION}s each)"
  echo "  $0 --build                Build fuzz targets without running"
  echo "  $0 --clean                Clean fuzz artifacts"
  echo "  $0 --coverage <target>    Generate coverage for target"
  echo ""
  echo "Available targets:"
  for target in "${TARGETS[@]}"; do
    echo "  $target"
  done
  echo ""
  echo "Requirements:"
  echo "  - Rust nightly toolchain"
  echo "  - cargo-fuzz: cargo install cargo-fuzz"
}

list_targets() {
  echo "Available fuzz targets:"
  echo ""
  for target in "${TARGETS[@]}"; do
    case "$target" in
      fuzz_crc32c)      echo "  $target      - CRC32-C (Castagnoli) implementation" ;;
      fuzz_crc32)       echo "  $target       - CRC32 (ISO-HDLC) implementation" ;;
      fuzz_crc64)       echo "  $target       - CRC64/XZ and CRC64/NVME implementations" ;;
      fuzz_crc16)       echo "  $target       - CRC16/IBM and CRC16/CCITT-FALSE" ;;
      fuzz_combine)     echo "  $target     - CRC combine operations" ;;
      fuzz_streaming)   echo "  $target   - Streaming API with arbitrary chunks" ;;
      fuzz_differential) echo "  $target - Differential testing vs reference impls" ;;
      *)                echo "  $target" ;;
    esac
  done
}

check_requirements() {
  if ! rustup run nightly --version &>/dev/null; then
    echo "Error: Rust nightly toolchain not found"
    echo "Install with: rustup toolchain install nightly"
    exit 1
  fi

  if ! cargo +nightly fuzz --version &>/dev/null; then
    echo "Error: cargo-fuzz not found"
    echo "Install with: cargo install cargo-fuzz"
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

  cd "$FUZZ_DIR"
  if [ -n "$duration" ]; then
    cargo +nightly fuzz run "$target" -- -max_total_time="$duration"
  else
    cargo +nightly fuzz run "$target"
  fi
}

run_smoke() {
  local duration="${1:-$DEFAULT_SMOKE_DURATION}"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🚀 Fuzz Smoke Test"
  echo "⏱️  Duration: ${duration}s per target"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  cd "$FUZZ_DIR"
  local failed=()

  for target in "${TARGETS[@]}"; do
    echo ""
    echo "▶ Running $target..."
    if cargo +nightly fuzz run "$target" -- -max_total_time="$duration"; then
      echo "✅ $target passed"
    else
      echo "❌ $target failed"
      failed+=("$target")
    fi
  done

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if [ ${#failed[@]} -eq 0 ]; then
    echo "✅ All fuzz targets passed!"
  else
    echo "❌ Failed targets: ${failed[*]}"
    exit 1
  fi
}

build_targets() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🔨 Building fuzz targets..."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  cd "$FUZZ_DIR"
  cargo +nightly fuzz build

  echo "✅ Fuzz targets built successfully!"
}

clean_artifacts() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🧹 Cleaning fuzz artifacts..."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  cd "$FUZZ_DIR"
  cargo +nightly fuzz clean

  echo "✅ Fuzz artifacts cleaned!"
}

run_coverage() {
  local target="$1"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "📊 Generating coverage for: $target"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  cd "$FUZZ_DIR"
  cargo +nightly fuzz coverage "$target"
}

# Main
case "${1:-}" in
  -h|--help)
    show_help
    ;;
  "")
    list_targets
    ;;
  --smoke)
    check_requirements
    run_smoke "${2:-}"
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
      exit 1
    fi
    run_coverage "$2"
    ;;
  *)
    check_requirements
    run_target "$1" "${2:-}"
    ;;
esac
