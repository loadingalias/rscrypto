#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fuzz Testing for rscrypto
#
# Fuzzing now has two layers:
#   1. full-surface regression harness in fuzz/
#   2. scoped feature-accurate harnesses in fuzz-packages/*
#
# Usage:
#   ./scripts/test/test-fuzz.sh                    # Build scoped packages, run full harness
#   ./scripts/test/test-fuzz.sh --all              # Build + run full and scoped packages
#   ./scripts/test/test-fuzz.sh --full             # Run the full harness only
#   ./scripts/test/test-fuzz.sh --scoped           # Run all scoped packages
#   ./scripts/test/test-fuzz.sh --scoped-build     # Build scoped packages only
#   ./scripts/test/test-fuzz.sh <target>           # Run specific target
#   ./scripts/test/test-fuzz.sh --build            # Build without running
#   ./scripts/test/test-fuzz.sh --list             # List available targets
#   ./scripts/test/test-fuzz.sh --clean            # Clean fuzz artifacts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

export RSCRYPTO_TEST_MODE=${RSCRYPTO_TEST_MODE:-local}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/fuzz-packages.sh
source "$SCRIPT_DIR/../lib/fuzz-packages.sh"

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
  echo "  $0                        Build scoped packages, run full harness"
  echo "  $0 --all                  Build and run full + scoped packages"
  echo "  $0 --full                 Run full harness targets (${DURATION_SECS}s each)"
  echo "  $0 --scoped               Run scoped targets (${DURATION_SECS}s each)"
  echo "  $0 --scoped-build         Build scoped packages without running"
  echo "  $0 <target>               Run specific target"
  echo "  $0 --build [--full|--scoped|--all]  Build selected fuzz packages"
  echo "  $0 --list                 List available targets by package"
  echo "  $0 --clean                Clean fuzz artifacts"
  echo ""
  echo "Environment variables:"
  echo "  RSCRYPTO_FUZZ_DURATION_SECS  Duration per target (default: 60)"
  echo "  RSCRYPTO_FUZZ_TIMEOUT_SECS   Timeout per test case (default: 10)"
  echo "  RSCRYPTO_FUZZ_RSS_LIMIT_MB   Memory limit in MB (default: 2048)"
  echo "  RSCRYPTO_FUZZ_MAX_LEN        Max input length (default: 65536)"
  echo "  RSCRYPTO_FUZZ_JOBS           Parallel jobs (default: 1)"
  echo "  RSCRYPTO_FUZZ_TARGET_DIR     Shared cargo target dir (default: fuzz/target)"
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
  if [ ! -d "$FUZZ_ROOT" ]; then
    echo "No fuzz/ directory found at repo root"
    exit 0
  fi

  discover_fuzz_packages
  if [ ${#FUZZ_ALL_PACKAGES[@]} -eq 0 ]; then
    echo "No cargo-fuzz packages found under fuzz/"
    exit 0
  fi
}

list_targets() {
  local package_dir

  echo "Available fuzz targets:"
  echo ""
  for package_dir in "${FUZZ_ALL_PACKAGES[@]}"; do
    echo "$(fuzz_package_label "$package_dir"):"
    while IFS= read -r target; do
      [ -z "$target" ] && continue
      echo "  $target"
    done < <(fuzz_list_targets "$package_dir")
    echo ""
  done
}

build_packages() {
  local scope=$1
  local package_dir

  fuzz_select_packages "$scope"
  if [ ${#SELECTED_FUZZ_PACKAGES[@]} -eq 0 ]; then
    echo "No fuzz packages selected for scope: $scope"
    return 0
  fi

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Building fuzz targets ($scope)..."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  for package_dir in "${SELECTED_FUZZ_PACKAGES[@]}"; do
    echo "  Building: $(fuzz_package_label "$package_dir")"
    fuzz_in_package "$package_dir" build --target "$(fuzz_host_target)"
  done

  echo "Selected fuzz packages built successfully"
}

clean_artifacts() {
  echo "Cleaning fuzz artifacts..."

  local package_dir
  for package_dir in "${FUZZ_ALL_PACKAGES[@]}"; do
    fuzz_in_package "$package_dir" clean 2>/dev/null || true
    rm -rf "$package_dir/artifacts" "$package_dir/corpus" "$package_dir/coverage"
  done
  rm -rf "$FUZZ_SHARED_TARGET_DIR"

  echo "Fuzz artifacts cleaned"
}

run_target_in_package() {
  local package_dir="$1"
  local target="$2"
  local duration="$3"

  local label
  label="$(fuzz_package_label "$package_dir")"

  echo "  Running: ${label}/${target} (${duration}s)"

  mkdir -p "$package_dir/artifacts/$target"

  local fuzz_args="-max_total_time=$duration -timeout=$TIMEOUT -rss_limit_mb=$RSS_LIMIT -max_len=$MAX_LEN"
  fuzz_args="$fuzz_args -artifact_prefix=$package_dir/artifacts/$target/"

  local fuzz_exit=0
  fuzz_in_package "$package_dir" run "$target" \
      --jobs="$JOBS" \
      --target "$(fuzz_host_target)" \
      -- $fuzz_args 2>&1 | { grep -v "^INFO:" || true; } || fuzz_exit=$?

  if [ "$fuzz_exit" -eq 0 ]; then
    echo "    Completed: ${label}/${target}"
    return 0
  fi

  echo "    Failed or found crash: ${label}/${target}"
  return 1
}

run_target() {
  local target="$1"
  local duration="$2"
  local search_order="scoped-first"

  case "${TARGET_SCOPE_OVERRIDE:-}" in
    full) search_order="full" ;;
    scoped) search_order="scoped" ;;
  esac

  local package_dir
  package_dir="$(fuzz_find_target_package "$target" "$search_order")" || {
    echo "Unknown fuzz target: $target"
    return 1
  }

  run_target_in_package "$package_dir" "$target" "$duration"
}

run_scope() {
  local scope="$1"
  local duration="$2"
  local package_dir
  local total=0
  local failed=0
  local crashed=""

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Fuzz Testing ($scope)"
  echo "Duration: ${duration}s per target"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  fuzz_select_packages "$scope"
  if [ ${#SELECTED_FUZZ_PACKAGES[@]} -eq 0 ]; then
    echo "No fuzz packages selected for scope: $scope"
    return 0
  fi

  local target
  for package_dir in "${SELECTED_FUZZ_PACKAGES[@]}"; do
    while IFS= read -r target; do
      [ -z "$target" ] && continue
      total=$((total + 1))
      if ! run_target_in_package "$package_dir" "$target" "$duration"; then
        failed=$((failed + 1))
        crashed="${crashed}  $(fuzz_package_label "$package_dir")/${target}\n"
      fi
    done < <(fuzz_list_targets "$package_dir")
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

ACTION="default"
PACKAGE_SCOPE="full"
TARGET_SCOPE_OVERRIDE=""
TARGET=""
TARGET_DURATION="$DURATION_SECS"

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    --list)
      ACTION="list"
      ;;
    --build)
      ACTION="build"
      ;;
    --clean)
      ACTION="clean"
      ;;
    --all)
      ACTION="run"
      PACKAGE_SCOPE="all"
      ;;
    --full)
      if [ "$ACTION" = "default" ]; then
        ACTION="run"
      fi
      PACKAGE_SCOPE="full"
      TARGET_SCOPE_OVERRIDE="full"
      ;;
    --scoped)
      if [ "$ACTION" = "default" ]; then
        ACTION="run"
      fi
      PACKAGE_SCOPE="scoped"
      TARGET_SCOPE_OVERRIDE="scoped"
      ;;
    --scoped-build)
      ACTION="build"
      PACKAGE_SCOPE="scoped"
      ;;
    *)
      if [ -z "$TARGET" ]; then
        TARGET="$1"
      else
        TARGET_DURATION="$1"
      fi
      ;;
  esac
  shift
done

check_requirements

case "$ACTION" in
  list)
    list_targets
    ;;
  build)
    build_packages "$PACKAGE_SCOPE"
    ;;
  clean)
    clean_artifacts
    ;;
  run)
    if [ -n "$TARGET" ]; then
      run_target "$TARGET" "$TARGET_DURATION"
    else
      run_scope "$PACKAGE_SCOPE" "$TARGET_DURATION"
    fi
    ;;
  default)
    if [ -n "$TARGET" ]; then
      run_target "$TARGET" "$TARGET_DURATION"
    else
      build_packages scoped
      run_scope full "$TARGET_DURATION"
    fi
    ;;
esac
