#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# check.sh - Local development checks with cross-target compilation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage:
#   just check           - Full workspace + cross-target validation
#   just check foo bar   - Specific crates only (no cross-target)
#
# Cross-target validation:
#   - Uses zig as C compiler for cross-platform compilation
#   - Validates all targets from CI before pushing
#   - Catches cross-compilation issues locally (no "push and pray")
#   - CI compiles natively on actual target machines; this is pre-flight only
#
# Requirements:
#   - zig (brew install zig / apt install zig)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ZIG_CC="$SCRIPT_DIR/zig-cc.sh"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Target Tiers (aligned with CI workflow)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Tier A: Primary platforms
TIER_A_TARGETS=(
  "aarch64-apple-darwin"        # macOS ARM64 (host on M-series Macs)
  "x86_64-unknown-linux-gnu"    # Linux x86_64
  "aarch64-unknown-linux-gnu"   # Linux ARM64
  "x86_64-pc-windows-msvc"      # Windows x86_64
)

# Tier B: Secondary platforms
TIER_B_TARGETS=(
  "x86_64-unknown-linux-musl"   # Linux x86_64 (static linking)
  "aarch64-unknown-linux-musl"  # Linux ARM64 (static linking)
  "aarch64-pc-windows-msvc"     # Windows ARM64
)

# Tier C: Constrained targets (no_std, WASM)
TIER_C_TARGETS=(
  "wasm32-unknown-unknown"      # WebAssembly
  "wasm32-wasip1"               # WASI Preview 1
  "thumbv6m-none-eabi"          # Embedded no_std (Cortex-M0)
  "riscv32imac-unknown-none-elf" # RISC-V 32-bit
  "aarch64-unknown-none"        # ARM64 bare metal
  "x86_64-unknown-none"         # x86_64 bare metal
)

# All library crates to check (exported for subshells)
export ALL_CRATES="platform backend traits checksum"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cross-target check functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Full check suite for a single target. Returns 0 on success, 1 on failure.
# Runs: cargo check + cargo clippy (same as CI)
# Uses separate target directory per-target to avoid parallel build conflicts.
check_target() {
  local target=$1
  local mode=${2:-"std"}  # std or no_std

  # Use target-specific build directory to avoid parallel conflicts
  local target_dir="target/cross-check/$target"
  local log_file="$target_dir/check.log"
  mkdir -p "$target_dir"

  # Determine if this is the host target (no cross-compilation needed)
  local is_host=false
  if [[ "$target" == "aarch64-apple-darwin" && "$(uname -m)" == "arm64" && "$(uname)" == "Darwin" ]]; then
    is_host=true
  elif [[ "$target" == "x86_64-apple-darwin" && "$(uname -m)" == "x86_64" && "$(uname)" == "Darwin" ]]; then
    is_host=true
  elif [[ "$target" == "x86_64-unknown-linux-gnu" && "$(uname -m)" == "x86_64" && "$(uname)" == "Linux" ]]; then
    is_host=true
  elif [[ "$target" == "aarch64-unknown-linux-gnu" && "$(uname -m)" == "aarch64" && "$(uname)" == "Linux" ]]; then
    is_host=true
  fi

  # Ensure target is installed
  if ! rustup target list --installed 2>/dev/null | grep -q "^${target}$"; then
    rustup target add "$target" >/dev/null 2>&1 || true
  fi

  # Common env vars for cross-compilation
  local cross_env=""
  if [ "$is_host" = false ]; then
    cross_env="CC=$ZIG_CC RUSTC_WRAPPER= CARGO_TARGET_DIR=$target_dir"
  fi

  if [ "$mode" = "no_std" ]; then
    # no_std targets: check + clippy for each crate
    for crate in $ALL_CRATES; do
      if [ "$is_host" = true ]; then
        if ! cargo check -p "$crate" --no-default-features --target "$target" --lib >>"$log_file" 2>&1; then
          return 1
        fi
        if ! cargo clippy -p "$crate" --no-default-features --target "$target" --lib -- -D warnings >>"$log_file" 2>&1; then
          return 1
        fi
      else
        if ! env $cross_env cargo check -p "$crate" --no-default-features --target "$target" --lib >>"$log_file" 2>&1; then
          return 1
        fi
        if ! env $cross_env cargo clippy -p "$crate" --no-default-features --target "$target" --lib -- -D warnings >>"$log_file" 2>&1; then
          return 1
        fi
      fi
    done
    return 0
  else
    # std targets: full workspace check + clippy
    if [ "$is_host" = true ]; then
      cargo check --workspace --lib --all-features --target "$target" >>"$log_file" 2>&1 || return 1
      cargo clippy --workspace --lib --all-features --target "$target" -- -D warnings >>"$log_file" 2>&1 || return 1
    else
      env $cross_env cargo check --workspace --lib --all-features --target "$target" >>"$log_file" 2>&1 || return 1
      env $cross_env cargo clippy --workspace --lib --all-features --target "$target" -- -D warnings >>"$log_file" 2>&1 || return 1
    fi
    return $?
  fi
}

# Export function for use in subshells
export -f check_target

run_cross_target_checks() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Cross-target compilation checks (pre-flight for CI)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "(Running in parallel for speed)"

  # Create temp directory for parallel job results
  local RESULTS_DIR
  RESULTS_DIR=$(mktemp -d)
  trap 'rm -rf "$RESULTS_DIR"' RETURN

  # ─────────────────────────────────────────────────────────────────────────────
  # Launch all checks in parallel
  # ─────────────────────────────────────────────────────────────────────────────
  local pids=()

  # Tier A targets
  for target in "${TIER_A_TARGETS[@]}"; do
    case "$target" in
      x86_64-pc-windows-msvc|aarch64-pc-windows-msvc)
        # Skip Windows targets (no SDK headers in zig)
        echo "skip" > "$RESULTS_DIR/$target"
        ;;
      *)
        (
          if check_target "$target" "std"; then
            echo "pass" > "$RESULTS_DIR/$target"
          else
            echo "fail" > "$RESULTS_DIR/$target"
          fi
        ) &
        pids+=($!)
        ;;
    esac
  done

  # Tier B targets
  for target in "${TIER_B_TARGETS[@]}"; do
    case "$target" in
      aarch64-pc-windows-msvc)
        echo "skip" > "$RESULTS_DIR/$target"
        ;;
      *)
        (
          if check_target "$target" "std"; then
            echo "pass" > "$RESULTS_DIR/$target"
          else
            echo "fail" > "$RESULTS_DIR/$target"
          fi
        ) &
        pids+=($!)
        ;;
    esac
  done

  # Tier C targets (no_std mode)
  for target in "${TIER_C_TARGETS[@]}"; do
    (
      if check_target "$target" "no_std"; then
        echo "pass" > "$RESULTS_DIR/$target"
      else
        echo "fail" > "$RESULTS_DIR/$target"
      fi
    ) &
    pids+=($!)
  done

  # Wait for all parallel jobs
  echo ""
  echo "Waiting for ${#pids[@]} parallel checks..."
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  # ─────────────────────────────────────────────────────────────────────────────
  # Collect and display results
  # ─────────────────────────────────────────────────────────────────────────────
  local failed=0

  # Helper to show result and log on failure
  show_result() {
    local target=$1
    local result=$2
    local log_file="target/cross-check/$target/check.log"

    case "$result" in
      pass) echo "  ✓ $target" ;;
      skip) echo "  ○ $target (skipped: Windows SDK headers not in zig)" ;;
      fail)
        echo "  ✗ $target"
        failed=1
        # Show last 20 lines of log on failure
        if [ -f "$log_file" ]; then
          echo "    ┌─ Error log (last 20 lines):"
          tail -20 "$log_file" | sed 's/^/    │ /'
          echo "    └─ Full log: $log_file"
        fi
        ;;
      *)
        echo "  ? $target (unknown)"
        failed=1
        ;;
    esac
  }

  echo ""
  echo "Tier A: Primary platforms"
  echo "─────────────────────────────────────────────────────────────────────────────"
  for target in "${TIER_A_TARGETS[@]}"; do
    local result
    result=$(cat "$RESULTS_DIR/$target" 2>/dev/null || echo "unknown")
    show_result "$target" "$result"
  done

  echo ""
  echo "Tier B: Secondary platforms"
  echo "─────────────────────────────────────────────────────────────────────────────"
  for target in "${TIER_B_TARGETS[@]}"; do
    local result
    result=$(cat "$RESULTS_DIR/$target" 2>/dev/null || echo "unknown")
    show_result "$target" "$result"
  done

  echo ""
  echo "Tier C: Constrained targets (no_std mode)"
  echo "─────────────────────────────────────────────────────────────────────────────"
  for target in "${TIER_C_TARGETS[@]}"; do
    local result
    result=$(cat "$RESULTS_DIR/$target" 2>/dev/null || echo "unknown")
    show_result "$target" "$result"
  done

  echo ""
  if [ $failed -eq 0 ]; then
    echo "✓ All cross-target checks passed"
  else
    echo "✗ Some cross-target checks failed"
    return 1
  fi
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Argument Parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRATES=()
for arg in "$@"; do
  CRATES+=("$arg")
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Local Development Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Determine scope and run checks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ ${#CRATES[@]} -gt 0 ]; then
  # ─────────────────────────────────────────────────────────────────────────────
  # Mode: Specific crates (user-specified)
  # Cross-target checks: NO (user wants fast iteration on specific crates)
  # ─────────────────────────────────────────────────────────────────────────────
  CRATE_FLAGS=""
  for crate in "${CRATES[@]}"; do
    CRATE_FLAGS="$CRATE_FLAGS -p $crate"
  done
  echo "Running checks for specific crate(s): ${CRATES[*]}"
  echo ""

  echo "📐 Formatting..."
  cargo fmt --all

  echo ""
  echo "🔧 Checking..."
  # shellcheck disable=SC2086
  cargo check $CRATE_FLAGS --all-targets --all-features

  echo ""
  echo "📎 Linting..."
  # shellcheck disable=SC2086
  cargo clippy $CRATE_FLAGS --all-targets --all-features --fix --allow-dirty -- -D warnings

  echo ""
  echo "📚 Building docs..."
  # shellcheck disable=SC2086
  cargo doc $CRATE_FLAGS --no-deps --all-features

else
  # ─────────────────────────────────────────────────────────────────────────────
  # Mode: Full workspace
  # Cross-target checks: YES (run in parallel)
  # ─────────────────────────────────────────────────────────────────────────────
  echo "Running checks for entire workspace"
  echo ""

  echo "📐 Formatting..."
  cargo fmt --all

  echo ""
  echo "🔧 Checking..."
  cargo check --workspace --all-targets --all-features

  echo ""
  echo "📎 Linting..."
  cargo clippy --workspace --all-targets --all-features --fix --allow-dirty -- -D warnings

  echo ""
  echo "🔒 Dependency audit..."
  cargo deny check all

  echo ""
  echo "📚 Building docs..."
  RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features

  echo ""
  echo "🛡️ Security audit..."
  cargo audit

  # Cross-target checks (parallel)
  run_cross_target_checks
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All local checks passed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
