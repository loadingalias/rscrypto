#!/usr/bin/env bash
set -euo pipefail

# Windows cross-compilation checks via cargo-xwin
# Usage: check-win.sh [--all] [crate1 crate2 ...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/targets.sh
source "$SCRIPT_DIR/../lib/targets.sh"

# Check for cargo-xwin
if ! cargo xwin --version >/dev/null 2>&1; then
  echo "Windows targets ${DIM}(skipped)${RESET}"
  skip "cargo-xwin not installed" "cargo install cargo-xwin && brew install llvm"
  exit 0
fi

maybe_disable_sccache

# Parse args and set CRATE_FLAGS, SCOPE_DESC
get_crate_flags "$@"

# Keep cargo-xwin's SDK cache inside the workspace `target/` dir so this script
# works in sandboxed environments that disallow writes to user cache locations.
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
XWIN_CACHE_DIR_DEFAULT="$REPO_ROOT/target/cross-check/xwin-cache"
mkdir -p "$XWIN_CACHE_DIR_DEFAULT"

TUNE_IN_SCOPE=false
if [[ "$CRATE_FLAGS" == "--workspace" || "$CRATE_FLAGS" == *"-p tune"* || "$CRATE_FLAGS" == *"-p checksum"* ]]; then
  TUNE_IN_SCOPE=true
fi

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT

echo "Windows targets ${DIM}($SCOPE_DESC)${RESET}"

if [[ ${#WIN_TARGETS[@]} -eq 0 ]]; then
  skip "no Windows targets configured" "config/target-matrix.toml"
  exit 0
fi

# Initialize xwin cache once (avoids race conditions)
step "Initializing SDK cache"
if ! XWIN_CACHE_DIR="$XWIN_CACHE_DIR_DEFAULT" \
     CARGO_TARGET_DIR="target/cross-check/xwin-init" \
     cargo xwin check -p traits --lib --target x86_64-pc-windows-msvc \
     >"$LOG_DIR/xwin-init.log" 2>&1; then
  # In sandboxed/offline environments, cargo-xwin can't fetch the MSVC CRT / SDK.
  # Treat that as a skip rather than a hard failure, while still surfacing
  # unexpected errors.
  if tail -80 "$LOG_DIR/xwin-init.log" | grep -Eq \
      'failed to lookup address information|Could not resolve host|Name or service not known|Temporary failure in name resolution|HTTP GET request .* failed'; then
    printf " ${YELLOW}○${RESET}\n"
    skip "Windows targets" "SDK cache init requires network downloads"
    exit 0
  fi

  fail
  show_error "$LOG_DIR/xwin-init.log"
  exit 1
fi
ok

for target in "${WIN_TARGETS[@]}"; do
  ensure_target "$target"
done

for target in "${WIN_TARGETS[@]}"; do
  mkdir -p "target/cross-check/$target"
done

pids=()
logs=()
targets=()

for i in "${!WIN_TARGETS[@]}"; do
  target="${WIN_TARGETS[$i]}"
  target_dir="target/cross-check/$target"
  target_cache_dir="$XWIN_CACHE_DIR_DEFAULT/$target"
  mkdir -p "$target_cache_dir"
  log_file="$LOG_DIR/$target.log"
  logs[$i]="$log_file"
  targets[$i]="$target"

  (
    # shellcheck disable=SC2086
    if ! XWIN_CACHE_DIR="$target_cache_dir" \
         CARGO_TARGET_DIR="$target_dir" \
         cargo xwin clippy $CRATE_FLAGS --lib --all-features --target "$target" -- -D warnings \
         >"$log_file" 2>&1; then
      exit 1
    fi

    if [[ "$TUNE_IN_SCOPE" == true && "$target" == "x86_64-pc-windows-msvc" ]]; then
      if ! XWIN_CACHE_DIR="$target_cache_dir" \
           CARGO_TARGET_DIR="$target_dir" \
           cargo xwin clippy -p tune --bin rscrypto-blake3-boundary --all-features --target "$target" -- -D warnings \
           >>"$log_file" 2>&1; then
        exit 1
      fi
    fi
  ) &
  pids[$i]=$!
done

FAILED=0
for i in "${!targets[@]}"; do
  target="${targets[$i]}"
  short_name="${target%-pc-windows-msvc}" # x86_64 or aarch64

  step "$short_name clippy"
  if wait "${pids[$i]}"; then
    ok
    if [[ "$TUNE_IN_SCOPE" == true && "$target" == "x86_64-pc-windows-msvc" ]]; then
      step "$short_name rscrypto-blake3-boundary"
      ok
    fi
  else
    fail
    show_error "${logs[$i]}"
    FAILED=1
  fi
done

if [ $FAILED -ne 0 ]; then
  exit 1
fi

echo "${GREEN}✓${RESET} Windows targets passed"
