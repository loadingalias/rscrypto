#!/usr/bin/env bash
set -euo pipefail

# Host-only checks: fmt, check, optional rscrypto feature matrix, clippy,
# optional deny/audit, doc
# Usage: check.sh [--skip-feature-matrix] [--all] [crate1 crate2 ...]

# Prefer cargo-installed tools over any preinstalled runner tools.
export PATH="$HOME/.cargo/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

maybe_disable_sccache

RUN_FEATURE_MATRIX=true
CHECK_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --skip-feature-matrix)
      RUN_FEATURE_MATRIX=false
      ;;
    --feature-matrix)
      RUN_FEATURE_MATRIX=true
      ;;
    *)
      CHECK_ARGS+=("$arg")
      ;;
  esac
done

# Parse args and set CRATE_FLAGS, SCOPE_DESC
if [[ ${#CHECK_ARGS[@]} -gt 0 ]]; then
  get_crate_flags "${CHECK_ARGS[@]}"
else
  get_crate_flags
fi

# Determine if full workspace (for audit/deny)
FULL_WORKSPACE=false
if [[ "$CRATE_FLAGS" == "--workspace" ]]; then
  FULL_WORKSPACE=true
fi

RSCRYPTO_FEATURE_MATRIX_IN_SCOPE=false
if [[ "$FULL_WORKSPACE" == true || "$CRATE_FLAGS" == *" -p rscrypto"* ]]; then
  RSCRYPTO_FEATURE_MATRIX_IN_SCOPE=true
fi

CHECK_RSCRYPTO_FEATURE_MATRIX=false
if [[ "$RSCRYPTO_FEATURE_MATRIX_IN_SCOPE" == true && "$RUN_FEATURE_MATRIX" == true ]]; then
  CHECK_RSCRYPTO_FEATURE_MATRIX=true
fi

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT

echo "Host checks ${DIM}($SCOPE_DESC)${RESET}"

# Format
step "Formatting"
if ! cargo fmt --all >"$LOG_DIR/fmt.log" 2>&1; then
  fail
  show_error "$LOG_DIR/fmt.log"
  exit 1
fi
ok

step "Checking assembly ledger"
if ! "$SCRIPT_DIR/asm-ledger.sh" >"$LOG_DIR/asm-ledger.log" 2>&1; then
  fail
  show_error "$LOG_DIR/asm-ledger.log"
  exit 1
fi
ok

# Check
step "Checking"
# shellcheck disable=SC2086
if ! cargo check $CRATE_FLAGS --all-targets --all-features >"$LOG_DIR/check.log" 2>&1; then
  fail
  show_error "$LOG_DIR/check.log"
  exit 1
fi
ok

if [[ "$CHECK_RSCRYPTO_FEATURE_MATRIX" == true ]]; then
  step "Checking rscrypto no_std matrix"
  for feature_set in "" "alloc" "crc16" "crc24" "crc32" "crc64" "alloc,crc32" "sha2" "sha3" "xxh3" "hmac" "hmac-sha3" "kmac" "hkdf" "poly1305" "rsa" "rsa,getrandom" "x25519" "chacha20poly1305" "ascon-aead" "checksums" "hashes" "macs" "kdfs" "signatures" "key-exchange" "auth" "aead" "full"; do
    if [[ -n "$feature_set" ]]; then
      if ! cargo check -p rscrypto --no-default-features --features "$feature_set" --lib >>"$LOG_DIR/check.log" 2>&1; then
        fail
        show_error "$LOG_DIR/check.log"
        exit 1
      fi
    else
      if ! cargo check -p rscrypto --no-default-features --lib >>"$LOG_DIR/check.log" 2>&1; then
        fail
        show_error "$LOG_DIR/check.log"
        exit 1
      fi
    fi
  done
  ok

  step "Testing rscrypto feature matrix"
  if ! "$SCRIPT_DIR/../test/test-feature-matrix.sh" >>"$LOG_DIR/check.log" 2>&1; then
    fail
    show_error "$LOG_DIR/check.log"
    exit 1
  fi
  ok
elif [[ "$RSCRYPTO_FEATURE_MATRIX_IN_SCOPE" == true ]]; then
  skip "rscrypto feature matrix" "disabled for this check profile"
fi

# Clippy
step "Linting"
# shellcheck disable=SC2086
if ! cargo clippy $CRATE_FLAGS --all-targets --all-features -- -D warnings >"$LOG_DIR/clippy.log" 2>&1; then
  fail
  show_error "$LOG_DIR/clippy.log"
  exit 1
fi
ok

# Audit/Deny (workspace only). CI owns this in the dedicated supply-chain lane.
if [[ "$FULL_WORKSPACE" == true && "${RSCRYPTO_SKIP_CHECK_SUPPLY_CHAIN:-}" != "1" ]]; then
  step "Auditing deps"
  if ! cargo deny check all >"$LOG_DIR/deny.log" 2>&1; then
    fail
    show_error "$LOG_DIR/deny.log"
    exit 1
  fi
  # RustCrypto `rsa` is used only as a dev/test/bench oracle. Production RSA
  # verification is implemented in `src/auth/rsa.rs`; keep this scoped to the
  # known Marvin advisory until the oracle dependency is removed or fixed.
  if ! cargo audit --ignore RUSTSEC-2023-0071 >>"$LOG_DIR/deny.log" 2>&1; then
    fail
    show_error "$LOG_DIR/deny.log"
    exit 1
  fi
  ok
fi

# Documentation
step "Building docs"
# shellcheck disable=SC2086
if ! cargo doc $CRATE_FLAGS --no-deps --all-features >"$LOG_DIR/doc.log" 2>&1; then
  fail
  show_error "$LOG_DIR/doc.log"
  exit 1
fi
ok

echo "${GREEN}✓${RESET} Host checks passed"
