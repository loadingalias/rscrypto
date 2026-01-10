#!/usr/bin/env bash
set -euo pipefail

# Host-only checks: fmt, check, clippy, deny, audit, doc
# Usage: check.sh [--all] [crate1 crate2 ...]

# Prefer cargo-installed tools over any preinstalled runner tools.
export PATH="$HOME/.cargo/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

maybe_disable_sccache

# Parse args and set CRATE_FLAGS, SCOPE_DESC
get_crate_flags "$@"

# Determine if full workspace (for audit/deny)
FULL_WORKSPACE=false
if [[ "$CRATE_FLAGS" == "--workspace" ]]; then
  FULL_WORKSPACE=true
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

# Check
step "Checking"
# shellcheck disable=SC2086
if ! cargo check $CRATE_FLAGS --all-targets --all-features >"$LOG_DIR/check.log" 2>&1; then
  fail
  show_error "$LOG_DIR/check.log"
  exit 1
fi
ok

# Clippy
step "Linting"
# shellcheck disable=SC2086
if ! cargo clippy $CRATE_FLAGS --all-targets --all-features -- -D warnings >"$LOG_DIR/clippy.log" 2>&1; then
  fail
  show_error "$LOG_DIR/clippy.log"
  exit 1
fi
ok

# Audit/Deny (workspace only)
if [[ "$FULL_WORKSPACE" == true ]]; then
  step "Auditing deps"
  if ! cargo deny check all >"$LOG_DIR/deny.log" 2>&1; then
    fail
    show_error "$LOG_DIR/deny.log"
    exit 1
  fi
  if ! cargo audit >>"$LOG_DIR/deny.log" 2>&1; then
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
