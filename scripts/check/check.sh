#!/usr/bin/env bash
set -euo pipefail

# Host-only Cargo checks: fmt, check, clippy, optional deny/audit, and docs.
# Repository policy and feature contracts have separate executors.
# Usage: check.sh [--all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

FORCE_ALL=false
case "$#" in
  0) ;;
  1)
    [[ "$1" == --all ]] || { echo "Usage: $0 [--all]" >&2; exit 2; }
    FORCE_ALL=true
    ;;
  *) echo "Usage: $0 [--all]" >&2; exit 2 ;;
esac

if [[ "$FORCE_ALL" == false ]]; then
  rail_prime_plan
fi

work_required() {
  local work_id=$1
  if [[ "$FORCE_ALL" == true ]]; then
    return 0
  fi
  local status=0
  rail_work_required "$work_id" || status=$?
  [[ "$status" -le 1 ]] || exit "$status"
  return "$status"
}

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT

echo "Host checks"

# Format
if work_required cargo.fmt; then
  step "Formatting"
  if ! cargo fmt --all -- --check >"$LOG_DIR/fmt.log" 2>&1; then
    fail
    show_error "$LOG_DIR/fmt.log"
    exit 1
  fi
  ok
else
  skip "Formatting" "not required by Cargo Rail"
fi

# Check
SCOPE_STATUS=0
select_cargo_scope cargo.build "$FORCE_ALL" || SCOPE_STATUS=$?
if [[ "$SCOPE_STATUS" -gt 1 ]]; then
  exit "$SCOPE_STATUS"
fi
if [[ "$SCOPE_STATUS" -ne 0 ]]; then
  skip "Checking" "no affected targets"
else
  step "Checking"
  if ! cargo check "${CARGO_ARGS[@]:+${CARGO_ARGS[@]}}" --all-targets --all-features --locked >"$LOG_DIR/check.log" 2>&1; then
    fail
    show_error "$LOG_DIR/check.log"
    exit 1
  fi
  ok
fi

# Clippy
SCOPE_STATUS=0
select_cargo_scope cargo.clippy "$FORCE_ALL" || SCOPE_STATUS=$?
if [[ "$SCOPE_STATUS" -gt 1 ]]; then
  exit "$SCOPE_STATUS"
fi
if [[ "$SCOPE_STATUS" -ne 0 ]]; then
  skip "Linting" "no affected targets"
else
  step "Linting"
  if ! cargo clippy "${CARGO_ARGS[@]:+${CARGO_ARGS[@]}}" --all-targets --all-features --locked >"$LOG_DIR/clippy.log" 2>&1; then
    fail
    show_error "$LOG_DIR/clippy.log"
    exit 1
  fi
  ok
fi

if work_required contracts.auxiliary; then
  step "Linting independent workspaces"
  if ! "$SCRIPT_DIR/lint-independent-workspaces.sh" >"$LOG_DIR/independent-lints.log" 2>&1; then
    fail
    show_error "$LOG_DIR/independent-lints.log"
    exit 1
  fi
  ok
fi

if [[ "${RSCRYPTO_SKIP_CHECK_SUPPLY_CHAIN:-}" != "1" ]] \
  && { work_required dependency-policy || work_required dependencies.auxiliary; }; then
  step "Auditing deps"
  if ! cargo deny --locked check all >"$LOG_DIR/deny.log" 2>&1; then
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
SCOPE_STATUS=0
select_cargo_scope cargo.doc "$FORCE_ALL" || SCOPE_STATUS=$?
if [[ "$SCOPE_STATUS" -gt 1 ]]; then
  exit "$SCOPE_STATUS"
fi
if [[ "$SCOPE_STATUS" -ne 0 ]]; then
  skip "Building docs" "no affected targets"
else
  step "Building docs"
  if ! cargo doc "${CARGO_ARGS[@]:+${CARGO_ARGS[@]}}" --no-deps --all-features --locked >"$LOG_DIR/doc.log" 2>&1; then
    fail
    show_error "$LOG_DIR/doc.log"
    exit 1
  fi
  ok
fi

echo "${GREEN}✓${RESET} Host checks passed"
