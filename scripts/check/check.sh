#!/usr/bin/env bash
set -euo pipefail

# Host-only checks: fmt, check, opt-in feature matrices, clippy, optional
# deny/audit, and docs.
# Usage: check.sh [--all] [--feature-matrix]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

RUN_FEATURE_MATRIX=false
FORCE_ALL=false
for arg in "$@"; do
  case "$arg" in
    --feature-matrix)
      RUN_FEATURE_MATRIX=true
      ;;
    --all)
      FORCE_ALL=true
      ;;
    *)
      echo "Usage: $0 [--all] [--feature-matrix]" >&2
      exit 2
      ;;
  esac
done

SCOPE_STATUS=0
select_cargo_scope cargo.build "$FORCE_ALL" || SCOPE_STATUS=$?
if [[ "$SCOPE_STATUS" -gt 1 ]]; then
  exit "$SCOPE_STATUS"
fi
CARGO_SELECTED=true
[[ "$SCOPE_STATUS" -eq 0 ]] || CARGO_SELECTED=false

# Determine if full workspace (for audit/deny)
FULL_WORKSPACE=false
if [[ "$CARGO_SELECTED" == true && "$CARGO_SCOPE_KIND" == workspace ]]; then
  FULL_WORKSPACE=true
fi

CHECK_RSCRYPTO_FEATURE_MATRIX=false
if [[ "$CARGO_SELECTED" == true && "$RUN_FEATURE_MATRIX" == true ]]; then
  CHECK_RSCRYPTO_FEATURE_MATRIX=true
fi

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT
PYTHON="$("$SCRIPT_DIR/../lib/python.sh" --print)"

echo "Host checks ${DIM}($SCOPE_DESC)${RESET}"

# Format
step "Formatting"
if ! cargo fmt --all -- --check >"$LOG_DIR/fmt.log" 2>&1; then
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

step "Checking hash vector provenance"
if ! "$PYTHON" "$SCRIPT_DIR/hash-vector-provenance.py" >"$LOG_DIR/hash-vectors.log" 2>&1; then
  fail
  show_error "$LOG_DIR/hash-vectors.log"
  exit 1
fi
ok

step "Checking authentication vector provenance"
if ! "$PYTHON" "$SCRIPT_DIR/auth-vector-provenance.py" >"$LOG_DIR/auth-vectors.log" 2>&1; then
  fail
  show_error "$LOG_DIR/auth-vectors.log"
  exit 1
fi
ok

step "Checking feature boundaries"
if ! "$PYTHON" "$SCRIPT_DIR/feature-boundaries.py" >"$LOG_DIR/feature-boundaries.log" 2>&1; then
  fail
  show_error "$LOG_DIR/feature-boundaries.log"
  exit 1
fi
ok

step "Checking benchmark catalog"
if ! "$PYTHON" "$SCRIPT_DIR/../bench/benchmark_catalog_test.py" >"$LOG_DIR/benchmark-catalog.log" 2>&1; then
  fail
  show_error "$LOG_DIR/benchmark-catalog.log"
  exit 1
fi
ok

step "Checking CT assembly scanner"
if ! "$PYTHON" "$SCRIPT_DIR/../ct/asm_heuristics_test.py" >"$LOG_DIR/ct-asm-scanner.log" 2>&1; then
  fail
  show_error "$LOG_DIR/ct-asm-scanner.log"
  exit 1
fi
ok

step "Checking DudeCT evidence parsing"
if ! "$PYTHON" "$SCRIPT_DIR/../ct/dudect_report_test.py" >"$LOG_DIR/ct-dudect-report.log" 2>&1; then
  fail
  show_error "$LOG_DIR/ct-dudect-report.log"
  exit 1
fi
ok

step "Checking CT evidence validation"
if ! "$PYTHON" "$SCRIPT_DIR/../ct/evidence_validation_test.py" >"$LOG_DIR/ct-evidence-validation.log" 2>&1; then
  fail
  show_error "$LOG_DIR/ct-evidence-validation.log"
  exit 1
fi
ok

# Check
if [[ "$CARGO_SELECTED" == false ]]; then
  skip "Checking" "no affected targets"
else
  step "Checking"
  if ! cargo check "${CARGO_ARGS[@]}" --all-targets --all-features --locked >"$LOG_DIR/check.log" 2>&1; then
    fail
    show_error "$LOG_DIR/check.log"
    exit 1
  fi
  ok
fi

if [[ "$CHECK_RSCRYPTO_FEATURE_MATRIX" == true ]]; then
  step "Checking rscrypto no_std matrix"
  if ! "$SCRIPT_DIR/check-feature-matrix.sh" >>"$LOG_DIR/check.log" 2>&1; then
    fail
    show_error "$LOG_DIR/check.log"
    exit 1
  fi
  ok

  step "Testing rscrypto feature matrix"
  if ! "$SCRIPT_DIR/../test/test-feature-matrix.sh" >>"$LOG_DIR/check.log" 2>&1; then
    fail
    show_error "$LOG_DIR/check.log"
    exit 1
  fi
  ok
elif [[ "$CARGO_SELECTED" == true ]]; then
  skip "rscrypto feature matrix" "disabled for this check profile"
fi

# Clippy
if [[ "$CARGO_SELECTED" == false ]]; then
  skip "Linting" "no affected targets"
else
  step "Linting"
  if ! cargo clippy "${CARGO_ARGS[@]}" --all-targets --all-features --locked >"$LOG_DIR/clippy.log" 2>&1; then
    fail
    show_error "$LOG_DIR/clippy.log"
    exit 1
  fi
  ok
fi

if [[ "$FULL_WORKSPACE" == true ]]; then
  step "Linting independent workspaces"
  if ! "$SCRIPT_DIR/lint-independent-workspaces.sh" >"$LOG_DIR/independent-lints.log" 2>&1; then
    fail
    show_error "$LOG_DIR/independent-lints.log"
    exit 1
  fi
  ok
fi

# Audit/Deny (workspace only). CI owns this in the dedicated supply-chain lane.
if [[ "$FULL_WORKSPACE" == true && "${RSCRYPTO_SKIP_CHECK_SUPPLY_CHAIN:-}" != "1" ]]; then
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
if [[ "$CARGO_SELECTED" == false ]]; then
  skip "Building docs" "no affected targets"
else
  step "Building docs"
  if ! cargo doc "${CARGO_ARGS[@]}" --no-deps --all-features --locked >"$LOG_DIR/doc.log" 2>&1; then
    fail
    show_error "$LOG_DIR/doc.log"
    exit 1
  fi
  ok
fi

echo "${GREEN}✓${RESET} Host checks passed"
