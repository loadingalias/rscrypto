#!/usr/bin/env bash
# Run affected repository-owned policy. Cargo work stays in check.sh/test.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

force_all=false
case "$#" in
  0) ;;
  1)
    [[ "$1" == --all ]] || { echo "Usage: $0 [--all]" >&2; exit 2; }
    force_all=true
    ;;
  *) echo "Usage: $0 [--all]" >&2; exit 2 ;;
esac

if [[ "$force_all" == false ]]; then
  rail_prime_plan
fi

work_required() {
  local status=0
  [[ "$force_all" == true ]] && return 0
  rail_work_required "$1" || status=$?
  [[ "$status" -le 1 ]] || exit "$status"
  return "$status"
}

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT
PYTHON="$("$SCRIPT_DIR/../lib/python.sh" --print)"

echo "Repository policy"

if work_required policy.actions; then
  step "Checking Actions policy"
  if ! "$SCRIPT_DIR/../ci/actions-policy.sh" >"$LOG_DIR/actions.log" 2>&1; then
    fail
    show_error "$LOG_DIR/actions.log"
    exit 1
  fi
  ok
else
  skip "Actions policy" "not required by Cargo Rail"
fi

if work_required contracts.cargo-graph; then
  step "Checking Cargo graph consistency"
  if ! cargo rail unify --check >"$LOG_DIR/cargo-graph.log" 2>&1; then
    fail
    show_error "$LOG_DIR/cargo-graph.log"
    exit 1
  fi
  ok
else
  skip "Cargo graph consistency" "not required by Cargo Rail"
fi

if work_required policy.repository; then
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
else
  skip "Repository policy" "not required by Cargo Rail"
fi

echo "${GREEN}✓${RESET} Repository policy passed"
