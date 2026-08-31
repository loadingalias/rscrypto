#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

usage() {
  echo "Usage: $0 [--all]" >&2
  exit 2
}

force_all=false
case "$#" in
  0) ;;
  1)
    [[ "$1" == --all ]] || usage
    force_all=true
    ;;
  *) usage ;;
esac

apply_ci_resource_profile

echo "Running unit, integration, property, and documentation tests..."
export RSCRYPTO_TEST_MODE=${RSCRYPTO_TEST_MODE:-${CARGO_RAIL_TEST_MODE:-local}}
export CARGO_RAIL_TEST_MODE=${CARGO_RAIL_TEST_MODE:-$RSCRYPTO_TEST_MODE}
echo "Test mode: $RSCRYPTO_TEST_MODE"

has_nextest=true
if ! command -v cargo-nextest >/dev/null 2>&1; then
  has_nextest=false
  echo "cargo-nextest not found; using cargo test"
fi

case "$RSCRYPTO_TEST_MODE" in
  commit) profile=commit ;;
  weekly) profile=weekly ;;
  *) profile=default ;;
esac
echo "Nextest profile: $profile"

nextest_thread_args=()
if [[ -n "${RSCRYPTO_TEST_THREADS:-}" ]]; then
  nextest_thread_args=(--test-threads "$RSCRYPTO_TEST_THREADS")
  echo "Test threads: $RSCRYPTO_TEST_THREADS"
fi

skip_doctests=false
case "${RSCRYPTO_SKIP_DOCTESTS:-}" in
  1 | true | TRUE | yes | YES)
    skip_doctests=true
    echo "Doctests disabled by RSCRYPTO_SKIP_DOCTESTS"
    ;;
esac

scope_status=0
select_cargo_scope cargo.test "$force_all" || scope_status=$?
if [[ "$scope_status" -gt 1 ]]; then
  exit "$scope_status"
fi

if [[ "$scope_status" -eq 0 ]]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Testing $SCOPE_DESC"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if [[ "$has_nextest" == true ]]; then
    cargo nextest run --locked "${CARGO_ARGS[@]}" -P "$profile" --all-features \
      --config-file .config/nextest.toml \
      "${nextest_thread_args[@]:+${nextest_thread_args[@]}}"
  else
    cargo test --locked "${CARGO_ARGS[@]}" --all-features --lib --tests
  fi
else
  echo "No unit or integration test targets selected by Cargo Rail"
fi

if [[ "$skip_doctests" == true ]]; then
  echo "Doctests skipped"
  exit 0
fi

scope_status=0
select_cargo_scope cargo.doctest "$force_all" || scope_status=$?
if [[ "$scope_status" -gt 1 ]]; then
  exit "$scope_status"
fi
if [[ "$scope_status" -eq 1 ]]; then
  echo "No doctest targets selected by Cargo Rail"
  exit 0
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running doctests for $SCOPE_DESC"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cargo test --locked "${CARGO_ARGS[@]}" --doc --all-features
