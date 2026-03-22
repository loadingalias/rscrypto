#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

maybe_disable_sccache

echo "Running Unit, Integration, and Property Tests via Nextest..."
export RSCRYPTO_TEST_MODE=${RSCRYPTO_TEST_MODE:-${CARGO_RAIL_TEST_MODE:-local}}
export CARGO_RAIL_TEST_MODE=${CARGO_RAIL_TEST_MODE:-$RSCRYPTO_TEST_MODE}
echo "Test mode: $RSCRYPTO_TEST_MODE"

HAS_NEXTEST=true
if ! command -v cargo-nextest >/dev/null 2>&1; then
  HAS_NEXTEST=false
  echo "cargo-nextest not found; falling back to cargo test"
fi

# Select nextest profile based on test mode
case "$RSCRYPTO_TEST_MODE" in
  commit)
    PROFILE="commit"
    ;;
  local | *)
    PROFILE="default"
    ;;
esac

echo "Using nextest profile: $PROFILE"

run_workspace_doctests() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Running doctests for entire workspace"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  cargo test --workspace --doc --all-features
}

run_crate_doctests() {
  local crates=("$@")
  if [ ${#crates[@]} -eq 0 ]; then
    echo "no doc-test targets"
    return 0
  fi

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Running doctests for crate(s): ${crates[*]}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  for crate in "${crates[@]}"; do
    cargo test -p "$crate" --doc --all-features
  done
}

run_changed_doctests() {
  local affected
  affected="$(rail_plan_crates)"
  if [ -z "$affected" ]; then
    echo "no doc-test targets"
    return 0
  fi

  local crates=()
  local crate
  for crate in $affected; do
    crates+=("$crate")
  done
  run_crate_doctests "${crates[@]}"
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Priority:
#   1. --changed flag: cargo rail change detection
#   2. User-specified crate(s): Test those directly
#   3. Otherwise (--all or no args): full workspace
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALL_FLAG=false
CHANGED_FLAG=false
CRATES=()
for arg in "$@"; do
  case "$arg" in
    --all)
      ALL_FLAG=true
      ;;
    --changed)
      CHANGED_FLAG=true
      ;;
    *)
      CRATES+=("$arg")
      ;;
  esac
done

if [ "$CHANGED_FLAG" = true ] && { [ "$ALL_FLAG" = true ] || [ ${#CRATES[@]} -gt 0 ]; }; then
  echo "error: --changed cannot be combined with --all or crate arguments" >&2
  exit 2
fi

if [ "$CHANGED_FLAG" = true ]; then
  RAIL_SCOPE_ARGS=()
  if [ -n "${RAIL_SINCE:-}" ]; then
    RAIL_SCOPE_ARGS=(--since "$RAIL_SINCE")
    echo "Using base ref from CI: $RAIL_SINCE"
  else
    RAIL_SCOPE_ARGS=(--merge-base)
  fi
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Running tests for changed crates (cargo rail run)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if [ "$HAS_NEXTEST" = true ]; then
    cargo rail run "${RAIL_SCOPE_ARGS[@]}" --surface test -- -P "$PROFILE" --all-features --config-file .config/nextest.toml
    run_changed_doctests
  else
    affected="$(rail_plan_crates)"
    if [ -z "$affected" ]; then
      echo "no test targets"
    else
      for crate in $affected; do
        cargo test -p "$crate" --all-features
      done
    fi
  fi
elif [ ${#CRATES[@]} -gt 0 ]; then
  CRATE_FLAGS=""
  for crate in "${CRATES[@]}"; do
    CRATE_FLAGS="$CRATE_FLAGS -p $crate"
  done
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Testing specific crate(s): ${CRATES[*]}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if [ "$HAS_NEXTEST" = true ]; then
    # shellcheck disable=SC2086
    cargo nextest run $CRATE_FLAGS -P "$PROFILE" --all-features --config-file .config/nextest.toml
    run_crate_doctests "${CRATES[@]}"
  else
    # shellcheck disable=SC2086
    cargo test $CRATE_FLAGS --all-features
  fi
else
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Testing entire workspace"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if [ "$HAS_NEXTEST" = true ]; then
    cargo nextest run --workspace -P "$PROFILE" --all-features --config-file .config/nextest.toml
    run_workspace_doctests
  else
    cargo test --workspace --all-features
  fi
fi
