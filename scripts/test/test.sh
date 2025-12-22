#!/usr/bin/env bash
set -euo pipefail

maybe_disable_sccache() {
  if [[ -n "${RUSTC_WRAPPER:-}" && "${RUSTC_WRAPPER##*/}" == "sccache" ]]; then
    if ! "$RUSTC_WRAPPER" rustc -vV >/dev/null 2>&1; then
      echo "⚠️  WARNING: sccache is configured but not usable; disabling RUSTC_WRAPPER for this run."
      export RUSTC_WRAPPER=
    fi
  fi
}

maybe_disable_sccache

echo "Running Unit, Integration, and Property Tests via Nextest..."
export RSCRYPTO_TEST_MODE=${RSCRYPTO_TEST_MODE:-${CARGO_RAIL_TEST_MODE:-local}}
export CARGO_RAIL_TEST_MODE=${CARGO_RAIL_TEST_MODE:-$RSCRYPTO_TEST_MODE}
echo "Test mode: $RSCRYPTO_TEST_MODE"

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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Priority:
#   1. --all flag: Force full workspace
#   2. User-specified crate(s): Test those directly
#   3. Otherwise: cargo rail change detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALL_FLAG=false
CRATES=()
for arg in "$@"; do
  if [ "$arg" = "--all" ]; then
    ALL_FLAG=true
  else
    CRATES+=("$arg")
  fi
done

if [ ${#CRATES[@]} -gt 0 ]; then
  CRATE_FLAGS=""
  for crate in "${CRATES[@]}"; do
    CRATE_FLAGS="$CRATE_FLAGS -p $crate"
  done
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Testing specific crate(s): ${CRATES[*]}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  # shellcheck disable=SC2086
  cargo nextest run $CRATE_FLAGS -P "$PROFILE" --all-features --config-file .config/nextest.toml
elif [ "$ALL_FLAG" = true ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Testing entire workspace (--all)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  cargo nextest run --workspace -P "$PROFILE" --all-features --config-file .config/nextest.toml
else
  SINCE_ARG=""
  if [ -n "${RAIL_SINCE:-}" ]; then
    SINCE_ARG="--since $RAIL_SINCE"
    echo "Using base ref from CI: $RAIL_SINCE"
  fi
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Running tests for affected crates (cargo rail change detection)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  # shellcheck disable=SC2086
  cargo rail test $SINCE_ARG -- -P "$PROFILE" --all-features --config-file .config/nextest.toml
fi
