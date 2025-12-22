#!/usr/bin/env bash
set -euo pipefail

CRATE="${1:-}"

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
export CARGO_RAIL_TEST_MODE=${CARGO_RAIL_TEST_MODE:-local}
echo "Test mode: $CARGO_RAIL_TEST_MODE"

# Select nextest profile based on test mode
case "$CARGO_RAIL_TEST_MODE" in
  commit)
    PROFILE="commit"
    ;;
  local | *)
    PROFILE="default"
    ;;
esac

echo "Using nextest profile: $PROFILE"

if [ -n "$CRATE" ]; then
  # User specified a crate explicitly
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🎯 Running tests for specific crate: $CRATE"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  cargo nextest run -p "$CRATE" -P "$PROFILE" --all-features --config-file .config/nextest.toml
else
  # Full workspace test
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🔄 Running tests for entire workspace"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  cargo nextest run --workspace -P "$PROFILE" --all-features --config-file .config/nextest.toml
fi
