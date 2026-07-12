#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

ALL_TARGETS=false
for arg in "$@"; do
  case "$arg" in
    --all-targets) ALL_TARGETS=true ;;
    *)
      echo "usage: native-check.sh [--all-targets]" >&2
      exit 2
      ;;
  esac
done

maybe_disable_sccache
apply_ci_resource_profile

echo "Native validation: $(rustc -vV | sed -n 's/^host: //p')"

echo ""
echo "Checking no-default-features library boundary..."
cargo check --workspace --lib --no-default-features

TARGET_ARGS=(--lib)
if [[ "$ALL_TARGETS" == true ]]; then
  TARGET_ARGS=(--all-targets)
fi

echo ""
echo "Checking all-feature native targets..."
cargo check --workspace "${TARGET_ARGS[@]}" --all-features

echo ""
echo "Linting all-feature native targets..."
cargo clippy --workspace "${TARGET_ARGS[@]}" --all-features -- -D warnings

if [[ "$ALL_TARGETS" == true ]]; then
  echo ""
  echo "Building all-feature native targets..."
  cargo build --workspace --all-targets --all-features
fi

echo "Native validation passed"
