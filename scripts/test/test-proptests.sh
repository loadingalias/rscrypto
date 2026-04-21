#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

maybe_disable_sccache
apply_ci_resource_profile

export PROPTEST_CASES="${PROPTEST_CASES:-10000}"

echo "Running property and differential tests via Nextest..."
echo "PROPTEST_CASES=$PROPTEST_CASES"

HAS_NEXTEST=true
if ! command -v cargo-nextest >/dev/null 2>&1; then
  HAS_NEXTEST=false
  echo "cargo-nextest not found; falling back to cargo test"
fi

test_targets=()
while IFS= read -r path; do
  stem="$(basename "$path" .rs)"
  test_targets+=("$stem")
done < <(rg --files tests -g '*_proptest.rs' -g '*_properties.rs' -g '*_differential.rs' | sort)

if [ ${#test_targets[@]} -eq 0 ]; then
  echo "No property or differential integration tests found"
  exit 0
fi

echo "Selected integration test targets:"
printf '  %s\n' "${test_targets[@]}"

if [ "$HAS_NEXTEST" = true ]; then
  cmd=(cargo nextest run --workspace --all-features --lib --config-file .config/nextest.toml --test-threads 1)
  for target in "${test_targets[@]}"; do
    cmd+=(--test "$target")
  done
  "${cmd[@]}"
else
  cmd=(cargo test --workspace --all-features --lib)
  for target in "${test_targets[@]}"; do
    cmd+=(--test "$target")
  done
  "${cmd[@]}"
fi
