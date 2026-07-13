#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ci/dependabot-smoke.sh
source "$SCRIPT_DIR/dependabot-smoke.sh"

assert_plan() {
  local expected_root=$1
  local expected_automation=$2
  local expected_manifests=$3
  shift 3

  CHANGED=("$@")
  classify_changed_files

  [[ "$RUN_ROOT" == "$expected_root" ]]
  [[ "$RUN_AUTOMATION" == "$expected_automation" ]]
  [[ "${#MANIFESTS[@]}" -eq "$expected_manifests" ]]
  [[ "${#UNSUPPORTED[@]}" -eq 0 ]]
}

assert_plan true false 0 Cargo.toml Cargo.lock
assert_plan false false 1 fuzz/Cargo.toml fuzz/Cargo.lock
assert_plan false false 1 fuzz-packages/fast-rapidhash/Cargo.toml fuzz-packages/fast-rapidhash/Cargo.lock
assert_plan false false 1 tools/ct-dudect/Cargo.toml tools/ct-dudect/Cargo.lock
assert_plan false true 0 .github/workflows/ci.yaml
assert_plan true true 0 Cargo.lock .github/workflows/ci.yaml .github/actions-lock.yaml

CHANGED=(src/lib.rs)
classify_changed_files
[[ "${#UNSUPPORTED[@]}" -eq 1 ]]

echo "Dependabot smoke routing tests passed"
