#!/usr/bin/env bash
set -euo pipefail

ROOT=""
if [[ ${1:-} == "--root" ]]; then
  ROOT=${2:?missing path after --root}
  shift 2
fi
if [[ $# -ne 0 ]]; then
  echo "usage: check-ci-ownership.sh [--root PATH]" >&2
  exit 2
fi

if [[ -z "$ROOT" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

WORKFLOWS="$ROOT/.github/workflows"
SUITE="$WORKFLOWS/_ci-suite.yaml"
WEEKLY="$WORKFLOWS/weekly.yaml"
MANIFEST="$ROOT/.config/target-matrix.json"
CROSS_SCRIPT="$ROOT/scripts/ci/cross-targets.sh"
COMPILE_MATRIX="$ROOT/scripts/check/check-feature-matrix.sh"
EXECUTABLE_MATRIX="$ROOT/scripts/test/test-feature-matrix.sh"
CHECK_ALL="$ROOT/scripts/check/check-all.sh"

fail() {
  echo "CI ownership error: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || fail "missing $1"
}

count_matches() {
  local pattern=$1
  shift
  local count
  count=$({ grep -ERho "$pattern" "$@" 2>/dev/null || true; } | wc -l | tr -d ' ')
  echo "$count"
}

count_feature_sets() {
  awk '
    /^FEATURE_SETS=\($/ { in_array = 1; next }
    in_array && /^\)$/ { exit }
    in_array && /^[[:space:]]+"/ { count += 1 }
    END { print count + 0 }
  ' "$1"
}

require_unique_feature_sets() {
  local duplicate
  duplicate=$(awk '
    /^FEATURE_SETS=\($/ { in_array = 1; next }
    in_array && /^\)$/ { exit }
    in_array && /^[[:space:]]+"/ {
      value = $0
      sub(/^[[:space:]]+"/, "", value)
      sub(/"$/, "", value)
      print value
    }
  ' "$1" | sort | uniq -d | head -1)
  [[ -z "$duplicate" ]] || fail "duplicate feature profile in $1: $duplicate"
}

require_file "$SUITE"
require_file "$WEEKLY"
require_file "$MANIFEST"
require_file "$CROSS_SCRIPT"
require_file "$COMPILE_MATRIX"
require_file "$EXECUTABLE_MATRIX"
require_file "$CHECK_ALL"

[[ $(count_feature_sets "$COMPILE_MATRIX") -eq 29 ]] \
  || fail "compile feature matrix must retain all 29 profiles"
[[ $(count_feature_sets "$EXECUTABLE_MATRIX") -eq 38 ]] \
  || fail "executable feature matrix must retain all 38 profiles"
require_unique_feature_sets "$COMPILE_MATRIX"
require_unique_feature_sets "$EXECUTABLE_MATRIX"

grep -Eq 'HOST_ARGS\+=\(--feature-matrix\)' "$CHECK_ALL" \
  || fail "local check-all must retain one explicit feature-matrix execution"

[[ $(count_matches 'just test-feature-matrix' "$WORKFLOWS") -eq 1 ]] \
  || fail "ordinary workflows must have exactly one executable feature-matrix owner"
[[ $(count_matches 'just check-feature-matrix' "$WORKFLOWS") -eq 1 ]] \
  || fail "ordinary workflows must have exactly one compile feature-matrix owner"

if grep -ERn 'just check --all|check-all\.sh' "$WORKFLOWS" >/dev/null; then
  fail "native workflows must not invoke comprehensive cross-target checks"
fi

if grep -En 'test-feature-matrix|check-feature-matrix' "$WEEKLY" >/dev/null; then
  fail "weekly must inherit feature contracts from the reusable suite"
fi

[[ $(count_matches 'scripts/ci/cross-targets\.sh' "$SUITE") -eq 1 ]] \
  || fail "the reusable suite must have exactly one cross-target owner"
[[ $(count_matches 'scripts/ci/native-check\.sh' "$SUITE") -eq 4 ]] \
  || fail "native validation must be owned by Linux, IBM Z, POWER10, and RISC-V job definitions"

ci_musl=$(jq '[.ci[] | select(.name | contains("musl"))] | length' "$MANIFEST")
[[ "$ci_musl" -eq 0 ]] || fail "MUSL targets must not masquerade as native host jobs"

ci_linux=$(jq '[.ci[] | select(.name | endswith("unknown-linux-gnu"))] | length' "$MANIFEST")
[[ "$ci_linux" -eq 2 ]] || fail "native CI must contain exactly x86_64 and AArch64 GNU hosts"

group_musl=$(jq '[.groups.linux[] | select(contains("musl"))] | length' "$MANIFEST")
[[ "$group_musl" -eq 2 ]] || fail "the target manifest must retain both MUSL triples"

[[ $(count_matches 'cargo (check|clippy|build) --target "\$target"' "$CROSS_SCRIPT") -ge 3 ]] \
  || fail "MUSL evidence must pass the target triple explicitly to Cargo"

echo "CI ownership contract passed"
