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
RELEASE="$WORKFLOWS/release.yaml"
RSA="$WORKFLOWS/rsa.yaml"
MANIFEST="$ROOT/.config/target-matrix.json"
CROSS_SCRIPT="$ROOT/scripts/ci/cross-targets.sh"
COMPILE_MATRIX="$ROOT/scripts/check/check-feature-matrix.sh"
EXECUTABLE_MATRIX="$ROOT/scripts/test/test-feature-matrix.sh"
CHECK_ALL="$ROOT/scripts/check/check-all.sh"
CI_CHECK="$ROOT/scripts/ci/ci-check.sh"
RELEASE_PREFLIGHT="$ROOT/scripts/ci/release-preflight.sh"
RELEASE_CI="$ROOT/scripts/ci/release-ci-check.sh"
RELEASE_EVIDENCE="$ROOT/scripts/ci/release-evidence-check.sh"
DEPENDABOT="$ROOT/.github/dependabot.yaml"

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
require_file "$RELEASE"
require_file "$RSA"
require_file "$MANIFEST"
require_file "$CROSS_SCRIPT"
require_file "$COMPILE_MATRIX"
require_file "$EXECUTABLE_MATRIX"
require_file "$CHECK_ALL"
require_file "$CI_CHECK"
require_file "$RELEASE_PREFLIGHT"
require_file "$RELEASE_CI"
require_file "$RELEASE_EVIDENCE"
require_file "$DEPENDABOT"

[[ $(yq eval '.version' "$DEPENDABOT") == "2" ]] || fail "Dependabot config must use version 2"
[[ $(yq eval '[.updates[] | select(."package-ecosystem" == "cargo")] | length' "$DEPENDABOT") == "1" ]] \
  || fail "Dependabot must have exactly one non-overlapping Cargo update entry"
[[ $(yq eval '.updates[] | select(."package-ecosystem" == "cargo") | .directories | sort | join(",")' "$DEPENDABOT") \
  == "/,/fuzz,/fuzz-packages/*,/tools/*" ]] \
  || fail "Dependabot Cargo coverage must include root, fuzz, scoped fuzz packages, and standalone tools"
[[ $(yq eval '.updates[] | select(."package-ecosystem" == "cargo") | ."open-pull-requests-limit"' "$DEPENDABOT") == "0" ]] \
  || fail "routine Cargo version PRs must stay disabled in favor of coordinated updates"
if grep -En 'group-by:[[:space:]]*dependency-name' "$DEPENDABOT" >/dev/null; then
  fail "Dependabot must not use the upstream-broken cross-directory dependency-name grouping"
fi

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
# shellcheck disable=SC2016 # `$crate` is an intentional literal in the release-preflight contract regex.
[[ $(count_matches 'cargo semver-checks --package "\$crate" --all-features' "$RELEASE_PREFLIGHT") -eq 1 ]] \
  || fail "tag preflight must have exactly one final-version SemVer owner"
if grep -ERn 'cargo semver-checks' "$WORKFLOWS" >/dev/null; then
  fail "ordinary workflows must leave version-aware SemVer analysis to cargo-rail release planning"
fi

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
[[ $(count_matches 'cargo rail unify --check' "$SUITE") -eq 1 ]] \
  || fail "the reusable suite must have exactly one Cargo graph assurance owner"
if grep -En 'cargo rail unify --check' "$RELEASE_PREFLIGHT" >/dev/null; then
  fail "tag preflight must consume exact-commit CI graph assurance instead of repeating it"
fi
grep -Fq 'scripts/ci/release-ci-check.sh --commit "$GITHUB_SHA" --wait' "$RELEASE" \
  || fail "release CI Gate must use the shared exact-commit checker"
grep -Fq 'CI Suite / Cargo Graph Assurance / run' "$RELEASE_CI" \
  || fail "release CI Gate must verify exact-commit Cargo Graph Assurance"
[[ $(yq eval '.concurrency.group' "$RSA") == 'rsa-${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}' ]] \
  || fail "reusable RSA workflow concurrency must not collide with its caller"
[[ $(yq eval '.jobs.ct.with.upload_raw_artifacts' "$WEEKLY") == "true" ]] \
  || fail "Weekly must preserve raw CT artifacts for exact-commit release promotion"
grep -Fq 'scripts/ci/release-evidence-check.sh --commit "$GITHUB_SHA"' "$RELEASE" \
  || fail "release must require exact or release-equivalent Weekly CT and RSA evidence"
grep -Fq 'run-id: ${{ needs.evidence-gate.outputs.weekly_run_id }}' "$RELEASE" \
  || fail "release must consume CT artifacts from the validated Weekly run"
if grep -Eq 'uses: ./\.github/workflows/(ct|rsa)\.yaml' "$RELEASE"; then
  fail "tag workflow must promote exact-commit evidence instead of rerunning CT or RSA"
fi
if grep -n 'check-unify' "$CI_CHECK" >/dev/null; then
  fail "the fast quality lane must not own exhaustive Cargo graph assurance"
fi

ci_musl=$(jq '[.ci[] | select(.name | contains("musl"))] | length' "$MANIFEST")
[[ "$ci_musl" -eq 0 ]] || fail "MUSL targets must not masquerade as native host jobs"

ci_linux=$(jq '[.ci[] | select(.name | endswith("unknown-linux-gnu"))] | length' "$MANIFEST")
[[ "$ci_linux" -eq 2 ]] || fail "native CI must contain exactly x86_64 and AArch64 GNU hosts"

group_musl=$(jq '[.groups.linux[] | select(contains("musl"))] | length' "$MANIFEST")
[[ "$group_musl" -eq 2 ]] || fail "the target manifest must retain both MUSL triples"

# shellcheck disable=SC2016 # `$target` is an intentional literal in the workflow contract regex.
[[ $(count_matches 'cargo (check|clippy|build) --target "\$target"' "$CROSS_SCRIPT") -ge 3 ]] \
  || fail "MUSL evidence must pass the target triple explicitly to Cargo"

echo "CI ownership contract passed"
