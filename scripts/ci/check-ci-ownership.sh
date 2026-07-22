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
CI="$WORKFLOWS/ci.yaml"
SUITE="$WORKFLOWS/_ci-suite.yaml"
RUST_JOB="$WORKFLOWS/_rust-job.yaml"
WEEKLY="$WORKFLOWS/weekly.yaml"
RISCV="$WORKFLOWS/riscv.yaml"
RELEASE="$WORKFLOWS/release.yaml"
RSA="$WORKFLOWS/rsa.yaml"
MANIFEST="$ROOT/.config/target-matrix.json"
CROSS_SCRIPT="$ROOT/scripts/ci/cross-targets.sh"
COMPILE_MATRIX="$ROOT/scripts/check/check-feature-matrix.sh"
EXECUTABLE_MATRIX="$ROOT/scripts/test/test-feature-matrix.sh"
CHECK_ALL="$ROOT/scripts/check/check-all.sh"
CI_CHECK="$ROOT/scripts/ci/ci-check.sh"
RUN_RUST_JOB="$ROOT/scripts/ci/run-rust-job.sh"
RAIL_PLAN_RESOLVER="$ROOT/scripts/ci/resolve-rail-plan.sh"
RELEASE_PREFLIGHT="$ROOT/scripts/ci/release-preflight.sh"
RELEASE_EVIDENCE="$ROOT/scripts/ci/release-evidence-check.sh"
RELEASE_SOURCE="$ROOT/scripts/ci/package-release-source.sh"
RELEASE_MANIFEST="$ROOT/scripts/ci/write-release-manifest.sh"
RELEASE_IDENTITY_TEST="$ROOT/scripts/ci/release-identity-test.sh"
PUBLISH_RELEASE="$ROOT/scripts/ci/publish-immutable-release.sh"
PUBLISH_RELEASE_TEST="$ROOT/scripts/ci/publish-immutable-release-test.sh"
REPOSITORY_CONTROLS="$ROOT/scripts/ci/repository-controls-evidence.sh"
REPOSITORY_CONTROLS_TEST="$ROOT/scripts/ci/repository-controls-evidence-test.sh"
REPOSITORY_POLICY="$ROOT/.github/rulesets/protect-main.json"
RELEASE_TAG_POLICY="$ROOT/.github/rulesets/protect-release-tags.json"
RELEASE_IMMUTABILITY_POLICY="$ROOT/.github/repository-settings/release-immutability.json"
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

require_file "$CI"
require_file "$SUITE"
require_file "$RUST_JOB"
require_file "$WEEKLY"
require_file "$RISCV"
require_file "$RELEASE"
require_file "$RSA"
require_file "$MANIFEST"
require_file "$CROSS_SCRIPT"
require_file "$COMPILE_MATRIX"
require_file "$EXECUTABLE_MATRIX"
require_file "$CHECK_ALL"
require_file "$CI_CHECK"
require_file "$RUN_RUST_JOB"
require_file "$RAIL_PLAN_RESOLVER"
require_file "$RELEASE_PREFLIGHT"
require_file "$RELEASE_EVIDENCE"
require_file "$RELEASE_SOURCE"
require_file "$RELEASE_MANIFEST"
require_file "$RELEASE_IDENTITY_TEST"
require_file "$PUBLISH_RELEASE"
require_file "$PUBLISH_RELEASE_TEST"
require_file "$REPOSITORY_CONTROLS"
require_file "$REPOSITORY_CONTROLS_TEST"
require_file "$REPOSITORY_POLICY"
require_file "$RELEASE_TAG_POLICY"
require_file "$RELEASE_IMMUTABILITY_POLICY"
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
[[ $(yq eval '[.updates[] | select(."package-ecosystem" == "github-actions")] | length' "$DEPENDABOT") == "1" ]] \
  || fail "Dependabot must have exactly one GitHub Actions update entry"
[[ $(yq eval '.updates[] | select(."package-ecosystem" == "github-actions") | .directory' "$DEPENDABOT") == "/" ]] \
  || fail "Dependabot must scan all repository workflows"
[[ $(yq eval '.updates[] | select(."package-ecosystem" == "github-actions") | ."open-pull-requests-limit"' "$DEPENDABOT") == "1" ]] \
  || fail "Dependabot must limit GitHub Actions updates to one open pull request"

[[ $(yq eval '.on.push // "missing"' "$CI") == "missing" ]] \
  || fail "ordinary CI must not repeat the PR suite after merge"
[[ $(yq eval '[.on.pull_request.types[]] | sort | join(",")' "$CI") \
  == "opened,ready_for_review,reopened,synchronize" ]] \
  || fail "CI must run when a ready pull request is opened, updated, reopened, or leaves draft"
[[ $(yq eval '[.jobs[] | select((.if // "") | contains("pull_request.draft"))] | length' "$CI") \
  == $(yq eval '.jobs | length' "$CI") ]] \
  || fail "every CI job must defer draft pull requests"

[[ $(yq eval '.jobs."rail-plan".steps[] | select(.id == "resolve") | .if' "$CI") == "always()" ]] \
  || fail "CI must resolve a conservative plan after planner failure"
suite_condition=$(yq eval '.jobs.suite."if"' "$CI")
[[ "$suite_condition" == *"always()"* && "$suite_condition" == *"needs.rail-plan.result != 'success'"* ]] \
  || fail "planner job failure must still run the CI suite"
# shellcheck disable=SC2016 # GitHub expressions are intentional literal workflow contracts.
grep -Fq 'plan_valid: ${{ steps.resolve.outputs.valid }}' "$CI" \
  || fail "CI must expose validated planner state"
# shellcheck disable=SC2016 # GitHub expressions are intentional literal workflow contracts.
grep -Fq 'plan_empty: ${{ steps.resolve.outputs.empty }}' "$CI" \
  || fail "CI must expose explicit empty planner state"
grep -Fq 'if [[ "$PLAN_VALID" != "true" || "$PLAN_EMPTY" != "true" ]]' "$CI" \
  || fail "Complete must reject an unvalidated suite skip"
[[ $(count_matches 'scripts/ci/resolve-rail-plan\.sh' "$CI") -eq 1 ]] \
  || fail "CI must use exactly one repository-owned plan resolver"

if grep -ERn '^[[:space:]]+(pre_script|run_script):' "$WORKFLOWS" >/dev/null; then
  fail "reusable workflows must not accept executable shell fragments"
fi
rust_job_calls=$(count_matches 'uses:[[:space:]]+\./\.github/workflows/_rust-job\.yaml' "$WORKFLOWS")
rust_job_operations=$(count_matches '^[[:space:]]+operation:[[:space:]]+[-[:alnum:]]+[[:space:]]*$' "$WORKFLOWS")
[[ "$rust_job_calls" -eq "$rust_job_operations" ]] \
  || fail "every reusable Rust job caller must select exactly one typed operation"
while IFS= read -r operation; do
  [[ -n "$operation" ]] || continue
  grep -Eq "^[[:space:]]+$operation\\)" "$RUN_RUST_JOB" \
    || fail "reusable Rust job caller selects unsupported operation: $operation"
done < <(
  awk '/^[[:space:]]+operation:[[:space:]]+[-[:alnum:]]+[[:space:]]*$/ { print $2 }' \
    "$WORKFLOWS"/*.yaml | sort -u
)
[[ $(yq eval '.on.workflow_call.inputs.operation.required' "$RUST_JOB") == "true" ]] \
  || fail "the reusable Rust job operation must be required"
[[ $(yq eval '.on.workflow_call.inputs.operation.type' "$RUST_JOB") == "string" ]] \
  || fail "the reusable Rust job operation must be typed as a string"
[[ $(yq eval '[.jobs.run.steps[] | select(has("run"))] | length' "$RUST_JOB") -eq 1 ]] \
  || fail "the reusable Rust job must expose one fixed command step"
[[ $(yq eval '.jobs.run.steps[] | select(has("run")) | .run' "$RUST_JOB") == "scripts/ci/run-rust-job.sh" ]] \
  || fail "the reusable Rust job must invoke the repository-owned dispatcher"
# shellcheck disable=SC2016 # GitHub expression is an intentional literal workflow contract.
grep -Fq 'RSCRYPTO_CI_OPERATION: ${{ inputs.operation }}' "$RUST_JOB" \
  || fail "the reusable Rust job must pass its operation as environment data"
if yq eval '.. | select(tag == "!!map" and has("run") and (.run | tag == "!!str")) | .run' \
  "$WORKFLOWS"/*.yaml | grep -Eq '\$\{\{[[:space:]]*inputs\.'; then
  fail "workflow inputs must not be interpolated into shell programs"
fi
if grep -En '(^|[[:space:]])eval[[:space:]]|(^|[[:space:]])(bash|sh)[[:space:]]+-c|<<<' \
  "$RUN_RUST_JOB" >/dev/null; then
  fail "the Rust job dispatcher must not invoke a dynamic shell interpreter"
fi

[[ $(count_feature_sets "$COMPILE_MATRIX") -eq 29 ]] \
  || fail "compile feature matrix must retain all 29 profiles"
[[ $(count_feature_sets "$EXECUTABLE_MATRIX") -eq 38 ]] \
  || fail "executable feature matrix must retain all 38 profiles"
require_unique_feature_sets "$COMPILE_MATRIX"
require_unique_feature_sets "$EXECUTABLE_MATRIX"

grep -Eq 'HOST_ARGS\+=\(--feature-matrix\)' "$CHECK_ALL" \
  || fail "local check-all must retain one explicit feature-matrix execution"

[[ $(count_matches 'just test-feature-matrix' "$WORKFLOWS" "$RUN_RUST_JOB") -eq 1 ]] \
  || fail "ordinary workflows must have exactly one executable feature-matrix owner"
[[ $(count_matches 'just check-feature-matrix' "$WORKFLOWS" "$RUN_RUST_JOB") -eq 1 ]] \
  || fail "ordinary workflows must have exactly one compile feature-matrix owner"
# shellcheck disable=SC2016 # `$crate` is an intentional literal in the release-preflight contract regex.
[[ $(count_matches 'cargo semver-checks --package "\$crate" --all-features' "$RELEASE_PREFLIGHT") -eq 1 ]] \
  || fail "tag preflight must have exactly one final-version SemVer owner"
if grep -ERn 'cargo semver-checks' "$WORKFLOWS" >/dev/null; then
  fail "ordinary workflows must leave version-aware SemVer analysis to cargo-rail release planning"
fi

if grep -ERn 'just check --all|check-all\.sh' "$WORKFLOWS" "$RUN_RUST_JOB" >/dev/null; then
  fail "native workflows must not invoke comprehensive cross-target checks"
fi

if grep -En 'test-feature-matrix|check-feature-matrix' "$WEEKLY" >/dev/null; then
  fail "weekly must inherit feature contracts from the reusable suite"
fi

[[ $(count_matches 'operation:[[:space:]]+cross-targets' "$SUITE") -eq 1 ]] \
  || fail "the reusable suite must have exactly one cross-target owner"
[[ $(count_matches 'scripts/ci/cross-targets\.sh' "$RUN_RUST_JOB") -eq 1 ]] \
  || fail "the Rust job dispatcher must define exactly one cross-target operation"
[[ $(count_matches 'operation:[[:space:]]+native$' "$SUITE") -eq 1 ]] \
  || fail "the reusable suite must own exactly one target-matrix native operation"
[[ $(count_matches 'operation:[[:space:]]+native-ibm' "$SUITE") -eq 2 ]] \
  || fail "the reusable suite must own only Linux, IBM Z, and POWER10 native validation"
[[ $(count_matches 'operation:[[:space:]]+native-riscv' "$RISCV") -eq 1 ]] \
  || fail "the RISC-V workflow must own exactly one native validation lane"
[[ $(count_matches 'scripts/ci/native-check\.sh' "$RUN_RUST_JOB") -eq 3 ]] \
  || fail "the Rust job dispatcher must retain Linux, IBM, and RISC-V native operations"
if grep -Ein 'riscv' "$SUITE" >/dev/null; then
  fail "the reusable CI suite must not own physical RISC-V work"
fi
if grep -Ein 'riscv' "$WEEKLY" >/dev/null; then
  fail "Weekly must not own physical RISC-V work"
fi
if grep -Ein 'riscv' "$WORKFLOWS/bench.yaml" >/dev/null; then
  fail "the generic benchmark workflow must not expose RISC-V"
fi
[[ $(yq eval '.jobs.native.with.runner' "$RISCV") == "ubuntu-24.04-riscv" ]] \
  || fail "the RISC-V workflow must own the RISE native runner"
[[ $(yq eval '.jobs.ct.with.platforms' "$RISCV") == "rise-riscv" ]] \
  || fail "the RISC-V workflow must select only the RISE CT lane"
[[ $(yq eval '.jobs.ct.with.upload_raw_artifacts' "$RISCV") == "true" ]] \
  || fail "the RISC-V workflow must retain raw CT artifacts for release promotion"
[[ $(count_matches 'operation:[[:space:]]+cargo-graph' "$SUITE") -eq 1 ]] \
  || fail "the reusable suite must have exactly one Cargo graph assurance owner"
[[ $(count_matches 'cargo rail unify --check' "$RUN_RUST_JOB") -eq 1 ]] \
  || fail "the Rust job dispatcher must define exactly one Cargo graph assurance operation"
if grep -En 'cargo rail unify --check' "$RELEASE_PREFLIGHT" >/dev/null; then
  fail "tag preflight must consume exact-commit Weekly graph assurance instead of repeating it"
fi
grep -Fq 'CI Suite (weekly) / Cargo Graph Assurance / run' "$RELEASE_EVIDENCE" \
  || fail "release evidence must require exact-commit Weekly Cargo Graph Assurance"
[[ $(yq eval '.concurrency.group' "$RSA") == 'rsa-${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}' ]] \
  || fail "reusable RSA workflow concurrency must not collide with its caller"
[[ $(yq eval '.jobs.ct.with.upload_raw_artifacts' "$WEEKLY") == "true" ]] \
  || fail "Weekly must preserve raw CT artifacts for exact-commit release promotion"
grep -Fq 'scripts/ci/release-evidence-check.sh --commit "$GITHUB_SHA"' "$RELEASE" \
  || fail "release must require paired Weekly and RISC-V evidence from one valid commit"
grep -Fq 'scripts/ci/repository-controls-evidence.sh' "$RELEASE" \
  || fail "release must capture the live repository controls"
grep -Fq 'scripts/ci/package-release-source.sh' "$RELEASE_PREFLIGHT" \
  || fail "release preflight must build the exact-commit source archive"
grep -Fq 'scripts/ci/write-release-manifest.sh' "$RELEASE" \
  || fail "release must bind artifacts and toolchain metadata in one identity manifest"
grep -Fq -- '--allow-redacted-bypass' "$RELEASE" \
  || fail "release must explicitly acknowledge GitHub's workflow-token bypass redaction"
# shellcheck disable=SC2016 # GitHub expression is an intentional literal workflow contract.
grep -Fq 'subject-path: ${{ steps.repository_controls.outputs.evidence_path }}' "$RELEASE" \
  || fail "release must attest the repository controls evidence"
grep -Fq 'REPOSITORY_CONTROLS_SHA256' "$RELEASE" \
  || fail "release must checksum the repository controls evidence"
# shellcheck disable=SC2016 # Workflow shell variable is an intentional literal contract.
grep -Fq '"$REPOSITORY_CONTROLS_PATH"' "$RELEASE" \
  || fail "release must publish the repository controls evidence"
# shellcheck disable=SC2016 # GitHub expression is an intentional literal workflow contract.
grep -Fq 'subject-path: ${{ steps.package.outputs.source_path }}' "$RELEASE" \
  || fail "release must attest the deterministic source archive"
# shellcheck disable=SC2016 # GitHub expression is an intentional literal workflow contract.
grep -Fq 'subject-path: ${{ steps.release_manifest.outputs.manifest_path }}' "$RELEASE" \
  || fail "release must attest the identity manifest"
grep -Fq 'subject-path: SHA256SUMS' "$RELEASE" \
  || fail "release must attest its checksum set"
grep -Fq 'SOURCE_SHA256' "$RELEASE" \
  || fail "release must checksum the deterministic source archive"
grep -Fq 'RELEASE_MANIFEST_SHA256' "$RELEASE" \
  || fail "release must checksum the identity manifest"
grep -Fq 'scripts/ci/publish-immutable-release.sh' "$RELEASE" \
  || fail "release workflow must use the tested immutable publication state machine"
immutable_release_line=$(grep -nF 'scripts/ci/publish-immutable-release.sh' "$RELEASE" | cut -d: -f1)
crates_publish_line=$(grep -nF 'cargo publish -p rscrypto --locked' "$RELEASE" | cut -d: -f1)
[[ "$immutable_release_line" -lt "$crates_publish_line" ]] \
  || fail "release immutability must be verified before crates.io publication"
grep -Fq 'gh release create "$tag"' "$PUBLISH_RELEASE" \
  || fail "immutable publication must create the GitHub release"
grep -Fq -- '--draft' "$PUBLISH_RELEASE" \
  || fail "release assets must be assembled in a draft before immutable publication"
grep -Fq 'gh release verify "$tag"' "$PUBLISH_RELEASE" \
  || fail "release workflow must verify GitHub's immutable release attestation"
grep -Fq 'gh release verify-asset "$tag"' "$PUBLISH_RELEASE" \
  || fail "release workflow must verify assets against the immutable release"
grep -Fq -- "--jq '.assets[].name'" "$PUBLISH_RELEASE" \
  || fail "release workflow must reject missing or unexpected release assets"
grep -Fq -- '--stable-asset "$CRATE_PATH"' "$RELEASE" \
  || fail "release reruns must verify the crates.io-bound package asset"
grep -Fq -- '--stable-asset "$SOURCE_PATH"' "$RELEASE" \
  || fail "release reruns must verify the deterministic source archive"
grep -Fq '.github/rulesets/protect-main.json' "$REPOSITORY_CONTROLS" \
  || fail "repository controls evidence must validate the checked-in policy"
grep -Fq '.github/rulesets/protect-release-tags.json' "$REPOSITORY_CONTROLS" \
  || fail "repository controls evidence must validate immutable release tags"
jq -e '
  .target == "tag"
  and .enforcement == "active"
  and .bypass_actors == []
  and ([.rules[].type] | sort) == ["deletion", "update"]
' "$RELEASE_TAG_POLICY" >/dev/null || fail "release tags must reject updates and deletion without bypass"
jq -e '.enabled == true and (keys == ["enabled"])' "$RELEASE_IMMUTABILITY_POLICY" >/dev/null \
  || fail "repository policy must require immutable releases"
grep -Fq 'repos/$repo/immutable-releases' "$REPOSITORY_CONTROLS" \
  || fail "repository controls evidence must validate immutable releases before tagging"
grep -Fq 'current_user_can_bypass == "never"' "$REPOSITORY_CONTROLS" \
  || fail "repository controls evidence must reject bypass access"
grep -Fq 'run-id: ${{ needs.evidence-gate.outputs.weekly_run_id }}' "$RELEASE" \
  || fail "release must consume non-RISC-V CT artifacts from the validated Weekly run"
grep -Fq 'run-id: ${{ needs.evidence-gate.outputs.riscv_run_id }}' "$RELEASE" \
  || fail "release must consume RISC-V CT artifacts from the validated RISC-V run"
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
