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
ACTIONS="$ROOT/.github/actions"
CI="$WORKFLOWS/ci.yaml"
SUITE="$WORKFLOWS/_ci-suite.yaml"
RUST_JOB="$WORKFLOWS/_rust-job.yaml"
WEEKLY="$WORKFLOWS/weekly.yaml"
CT="$WORKFLOWS/ct.yaml"
SCORECARD="$WORKFLOWS/scorecard.yaml"
RISCV="$WORKFLOWS/riscv.yaml"
RELEASE="$WORKFLOWS/release.yaml"
RSA="$WORKFLOWS/rsa.yaml"
SETUP_ACTION="$ACTIONS/setup/action.yaml"
TOOLCHAIN_ACTION="$ACTIONS/setup-toolchain/action.yaml"
MANIFEST="$ROOT/.config/target-matrix.json"
RAIL_CONFIG="$ROOT/.config/rail.toml"
RAIL_VARIANTS="$ROOT/.config/ci-plan-variants.json"
TOOL_ARCHIVES="$ROOT/.config/ci-tool-archives.tsv"
CARGO_CONFIG="$ROOT/.cargo/config.toml"
CROSS_SCRIPT="$ROOT/scripts/ci/cross-targets.sh"
NOSTD_WASM="$ROOT/scripts/ci/nostd-wasm-suite.sh"
INSTALL_TOOLS="$ROOT/scripts/ci/install-tools.sh"
MATERIALIZE_RAIL_PLAN="$ROOT/scripts/ci/materialize-rail-plan.sh"
INSTALL_CODECOV="$ROOT/scripts/ci/install-codecov.sh"
SETUP_TOOLCHAIN="$ROOT/scripts/ci/setup-toolchain.sh"
TOOL_INTEGRITY="$ROOT/scripts/lib/ci-tool-integrity.sh"
FEATURE_PROFILES="$ROOT/scripts/lib/feature-profiles.sh"
COMPILE_MATRIX="$ROOT/scripts/check/check-feature-matrix.sh"
HOST_CHECK="$ROOT/scripts/check/check.sh"
EXECUTABLE_MATRIX="$ROOT/scripts/test/test-feature-matrix.sh"
CHECK_ALL="$ROOT/scripts/check/check-all.sh"
CI_CHECK="$ROOT/scripts/ci/ci-check.sh"
RUN_RUST_JOB="$ROOT/scripts/ci/run-rust-job.sh"
RELEASE_PREFLIGHT="$ROOT/scripts/ci/release-preflight.sh"
RELEASE_EVIDENCE="$ROOT/scripts/ci/release-evidence-check.sh"
RELEASE_CT_RECOVERY="$ROOT/scripts/ci/release-ct-recovery-check.sh"
RELEASE_CT_RECOVERY_TEST="$ROOT/scripts/ci/release-ct-recovery-check-test.sh"
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
RUNS_ON="$ROOT/.github/runs-on.yml"

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

feature_sets() {
  local file=$1
  local array=$2
  awk -v array="$array" '
    $0 == array "=(" { in_array = 1; next }
    in_array && /^\)$/ { exit }
    in_array && /^[[:space:]]+"/ {
      value = $0
      sub(/^[[:space:]]+"/, "", value)
      sub(/"$/, "", value)
      print value
    }
  ' "$file"
}

count_feature_sets() {
  feature_sets "$1" "$2" | awk 'END { print NR + 0 }'
}

require_unique_feature_sets() {
  local file=$1
  local array=$2
  local duplicate
  duplicate=$(feature_sets "$file" "$array" | sort | uniq -d | head -1)
  [[ -z "$duplicate" ]] || fail "duplicate feature profile in $array: $duplicate"
}

require_feature_subset() {
  local child_file=$1
  local child_array=$2
  local parent_file=$3
  local parent_array=$4
  local missing
  missing=$(comm -23 \
    <(feature_sets "$child_file" "$child_array" | sort) \
    <(feature_sets "$parent_file" "$parent_array" | sort) | head -1)
  [[ -z "$missing" ]] || fail "executable feature profile lacks compile coverage: $missing"
}

require_file "$CI"
require_file "$SUITE"
require_file "$RUST_JOB"
require_file "$WEEKLY"
require_file "$CT"
require_file "$SCORECARD"
require_file "$RISCV"
require_file "$RELEASE"
require_file "$RSA"
require_file "$SETUP_ACTION"
require_file "$TOOLCHAIN_ACTION"
require_file "$MANIFEST"
require_file "$RAIL_CONFIG"
require_file "$RAIL_VARIANTS"
require_file "$TOOL_ARCHIVES"
require_file "$CARGO_CONFIG"
require_file "$CROSS_SCRIPT"
require_file "$NOSTD_WASM"
require_file "$INSTALL_TOOLS"
require_file "$MATERIALIZE_RAIL_PLAN"
require_file "$INSTALL_CODECOV"
require_file "$SETUP_TOOLCHAIN"
require_file "$TOOL_INTEGRITY"
require_file "$FEATURE_PROFILES"
require_file "$COMPILE_MATRIX"
require_file "$HOST_CHECK"
require_file "$EXECUTABLE_MATRIX"
require_file "$CHECK_ALL"
require_file "$CI_CHECK"
require_file "$RUN_RUST_JOB"
require_file "$RELEASE_PREFLIGHT"
require_file "$RELEASE_EVIDENCE"
require_file "$RELEASE_CT_RECOVERY"
require_file "$RELEASE_CT_RECOVERY_TEST"
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
require_file "$RUNS_ON"

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

[[ $(yq eval '.on.push.branches | join(",")' "$CI") == "main" ]] \
  || fail "CI must seed affected compiler results only after updates reach main"
[[ $(yq eval '[.on.pull_request.types[]] | sort | join(",")' "$CI") \
  == "opened,ready_for_review,reopened,synchronize" ]] \
  || fail "CI must run when a ready pull request is opened, updated, reopened, or leaves draft"
for job in rail-plan suite complete; do
  [[ $(yq eval ".jobs.\"$job\".if" "$CI") == *"pull_request.draft"* ]] \
    || fail "$job must defer draft pull requests"
done

suite_condition=$(yq eval '.jobs.suite."if"' "$CI")
[[ "$suite_condition" == *"!cancelled()"* && "$suite_condition" == *"needs.rail-plan.result == 'success'"* \
  && "$suite_condition" == *"needs.rail-plan.outputs.has-suite == 'true'"* \
  && "$suite_condition" == *"github.event_name != 'push'"* ]] \
  || fail "the CI suite must consume only a successful, non-empty Cargo Rail plan"
cache_seed_condition=$(yq eval '.jobs."cache-seed"."if"' "$CI")
[[ "$cache_seed_condition" == *"github.event_name == 'push'"* \
  && "$cache_seed_condition" == *"github.ref == 'refs/heads/main'"* \
  && "$cache_seed_condition" == *"needs.rail-plan.result == 'success'"* \
  && "$cache_seed_condition" == *"needs.rail-plan.outputs.has-suite == 'true'"* \
  && $(yq eval '.jobs."cache-seed".with.cache_mode' "$CI") == "read-write" ]] \
  || fail "only the affected main-branch seeder may request cache write authority"
# shellcheck disable=SC2016 # GitHub expressions are intentional literal workflow contracts.
grep -Fq 'matrix: ${{ steps.matrix.outputs.matrix }}' "$CI" \
  || fail "CI must export the exact Cargo Rail variant matrix"
# shellcheck disable=SC2016 # GitHub expressions are intentional literal workflow contracts.
grep -Fq 'identity: ${{ steps.rail.outputs.plan-identity }}' "$CI" \
  || fail "CI must export the exact Cargo Rail plan identity"
# shellcheck disable=SC2016 # GitHub expression is an intentional literal workflow contract.
grep -Fq 'matrix: ${{ needs.rail-plan.outputs.matrix }}' "$CI" \
  || fail "the reusable suite must consume the Cargo Rail matrix without replanning"
[[ $(yq -oy -p toml eval '.plan.work."ci-policy".scope' "$RAIL_CONFIG") == "repository" ]] \
  || fail "shared CI infrastructure must have one repository-scoped Cargo Rail policy decision"
grep -Fq 'is-required "$PLAN_FILE" ci-policy' "$MATERIALIZE_RAIL_PLAN" \
  || fail "CI matrix lowering must widen when shared Cargo Rail policy work is required"
if grep -ERn 'always\(\)' "$WORKFLOWS" >/dev/null; then
  fail "workflow cancellation must not start or prolong cleanup and aggregate jobs"
fi

if grep -ERn '^[[:space:]]+(pre_script|run_script):' "$WORKFLOWS" >/dev/null; then
  fail "reusable workflows must not accept executable shell fragments"
fi
if grep -ERin '(^|[^[:alnum:]_])(macos|darwin|apple)([^[:alnum:]_]|$)' "$WORKFLOWS" >/dev/null; then
  fail "Apple platform testing must remain local and must not appear in CI workflows"
fi
if awk '
  /^\[target\./ {
    apple_target = tolower($0) ~ /(apple-darwin|target_os[[:space:]]*=[[:space:]]*"macos")/
    next
  }
  /^\[/ { apple_target = 0 }
  apple_target && /^[[:space:]]*rustflags[[:space:]]*=/ { found = 1 }
  END { exit !found }
' "$CARGO_CONFIG"; then
  fail "Apple targets must not receive implicit rustflags from .cargo/config.toml"
fi
while IFS= read -r operation; do
  [[ -n "$operation" ]] || continue
  grep -Eq "^[[:space:]]+$operation\\)" "$RUN_RUST_JOB" \
    || fail "reusable Rust job caller selects unsupported operation: $operation"
done < <(
  awk '/^[[:space:]]+operation:[[:space:]]+[-[:alnum:]]+[[:space:]]*$/ { print $2 }' \
    "$WORKFLOWS"/*.yaml | sort -u
)
# shellcheck disable=SC2016 # GitHub expression is an intentional literal workflow contract.
[[ $(yq eval '.jobs.selected.with.operation' "$SUITE") == '${{ matrix.work.operation }}' ]] \
  || fail "the reusable suite must pass only Cargo Rail catalog operations"
while IFS= read -r operation; do
  [[ -n "$operation" ]] || continue
  grep -Eq "^[[:space:]]+$operation\\)" "$RUN_RUST_JOB" \
    || fail "Cargo Rail variant catalog selects unsupported operation: $operation"
done < <(jq -r '.variants[].dimensions.operation' "$RAIL_VARIANTS" | sort -u)
[[ $(yq eval '.on.workflow_call.inputs.operation.required' "$RUST_JOB") == "true" ]] \
  || fail "the reusable Rust job operation must be required"
[[ $(yq eval '.on.workflow_call.inputs.operation.type' "$RUST_JOB") == "string" ]] \
  || fail "the reusable Rust job operation must be typed as a string"
[[ $(yq eval '.on.workflow_call.inputs.checkout_ref.type' "$RUST_JOB") == "string" ]] \
  || fail "the reusable Rust job checkout ref must be typed as a string"
[[ $(yq eval '.on.workflow_call.inputs.rustflags.type' "$RUST_JOB") == "string" ]] \
  || fail "the reusable Rust job rustflags input must be typed as a string"
# shellcheck disable=SC2016 # GitHub expression is an intentional literal workflow contract.
[[ $(yq eval '.jobs.run.steps[] | select(.name == "Checkout") | .with.ref' "$RUST_JOB") \
  == '${{ inputs.plan_head_commit || inputs.checkout_ref || github.sha }}' ]] \
  || fail "the reusable Rust job must prefer the plan-bound source ref before execution"
# shellcheck disable=SC2016 # GitHub expression is an intentional literal workflow contract.
[[ $(yq eval '.jobs.run.steps[] | select(.name == "Download exact work plan") | .with.path' "$RUST_JOB") \
  == '${{ runner.temp }}/cargo-rail-plan' ]] \
  || fail "saved plan artifacts must remain outside the captured checkout"
# shellcheck disable=SC2016 # GitHub expressions are intentional literal workflow contracts.
rail_plan_file=$(yq eval '.jobs.run.steps[] | select(.name == "Run") | .env.RAIL_PLAN_FILE' "$RUST_JOB")
rail_plan_reader=$(yq eval '.jobs.run.steps[] | select(.name == "Run") | .env.RAIL_PLAN_READER' "$RUST_JOB")
[[ "$rail_plan_file" == *"inputs.plan_artifact"* && "$rail_plan_file" == *"runner.temp"* \
  && "$rail_plan_file" == *"/cargo-rail-plan/plan.json"* \
  && "$rail_plan_reader" == *"inputs.plan_artifact"* && "$rail_plan_reader" == *"runner.temp"* \
  && "$rail_plan_reader" == *"/cargo-rail-plan/read.py"* ]] \
  || fail "saved plan execution must consume the out-of-worktree artifact"
# shellcheck disable=SC2016 # GitHub expression is an intentional literal workflow contract.
[[ $(yq eval '.jobs.run.steps[] | select(.name == "Run") | .env.CARGO_TARGET_S390X_UNKNOWN_LINUX_GNU_RUSTFLAGS' "$RUST_JOB") \
  == '${{ inputs.target == '\''s390x-unknown-linux-gnu'\'' && '\''-C target-feature=+vector'\'' || '\'''\'' }}' ]] \
  || fail "s390x CT jobs must share one explicit vector target environment"
# shellcheck disable=SC2016 # GitHub expression is an intentional literal workflow contract.
[[ $(yq eval '.jobs.run.steps[] | select(.name == "Run") | .env.RSCRYPTO_CI_RUSTFLAGS' "$RUST_JOB") \
  == '${{ inputs.rustflags }}' ]] \
  || fail "the reusable Rust job must pass reviewed rustflags as inert environment data"
rust_job_run=$(yq eval '.jobs.run.steps[] | select(.name == "Run") | .run' "$RUST_JOB")
[[ -n "$rust_job_run" && "$rust_job_run" != "null" ]] \
  || fail "the reusable Rust job must expose one fixed command step"
grep -Fq 'if [[ -n "$RSCRYPTO_CI_RUSTFLAGS" ]]' <<<"$rust_job_run" \
  || fail "the reusable Rust job must leave RUSTFLAGS unset without a reviewed override"
grep -Fq 'export RUSTFLAGS="$RSCRYPTO_CI_RUSTFLAGS"' <<<"$rust_job_run" \
  || fail "the reusable Rust job must export only the reviewed rustflags value"
grep -Fq 'exec scripts/ci/run-rust-job.sh' <<<"$rust_job_run" \
  || fail "the reusable Rust job must invoke the repository-owned dispatcher"
# shellcheck disable=SC2016 # GitHub expression is an intentional literal workflow contract.
grep -Fq 'RSCRYPTO_CI_OPERATION: ${{ inputs.operation }}' "$RUST_JOB" \
  || fail "the reusable Rust job must pass its operation as environment data"
# shellcheck disable=SC2016 # GitHub expression is an intentional literal workflow contract.
[[ $(yq eval '.jobs.run.steps[] | select(.name == "Run") | .env.RAIL_PLAN_CHECKOUT_VERIFIED' "$RUST_JOB") \
  == '${{ inputs.plan_artifact != '\'''\'' && '\''true'\'' || '\''false'\'' }}' ]] \
  || fail "only jobs with a separately verified saved plan may skip duplicate checkout verification"
if yq eval '.. | select(tag == "!!map" and has("run") and (.run | tag == "!!str")) | .run' \
  "$WORKFLOWS"/*.yaml | grep -Eq '\$\{\{[[:space:]]*inputs\.'; then
  fail "workflow inputs must not be interpolated into shell programs"
fi
if grep -En '(^|[[:space:]])eval[[:space:]]|(^|[[:space:]])(bash|sh)[[:space:]]+-c|<<<' \
  "$RUN_RUST_JOB" >/dev/null; then
  fail "the Rust job dispatcher must not invoke a dynamic shell interpreter"
fi

bash -eu -o pipefail -c 'source "$1"; ci_tool_validate_manifest' _ "$TOOL_INTEGRITY" \
  || fail "direct CI tool archive manifest is invalid"

if grep -ERn 'uses:[[:space:]]+dtolnay/rust-toolchain@' \
  "$WORKFLOWS" "$ACTIONS" >/dev/null; then
  fail "CI must not delegate installation to an action with an unauthenticated executable fallback"
fi
if grep -En 'cargo[[:space:]]+binstall|cargo-binstall|releases/latest|/latest/|curl[^|]*\|[[:space:]]*(bash|sh)' \
  "$INSTALL_TOOLS" "$SETUP_TOOLCHAIN" "$NOSTD_WASM" "$CROSS_SCRIPT" "$INSTALL_CODECOV" >/dev/null; then
  fail "CI tool installers must reject Cargo-binstall, mutable URLs, and piped network installers"
fi

download_files=$(
  {
    grep -ERl --include='*.sh' --include='*.yaml' --include='*.yml' \
      '(^|[[:space:]])(curl|wget|aria2c)([[:space:]]|$)|Invoke-(WebRequest|RestMethod)|Start-BitsTransfer|gh[[:space:]]+release[[:space:]]+download' \
      "$ROOT/scripts" "$WORKFLOWS" "$ACTIONS" 2>/dev/null || true
  } | while IFS= read -r file; do
    case "$file" in
      "$ROOT/scripts/ci/check-ci-ownership-test.sh" \
        | "$ROOT/scripts/ci/tool-integrity-test.sh" \
        | "$ROOT/scripts/ci/check-ci-ownership.sh") continue ;;
    esac
    printf '%s\n' "${file#"$ROOT"/}"
  done | sort
)
expected_download_files=$(printf '%s\n' \
  '.github/workflows/release.yaml' \
  'scripts/ci/check-action-pins.sh' \
  'scripts/lib/ci-tool-integrity.sh')
[[ "$download_files" == "$expected_download_files" ]] \
  || fail "network downloads exist outside the tool verifier or reviewed non-tool paths"

installer_files=$(
  {
    grep -ERl --include='*.sh' --include='*.yaml' --include='*.yml' \
      'cargo[[:space:]]+(binstall|install)|go[[:space:]]+install|apt(-get)?[[:space:]]+install|opam[[:space:]]+(init|install|reinstall|switch[[:space:]]+create)|rustup[[:space:]]+(toolchain[[:space:]]+install|component[[:space:]]+add|target[[:space:]]+add)|install_args=\(toolchain[[:space:]]+install|pipx?[[:space:]]+install|uv[[:space:]]+tool[[:space:]]+install|npm[[:space:]]+(install|ci)|pnpm[[:space:]]+install|yarn[[:space:]]+install|brew[[:space:]]+install' \
      "$ROOT/scripts/ci" "$ROOT/scripts/lib" "$WORKFLOWS" "$ACTIONS" 2>/dev/null || true
  } | while IFS= read -r file; do
    case "$file" in
      "$ROOT/scripts/ci/check-ci-ownership-test.sh" \
        | "$ROOT/scripts/ci/tool-integrity-test.sh" \
        | "$ROOT/scripts/ci/check-ci-ownership.sh") continue ;;
    esac
    printf '%s\n' "${file#"$ROOT"/}"
  done | sort
)
expected_installer_files=$(printf '%s\n' \
  'scripts/ci/install-tools.sh' \
  'scripts/ci/nostd-wasm-suite.sh' \
  'scripts/ci/setup-toolchain.sh' \
  'scripts/lib/common.sh')
[[ "$installer_files" == "$expected_installer_files" ]] \
  || fail "package-manager installs exist outside the reviewed integrity boundaries"

rail_action=$(yq eval '.jobs."rail-plan".steps[] | select(.id == "rail") | .uses' "$CI")
[[ "$rail_action" =~ ^loadingalias/cargo-rail-action@[0-9a-f]{40}$ ]] \
  || fail "the PR planner must use commit-pinned cargo-rail-action"
rail_version=$(sed -n 's/^CARGO_RAIL_VERSION=//p' "$INSTALL_TOOLS")
[[ "$rail_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
  || fail "the Cargo Rail installer version must be exact"
[[ $(yq eval '.jobs."rail-plan".steps[] | select(.id == "rail") | .with.version' "$CI") \
  == "$rail_version" ]] \
  || fail "cargo-rail-action must use the authenticated Cargo Rail version"
[[ $(yq eval '.jobs."rail-plan".steps[] | select(.id == "rail") | .with.components' "$CI") \
  == "surface" ]] \
  || fail "cargo-rail-action must install and prepare the authenticated Surface component"
# shellcheck disable=SC2016 # GitHub expressions are intentional literal workflow contracts.
rail_since=$(yq eval '.jobs."rail-plan".steps[] | select(.id == "rail") | .with.since' "$CI")
[[ "$rail_since" == *'github.event.pull_request.base.sha'* \
  && "$rail_since" == *'github.event.before'* && "$rail_since" == *'||'* ]] \
  || fail "cargo-rail-action must plan pull requests and main pushes from immutable comparison commits"
# shellcheck disable=SC2016 # GitHub expression is an intentional literal workflow contract.
rail_all=$(yq eval '.jobs."rail-plan".steps[] | select(.id == "rail") | .with.all' "$CI")
[[ "$rail_all" == *"github.event_name == 'workflow_dispatch'"* ]] \
  || fail "manual CI must request Cargo Rail's typed all-work override"

cache_action=$(yq eval '.runs.steps[] | select(.name == "Setup Cargo Rail Cache") | .uses' "$SETUP_ACTION")
expected_cache_action="${rail_action%@*}/cache@${rail_action#*@}"
[[ "$cache_action" == "$expected_cache_action" ]] \
  || fail "compiler reuse must use the planner action's immutable v8 cache implementation"
cache_auth_run=$(yq eval '.runs.steps[] | select(.name == "Authenticate Cargo Rail Cache") | .run' "$SETUP_ACTION")
grep -Fq '[[ "$CACHE_URL" == r2://* ]]' <<<"$cache_auth_run" \
  || fail "the shared cache must reject non-canonical providers before authentication"
grep -Fq 'AWS_ACCESS_KEY_ID=%s' <<<"$cache_auth_run" \
  || fail "the shared cache must export the caller-selected R2 access key for Cargo Rail"
grep -Fq 'AWS_SECRET_ACCESS_KEY=%s' <<<"$cache_auth_run" \
  || fail "the shared cache must export the caller-selected R2 secret key for Cargo Rail"
grep -Fq 'AWS_SESSION_TOKEN=' <<<"$cache_auth_run" \
  || fail "the shared cache must clear an ambient AWS session token before installing a long-lived R2 key"
grep -Fq 'AWS_PROFILE=' <<<"$cache_auth_run" \
  || fail "the shared cache must clear an ambient AWS profile before installing its selected R2 identity"
grep -Fq 'AWS_SHARED_CREDENTIALS_FILE=' <<<"$cache_auth_run" \
  || fail "the shared cache must clear an ambient AWS credentials file before installing its selected R2 identity"
grep -Fq '::add-mask::%s' <<<"$cache_auth_run" \
  || fail "the shared cache must mask both caller-selected R2 credential fields"
if grep -ERn 'aws-actions/configure-aws-credentials|CARGO_RAIL_CACHE_(READ_ROLE_ARN|WRITE_ROLE_ARN|REGION)|cache-(read-role-arn|write-role-arn|region)' \
  "$WORKFLOWS" "$ACTIONS" >/dev/null; then
  fail "the canonical R2 integration must not retain AWS role or region compatibility plumbing"
fi
[[ $(yq eval '.runs.steps[] | select(.name == "Setup Cargo Rail Cache") | .with.version' "$SETUP_ACTION") \
  == "$rail_version" ]] \
  || fail "the Cargo Rail cache action must use the authenticated Cargo Rail version"
[[ $(yq eval '.runs.steps[] | select(.name == "Setup Cargo Rail Cache") | .with.url' "$SETUP_ACTION") \
  == '${{ inputs.cache-url }}' ]] \
  || fail "the Cargo Rail cache action must consume only machine-owned URL input"
[[ $(yq eval '.runs.steps[] | select(.name == "Setup Cargo Rail Cache") | .with.mode' "$SETUP_ACTION") \
  == '${{ inputs.cache-mode }}' ]] \
  || fail "the Cargo Rail cache action must consume the trust-selected remote mode"
[[ $(yq eval '.runs.steps[] | select(.name == "Setup Cargo Rail Cache") | .with."max-size"' "$SETUP_ACTION") \
  == '${{ inputs.cache-max-size }}' ]] \
  || fail "the Cargo Rail cache action must retain an explicit local size bound"
[[ $(yq eval '.runs.steps[] | select(.name == "Setup Cargo Rail Cache") | .with."root-portability"' "$SETUP_ACTION") \
  == '${{ inputs.cache-root-portability }}' ]] \
  || fail "the Cargo Rail cache action must own the selected root-portability transaction"
[[ $(yq eval '.runs.steps[] | select(.name == "Setup Cargo Rail Cache") | .with."strict-probe"' "$SETUP_ACTION") \
  == "true" ]] \
  || fail "CI cache setup must authenticate the provider and native-v6 protocol marker before compilation"
[[ $(yq eval '.runs.steps[] | select(.name == "Setup Cargo Rail Cache") | .if' "$SETUP_ACTION") \
  == "steps.cache-capability.outputs.enabled == 'true'" \
  && $(yq eval '.runs.steps[] | select(.name == "Authenticate Cargo Rail Cache") | .if' "$SETUP_ACTION") \
  == "steps.cache-capability.outputs.enabled == 'true'" ]] \
  || fail "Cargo Rail cache authentication and setup must consume one platform capability decision"
cache_capability_run=$(yq eval '.runs.steps[] | select(.name == "Select Cargo Rail Cache Capability") | .run' "$SETUP_ACTION")
grep -Fq '[[ -z "$CACHE_URL" ]]' <<<"$cache_capability_run" \
  && grep -Fq 's390x | ppc64le)' <<<"$cache_capability_run" \
  || fail "cache setup must skip absent configuration and hosts without verified native cache archives"
auth_step_index=$(yq eval '.runs.steps | to_entries | .[] | select(.value.name == "Authenticate Cargo Rail Cache") | .key' "$SETUP_ACTION")
cache_step_index=$(yq eval '.runs.steps | to_entries | .[] | select(.value.name == "Setup Cargo Rail Cache") | .key' "$SETUP_ACTION")
tools_step_index=$(yq eval '.runs.steps | to_entries | .[] | select(.value.name == "Install Cargo Tools") | .key' "$SETUP_ACTION")
[[ "$auth_step_index" =~ ^[0-9]+$ && "$cache_step_index" =~ ^[0-9]+$ \
  && "$tools_step_index" =~ ^[0-9]+$ && "$auth_step_index" -lt "$cache_step_index" \
  && "$cache_step_index" -lt "$tools_step_index" ]] \
  || fail "Cargo Rail cache setup must precede repository command execution"
[[ $(yq eval '.inputs."cache-root-portability".default' "$SETUP_ACTION") == "remap" ]] \
  || fail "ephemeral CI checkouts must select qualified cross-root Cargo Rail reuse"
# shellcheck disable=SC2016 # GitHub expression is an intentional literal workflow contract.
[[ $(yq eval '.runs.steps[] | select(.name == "Install Cargo Tools") | .env.RSCRYPTO_AUTHENTICATED_CARGO_RAIL' "$SETUP_ACTION") \
  == '${{ steps.cargo-rail-cache.outcome == '\''success'\'' && '\''true'\'' || '\''false'\'' }}' ]] \
  || fail "Cargo Rail reuse must be authorized only by the exact cache-action install output"
grep -Fq 'RSCRYPTO_AUTHENTICATED_CARGO_RAIL' "$INSTALL_TOOLS" \
  || fail "the tool installer must reuse authenticated Cargo Rail instead of reinstalling it"

# shellcheck disable=SC2016 # GitHub expressions are intentional literal workflow contracts.
expected_cache_url="\${{ secrets.cache_access_key_id != '' && secrets.cache_secret_access_key != '' && vars.CARGO_RAIL_CACHE_URL || '' }}"
[[ $(yq eval '.jobs.run.steps[] | select(.name == "Setup") | .with."cache-url"' "$RUST_JOB") \
  == "$expected_cache_url" ]] \
  || fail "reusable Rust jobs must disable L2 when a caller-selected R2 credential is unavailable"
# shellcheck disable=SC2016 # GitHub expressions are intentional literal workflow contracts.
[[ $(yq eval '.jobs.run.steps[] | select(.name == "Setup") | .with."cache-mode"' "$RUST_JOB") \
  == '${{ inputs.cache_mode }}' \
  && $(yq eval '.on.workflow_call.inputs.cache_mode.default' "$RUST_JOB") == "read" \
  && $(yq eval '.on.workflow_call.inputs.cache_mode.default' "$SUITE") == "read" ]] \
  || fail "ordinary reusable compiler jobs must default to read-only cache authority"
# shellcheck disable=SC2016 # GitHub expressions are intentional literal workflow contracts.
[[ $(yq eval '.jobs.run.steps[] | select(.name == "Setup") | .with."cache-access-key-id"' "$RUST_JOB") \
  == '${{ secrets.cache_access_key_id }}' \
  && $(yq eval '.jobs.run.steps[] | select(.name == "Setup") | .with."cache-secret-access-key"' "$RUST_JOB") \
  == '${{ secrets.cache_secret_access_key }}' ]] \
  || fail "reusable compiler jobs must consume only the caller-selected provider identity"
[[ $(yq eval '.on.workflow_call.secrets.cache_access_key_id.required' "$RUST_JOB") == "false" \
  && $(yq eval '.on.workflow_call.secrets.cache_secret_access_key.required' "$RUST_JOB") == "false" ]] \
  || fail "fork and Dependabot jobs must be able to execute without repository R2 secrets"
[[ $(yq eval '.jobs.selected.secrets.cache_access_key_id' "$SUITE") == '${{ secrets.cache_access_key_id }}' \
  && $(yq eval '.jobs.selected.secrets.cache_secret_access_key' "$SUITE") == '${{ secrets.cache_secret_access_key }}' ]] \
  || fail "the selected CI suite must forward only its caller-selected cache identity"
[[ $(yq eval '.jobs.suite.secrets.cache_access_key_id' "$CI") == '${{ secrets.CARGO_RAIL_R2_READ_ACCESS_KEY_ID }}' \
  && $(yq eval '.jobs.suite.secrets.cache_secret_access_key' "$CI") == '${{ secrets.CARGO_RAIL_R2_READ_SECRET_ACCESS_KEY }}' ]] \
  || fail "ordinary pull-request CI must receive only the bucket-scoped R2 reader"
[[ $(yq eval '.jobs."cache-seed".secrets.cache_access_key_id' "$CI") == '${{ secrets.CARGO_RAIL_R2_WRITE_ACCESS_KEY_ID }}' \
  && $(yq eval '.jobs."cache-seed".secrets.cache_secret_access_key' "$CI") == '${{ secrets.CARGO_RAIL_R2_WRITE_SECRET_ACCESS_KEY }}' ]] \
  || fail "the trusted main seeder must receive the distinct bucket-scoped R2 writer"
[[ $(count_matches 'CARGO_RAIL_R2_WRITE_(ACCESS_KEY_ID|SECRET_ACCESS_KEY)' "$WORKFLOWS") -eq 2 ]] \
  || fail "R2 writer credentials must exist only at the trusted main seeder boundary"
# shellcheck disable=SC2016 # GitHub expressions are intentional literal workflow contracts.
[[ $(yq eval '.jobs.coverage.steps[] | select(.name == "Setup") | .with."cache-url"' "$WEEKLY") \
  == *'vars.CARGO_RAIL_CACHE_URL'* \
  && $(yq eval '.jobs.coverage.steps[] | select(.name == "Setup") | .with."cache-mode"' "$WEEKLY") \
  == "read" ]] \
  || fail "Qualification coverage must consume the machine-owned Cargo Rail cache read-only"
# shellcheck disable=SC2016 # GitHub expressions are intentional literal workflow contracts.
[[ $(yq eval '.jobs.preflight.steps[] | select(.name == "Setup") | .with."cache-url"' "$RELEASE") \
  == *'vars.CARGO_RAIL_CACHE_URL'* \
  && $(yq eval '.jobs.preflight.steps[] | select(.name == "Setup") | .with."cache-mode"' "$RELEASE") \
  == "read" ]] \
  || fail "release preflight must consume the machine-owned Cargo Rail cache read-only"
# shellcheck disable=SC2016 # GitHub expressions are intentional literal workflow contracts.
[[ $(yq eval '.jobs.publish.steps[] | select(.name == "Setup") | .with."cache-url"' "$RELEASE") \
  == *'vars.CARGO_RAIL_CACHE_URL'* \
  && $(yq eval '.jobs.publish.steps[] | select(.name == "Setup") | .with."cache-mode"' "$RELEASE") \
  == "read" ]] \
  || fail "release publication must consume the machine-owned Cargo Rail cache read-only"
[[ $(yq eval '[.jobs.run.steps[] | select(.name == "Capture Cargo Rail Cache Status")] | length' "$RUST_JOB") == "1" \
  && $(yq eval '[.jobs.run.steps[] | select(.name == "Upload Cargo Rail Cache Status")] | length' "$RUST_JOB") == "1" ]] \
  || fail "representative compiler jobs must preserve one fail-closed Cargo Rail telemetry artifact"
[[ $(yq eval '[.jobs.coverage.steps[] | select(.name == "Capture Cargo Rail Cache Status")] | length' "$WEEKLY") == "1" \
  && $(yq eval '[.jobs.coverage.steps[] | select(.name == "Upload Cargo Rail Cache Status")] | length' "$WEEKLY") == "1" ]] \
  || fail "Qualification coverage must preserve one fail-closed Cargo Rail telemetry artifact"
if grep -ERn 'uses:[[:space:]]+(Swatinem/rust-cache|runs-on/action|actions/cache)@' \
  "$WORKFLOWS" "$ACTIONS" >/dev/null; then
  fail "Cargo Rail must be the only Rust compiler cache owner"
fi
[[ $(yq eval '[.runners[].extras[]? | select(. == "s3-cache")] | length' "$RUNS_ON") == "0" ]] \
  || fail "RunsOn MagicCache must not intercept Cargo Rail compiler results"
# shellcheck disable=SC2016 # GitHub expressions are intentional literal workflow contracts.
[[ $(yq eval '.jobs."rail-plan".steps[] | select(.name == "Check Release Intent Coverage") | .env.RAIL_BASE_REF' "$CI") \
  == '${{ steps.rail.outputs.base }}' ]] \
  || fail "release intent coverage must use cargo-rail-action's resolved base"
release_intent_condition=$(yq eval '.jobs."rail-plan".steps[] | select(.name == "Check Release Intent Coverage") | .if' "$CI")
[[ "$release_intent_condition" == *"startsWith(github.head_ref, 'rail/release-')"* \
  && "$release_intent_condition" == *"github.event.pull_request.head.repo.full_name == github.repository"* ]] \
  || fail "only repository-owned Cargo Rail release PRs may consume change intent"
grep -Fq 'scripts/ci/setup-toolchain.sh "$TOOLCHAIN" "$TOOLCHAIN_COMPONENTS" "$GITHUB_ENV"' "$TOOLCHAIN_ACTION" \
  || fail "toolchain setup must use the repository-owned rustup policy"
grep -Fq "printf 'RUSTUP_TOOLCHAIN=%s\\n' \"\$toolchain\" >>\"\$github_env\"" "$SETUP_TOOLCHAIN" \
  || fail "toolchain setup must activate the resolved contract for later steps"
grep -Fq 'RUSTUP_TOOLCHAIN="$TOOLCHAIN" rustc --version --verbose' "$TOOLCHAIN_ACTION" \
  || fail "toolchain setup must verify the activated contract without a rust-toolchain override"
if grep -Fq 'rustup default ' "$SETUP_TOOLCHAIN"; then
  fail "toolchain setup must not mutate a runner-global default"
fi
if grep -Eq '[.]cargo/(bin|[.]crates)|[.]opam' "$SETUP_ACTION"; then
  fail "CI tool executables and OPAM switches must not be restored from caches"
fi
if grep -Fq 'export PATH="$HOME/.cargo/bin:$PATH"' "$CI_CHECK" "$HOST_CHECK"; then
  fail "CI checks must not place unverified runner tools ahead of the authenticated tool root"
fi
grep -Fq 'RSCRYPTO_TOOL_ROOT=$(mktemp -d ' "$INSTALL_TOOLS" \
  || fail "package-manager tools must install into a fresh per-run root"
grep -Fq 'export CARGO_HOME="$RSCRYPTO_CARGO_HOME"' "$INSTALL_TOOLS" \
  || fail "Cargo tool sources and executables must use the fresh per-run root"
grep -Fq 'GOMODCACHE="$RSCRYPTO_TOOL_ROOT/go/pkg/mod"' "$INSTALL_TOOLS" \
  || fail "Go tool sources must use the fresh per-run root"
grep -Fq 'export OPAMROOT="$RSCRYPTO_TOOL_ROOT/opam"' "$INSTALL_TOOLS" \
  || fail "OPAM tool sources and executables must use the fresh per-run root"
grep -Fq 'cargo install --registry crates-io "$package" --locked --version "=$version" --force' "$INSTALL_TOOLS" \
  || fail "Cargo tools must use exact authenticated crates.io installs"
grep -Fq 'go install "github.com/rhysd/actionlint/cmd/actionlint@v$ACTIONLINT_VERSION"' "$INSTALL_TOOLS" \
  || fail "Go tools must use exact checksum-database-backed module versions"
opam_commit=$(sed -n 's/^OPAM_REPOSITORY_COMMIT=//p' "$INSTALL_TOOLS")
[[ "$opam_commit" =~ ^[0-9a-f]{40}$ ]] \
  || fail "OPAM repository must use a full Git commit"
[[ $(sed -n 's/^OPAM_REPOSITORY_REMOTE=//p' "$INSTALL_TOOLS") \
  == "https://github.com/ocaml/opam-repository.git" ]] \
  || fail "OPAM repository must use the reviewed HTTPS remote"
grep -Fq 'git -C "$repository" fetch --depth=1 --no-tags' "$INSTALL_TOOLS" \
  || fail "OPAM repository must fetch only the pinned commit"
grep -Fq 'actual=$(git -C "$repository" rev-parse HEAD)' "$INSTALL_TOOLS" \
  || fail "OPAM metadata must be checked against its pinned commit"
grep -Fq 'status=$(git -C "$repository" status --short --untracked-files=all)' "$INSTALL_TOOLS" \
  || fail "OPAM metadata must match the pinned commit exactly"
grep -Fq 'actual=$(dpkg-query -W -f=' "$INSTALL_TOOLS" \
  || fail "APT packages must be validated against exact versions"
grep -Fq 'ci_tool_download wasmtime' "$NOSTD_WASM" \
  || fail "Wasmtime must use the direct archive integrity contract"
grep -Fq 'ci_tool_download wasm-tools' "$NOSTD_WASM" \
  || fail "wasm-tools must use the direct archive integrity contract"
if grep -Eiq '(^|[^[:alnum:]_])zig([^[:alnum:]_]|$)' "$CROSS_SCRIPT"; then
  fail "cross-target CI must not depend on Zig"
fi
grep -Fq 'ci_tool_download codecov' "$INSTALL_CODECOV" \
  || fail "Codecov must use the direct executable integrity contract"
[[ $(yq eval '.jobs.coverage.steps[] | select(.id == "codecov") | .run' "$WEEKLY") \
  == "scripts/ci/install-codecov.sh" ]] \
  || fail "Qualification coverage must install the authenticated Codecov CLI"
# shellcheck disable=SC2016 # GitHub expression is an intentional literal contract.
grep -Fq 'binary: ${{ steps.codecov.outputs.binary }}' "$WEEKLY" \
  || fail "Codecov action must use the repository-verified CLI"
scorecard_action=$(yq eval '.jobs.scorecard.steps[] | select(.name == "Run Scorecard") | .uses' "$SCORECARD")
[[ "$scorecard_action" =~ ^ossf/scorecard-action@[0-9a-f]{40}$ ]] \
  || fail "Scorecard publication must call the official action at an immutable commit"
[[ $(yq eval '.jobs.scorecard.steps[] | select(.name == "Run Scorecard") | .with.publish_results' "$SCORECARD") \
  == "true" ]] \
  || fail "Scorecard must publish results"

[[ $(count_feature_sets "$FEATURE_PROFILES" COMPILE_FEATURE_SETS) -eq 60 ]] \
  || fail "compile feature matrix must retain all 60 profiles"
[[ $(count_feature_sets "$FEATURE_PROFILES" EXECUTABLE_FEATURE_SETS) -eq 9 ]] \
  || fail "executable feature matrix must contain the nine behavior transitions"
[[ $(count_feature_sets "$FEATURE_PROFILES" CONSTRAINED_FEATURE_SETS) -eq 32 ]] \
  || fail "constrained feature matrix must contain all 32 portable profiles"
require_unique_feature_sets "$FEATURE_PROFILES" COMPILE_FEATURE_SETS
require_unique_feature_sets "$FEATURE_PROFILES" EXECUTABLE_FEATURE_SETS
require_unique_feature_sets "$FEATURE_PROFILES" CONSTRAINED_FEATURE_SETS
require_feature_subset "$FEATURE_PROFILES" EXECUTABLE_FEATURE_SETS "$FEATURE_PROFILES" COMPILE_FEATURE_SETS
grep -Fq 'COMPILE_FEATURE_SETS' "$COMPILE_MATRIX" \
  || fail "compile feature matrix must consume the shared profile authority"
grep -Fq 'EXECUTABLE_FEATURE_SETS' "$EXECUTABLE_MATRIX" \
  || fail "executable feature matrix must consume the shared profile authority"
grep -Fq '"$SCRIPT_DIR/check.sh" --all --feature-matrix' "$CHECK_ALL" \
  || fail "local check-all must retain one explicit feature-matrix execution"

[[ $(count_matches 'just test-feature-matrix' "$WORKFLOWS" "$RUN_RUST_JOB") -eq 1 ]] \
  || fail "ordinary workflows must have exactly one executable feature-matrix owner"
[[ $(count_matches 'just check-feature-matrix' "$WORKFLOWS" "$RUN_RUST_JOB") -eq 1 ]] \
  || fail "ordinary workflows must have exactly one compile feature-matrix owner"
[[ $(yq eval '.on.workflow_dispatch.inputs.tag.required' "$RELEASE") == "true" ]] \
  || fail "release recovery must require an explicit existing tag"
[[ $(yq eval '.on.workflow_dispatch.inputs.tag.type' "$RELEASE") == "string" ]] \
  || fail "release recovery tag input must be a string"
[[ $(yq eval '.on.workflow_dispatch.inputs.s390x_ct_run.type' "$RELEASE") == "string" ]] \
  || fail "release recovery s390x CT run input must be a string"
[[ $(yq eval '.on.workflow_dispatch.inputs.x86_64_ct_run.type' "$RELEASE") == "string" ]] \
  || fail "release recovery x86_64 CT run input must be a string"
# shellcheck disable=SC2016 # GitHub expressions are intentional literal workflow contracts.
[[ $(yq eval '.jobs.preflight.steps[] | select(.name == "Checkout") | .with.ref' "$RELEASE") \
  == '${{ github.event_name == '\''workflow_dispatch'\'' && inputs.tag || github.ref }}' ]] \
  || fail "release preflight must check out the requested recovery tag"
# shellcheck disable=SC2016 # GitHub expressions are intentional literal workflow contracts.
[[ $(yq eval '.jobs.publish.steps[] | select(.name == "Checkout") | .with.ref' "$RELEASE") \
  == '${{ needs.preflight.outputs.release_tag }}' ]] \
  || fail "release publication must check out the preflight-verified tag"
identity_step=$(yq eval '.jobs.preflight.steps[] | select(.id == "identity") | .run' "$RELEASE")
grep -Fq 'refs/heads/main' <<<"$identity_step" \
  || fail "release recovery must reject workflow code outside protected main"
recovery_cleanup_step=$(yq eval '.jobs.preflight.steps[] | select(.name == "Remove reviewed recovery tooling") | .run' "$RELEASE")
grep -Fq 'rm -rf target/release-automation' <<<"$recovery_cleanup_step" \
  || fail "release recovery must remove its reviewed tooling checkout after preflight"
recovery_preflight_step=$(yq eval '.jobs.preflight.steps[] | select(.name == "Release preflight") | .run' "$RELEASE")
grep -Fq 'target/release-automation/scripts/ci/release-preflight.sh' <<<"$recovery_preflight_step" \
  || fail "release recovery must run reviewed preflight tooling from protected main"
if grep -Fq -- '--dependency-policy-root' <<<"$recovery_preflight_step"; then
  fail "tag recovery must not revive a release-specific dependency compatibility path"
fi
recovery_controls_step=$(yq eval '.jobs.publish.steps[] | select(.name == "Capture repository controls") | .run' "$RELEASE")
grep -Fq 'target/release-automation/scripts/ci/repository-controls-evidence.sh' <<<"$recovery_controls_step" \
  || fail "release recovery must capture controls with reviewed tooling from protected main"
grep -Fq -- '--policy-root target/release-automation' <<<"$recovery_controls_step" \
  || fail "release recovery must use the reviewed repository policy from protected main"
ct_recovery_step=$(yq eval '.jobs.preflight.steps[] | select(.name == "Verify s390x CT recovery evidence") | .run' "$RELEASE")
grep -Fq 'release-ct-recovery-check.sh' <<<"$ct_recovery_step" \
  || fail "release recovery must validate replacement s390x CT evidence"
grep -Fq -- '--workflow-commit "$WORKFLOW_COMMIT"' <<<"$ct_recovery_step" \
  || fail "replacement s390x CT evidence must come from the reviewed workflow commit"
grep -Fq -- '--platform-group s390x' <<<"$ct_recovery_step" \
  || fail "replacement s390x CT evidence must validate the s390x platform group"
x86_ct_recovery_step=$(yq eval '.jobs.preflight.steps[] | select(.name == "Verify x86_64 CT recovery evidence") | .run' "$RELEASE")
grep -Fq 'release-ct-recovery-check.sh' <<<"$x86_ct_recovery_step" \
  || fail "release recovery must validate replacement x86_64 CT evidence"
grep -Fq -- '--workflow-commit "$WORKFLOW_COMMIT"' <<<"$x86_ct_recovery_step" \
  || fail "replacement x86_64 CT evidence must come from the reviewed workflow commit"
grep -Fq -- '--platform-group x86_64' <<<"$x86_ct_recovery_step" \
  || fail "replacement x86_64 CT evidence must validate the complete x86_64 platform group"
[[ $(yq -oy -p toml eval '.release.semver_check' "$RAIL_CONFIG") == "off" ]] \
  || fail "pre-1.0 Cargo Rail SemVer enforcement must remain explicitly disabled"
if grep -ERn 'cargo[ -]semver-checks' "$WORKFLOWS" "$RELEASE_PREFLIGHT" "$INSTALL_TOOLS" >/dev/null; then
  fail "pre-1.0 workflows and installers must not reintroduce SemVer enforcement"
fi

if grep -ERn 'just check --all|check-all\.sh' "$WORKFLOWS" "$RUN_RUST_JOB" >/dev/null; then
  fail "native workflows must not invoke comprehensive cross-target checks"
fi

if grep -En 'test-feature-matrix|check-feature-matrix' "$WEEKLY" >/dev/null; then
  fail "weekly must inherit feature contracts from the reusable suite"
fi

[[ $(jq '[.variants[] | select(.dimensions.operation == "cross-targets")] | length' "$RAIL_VARIANTS") -eq 1 ]] \
  || fail "the Cargo Rail catalog must have exactly one cross-target owner"
[[ $(count_matches 'scripts/ci/cross-targets\.sh' "$RUN_RUST_JOB") -eq 1 ]] \
  || fail "the Rust job dispatcher must define exactly one cross-target operation"
[[ $(jq '[.variants[] | select(.dimensions.operation == "native")] | length' "$RAIL_VARIANTS") -eq 4 ]] \
  || fail "the Cargo Rail catalog must own the four native Linux and Windows rows"
[[ $(jq '[.variants[] | select(.dimensions.operation == "native-ibm")] | length' "$RAIL_VARIANTS") -eq 2 ]] \
  || fail "the Cargo Rail catalog must own the IBM Z and POWER10 native rows"
[[ $(count_matches 'operation:[[:space:]]+native-riscv' "$RISCV") -eq 1 ]] \
  || fail "the manual RISC-V workflow must own exactly one native diagnostic lane"
[[ $(count_matches 'operation:[[:space:]]+native-riscv' "$WEEKLY") -eq 1 ]] \
  || fail "Qualification must own exactly one RISC-V native evidence lane"
[[ $(count_matches 'scripts/ci/native-check\.sh' "$RUN_RUST_JOB") -eq 3 ]] \
  || fail "the Rust job dispatcher must retain Linux, IBM, and RISC-V native operations"
if grep -Ein 'riscv' "$RAIL_VARIANTS" >/dev/null; then
  fail "the Cargo Rail CI catalog must not own physical RISC-V work"
fi
if grep -Ein 'riscv' "$WORKFLOWS/bench.yaml" >/dev/null; then
  fail "the generic benchmark workflow must not expose RISC-V"
fi
[[ $(yq eval '.jobs.native.with.runner' "$RISCV") == "ubuntu-24.04-riscv" ]] \
  || fail "the RISC-V workflow must own the RISE native runner"
[[ $(yq eval '.jobs.ct.with.platforms' "$RISCV") == "rise-riscv" ]] \
  || fail "the RISC-V workflow must select only the RISE CT lane"
[[ $(yq eval '.on.schedule' "$RISCV") == "null" ]] \
  || fail "standalone RISC-V diagnostics must not duplicate scheduled Qualification evidence"
[[ $(yq eval '.jobs.ct.with.upload_raw_artifacts' "$RISCV") == "true" ]] \
  || fail "manual RISC-V evidence must retain raw CT artifacts"
[[ $(yq eval '.jobs.ct.with.artifact_retention_days' "$RISCV") == "90" ]] \
  || fail "manual RISC-V evidence must retain release-grade artifacts"
[[ $(jq '[.variants[] | select(.dimensions.operation == "cargo-graph")] | length' "$RAIL_VARIANTS") -eq 1 ]] \
  || fail "the Cargo Rail catalog must have exactly one Cargo graph assurance owner"
[[ $(count_matches 'cargo rail unify --check' "$RUN_RUST_JOB") -eq 1 ]] \
  || fail "the Rust job dispatcher must define exactly one Cargo graph assurance operation"
if grep -En 'cargo rail unify --check' "$RELEASE_PREFLIGHT" >/dev/null; then
  fail "tag preflight must consume exact-commit Qualification graph assurance instead of repeating it"
fi
if grep -En 'cargo (deny|audit)' "$RELEASE_PREFLIGHT" >/dev/null; then
  fail "tag preflight must consume exact-commit Qualification dependency evidence instead of repeating it"
fi
[[ $(yq -o=json -I=0 '.on.workflow_dispatch.inputs.mode.options' "$WEEKLY") == '["assurance","release"]' ]] \
  || fail "Qualification must expose only assurance and release modes"
[[ $(yq eval '.on.workflow_dispatch.inputs.mode.default' "$WEEKLY") == "assurance" ]] \
  || fail "manually dispatched Qualification runs must default to assurance"
[[ $(yq eval '.concurrency.group' "$WEEKLY") == \
  "\${{ github.workflow }}-\${{ github.ref }}-\${{ github.event_name == 'workflow_dispatch' && inputs.mode == 'release' && 'release' || 'assurance' }}" ]] \
  || fail "scheduled assurance must not cancel release qualification"
weekly_mode_script=$(yq eval '.jobs.mode.steps[] | select(.id == "mode") | .run' "$WEEKLY")
[[ "$weekly_mode_script" == *$'schedule)\n    mode=assurance'* ]] \
  || fail "scheduled Qualification runs must resolve to assurance"
[[ $(yq eval '.jobs.suite.with.supply_chain_mode' "$WEEKLY") == \
  "\${{ needs.mode.outputs.mode == 'release' && 'full' || 'light' }}" ]] \
  || fail "Qualification supply-chain depth must derive from the resolved mode"
[[ $(yq eval '.jobs."rail-plan".steps[] | select(.id == "rail") | .with.all' "$WEEKLY") == "true" ]] \
  || fail "Qualification must execute one typed all-work Cargo Rail plan"
[[ $(yq eval '.jobs.ct.with.upload_raw_artifacts' "$WEEKLY") == \
  "\${{ needs.mode.outputs.mode == 'release' }}" ]] \
  || fail "only release-mode Qualification may upload raw CT artifacts"
retention_expression="\${{ needs.mode.outputs.mode == 'release' && 90 || 14 }}"
[[ $(yq eval '.jobs.suite.with.artifact_retention_days' "$WEEKLY") == "$retention_expression" ]] \
  || fail "Qualification Cargo graph retention must derive from the resolved mode"
[[ $(yq eval '.jobs.fuzzing.with.artifact_retention_days' "$WEEKLY") == "$retention_expression" ]] \
  || fail "Qualification fuzz retention must derive from the resolved mode"
[[ $(yq eval '.jobs.mlkem-graviton.with.artifact_retention_days' "$WEEKLY") == "$retention_expression" ]] \
  || fail "Qualification ML-KEM retention must derive from the resolved mode"
[[ $(yq eval '.jobs.ct.with.artifact_retention_days' "$WEEKLY") == "$retention_expression" ]] \
  || fail "Qualification CT retention must derive from the resolved mode"
[[ $(yq eval '.jobs.rsa.with.artifact_retention_days' "$WEEKLY") == "$retention_expression" ]] \
  || fail "Qualification RSA retention must derive from the resolved mode"
[[ $(yq eval '.jobs.riscv-ct.with.artifact_retention_days' "$WEEKLY") == "$retention_expression" ]] \
  || fail "Qualification RISC-V CT retention must derive from the resolved mode"
[[ $(yq eval '.jobs.riscv-ct.with.upload_raw_artifacts' "$WEEKLY") == \
  "\${{ needs.mode.outputs.mode == 'release' }}" ]] \
  || fail "only release-mode Qualification may upload raw RISC-V CT artifacts"
[[ $(yq eval '.jobs.riscv-ct.with.platforms' "$WEEKLY") == "rise-riscv" ]] \
  || fail "Qualification must select the RISE CT evidence variant"
[[ $(yq eval '.jobs.riscv-native.with.runner' "$WEEKLY") == "ubuntu-24.04-riscv" ]] \
  || fail "Qualification must select the RISE native evidence runner"
[[ $(yq eval '.jobs.coverage.steps[] | select(.name == "Upload Coverage Artifacts") | .with."retention-days"' "$WEEKLY") == "$retention_expression" ]] \
  || fail "Qualification coverage retention must derive from the resolved mode"
[[ $(yq eval '.jobs.complete.name' "$WEEKLY") == 'Complete (${{ needs.mode.outputs.mode }})' ]] \
  || fail "Qualification must expose a mode-specific terminal gate"
ct_artifact_name="ct-\${{ inputs.upload_raw_artifacts && 'raw-' || '' }}\${{ matrix.artifact_suffix }}"
[[ $(yq eval '.jobs.ct.with.artifact_name' "$CT") == "$ct_artifact_name" ]] \
  || fail "raw CT artifact names must be distinguishable before release"
[[ $(yq eval '.on.workflow_call.inputs.artifact_retention_days.default' "$RUST_JOB") == "90" ]] \
  || fail "reusable Rust artifacts must preserve long retention by default"
artifact_steps='[.jobs.run.steps[] | select(.name == "Upload Artifact after completion" or .name == "Upload Artifact (success)")]'
[[ $(yq eval "$artifact_steps | length" "$RUST_JOB") -eq 2 ]] \
  || fail "reusable Rust job must retain both artifact upload paths"
[[ $(yq eval "$artifact_steps | map(.with.\"retention-days\" == \"\${{ inputs.artifact_retention_days }}\") | all" "$RUST_JOB") == "true" ]] \
  || fail "reusable Rust artifact retention must be caller-controlled"
[[ $(yq eval "$artifact_steps | map(.with.\"if-no-files-found\" == \"error\") | all" "$RUST_JOB") == "true" ]] \
  || fail "declared Rust evidence artifacts must fail closed when absent"
grep -Fq 'CI Suite (release) / Cargo Graph Assurance / run' "$RELEASE_EVIDENCE" \
  || fail "release evidence must require release-mode Cargo Graph Assurance"
grep -Fq 'Constant-Time Evidence (release) / Complete (CT)' "$RELEASE_EVIDENCE" \
  || fail "release evidence must require release-mode CT completion"
grep -Fq 'RSA Evidence (release) / Complete (RSA)' "$RELEASE_EVIDENCE" \
  || fail "release evidence must require release-mode RSA completion"
grep -Fq 'RISC-V Native Evidence / run' "$RELEASE_EVIDENCE" \
  || fail "release evidence must require RISC-V native qualification"
grep -Fq 'RISC-V CT Evidence (release) / Complete (CT)' "$RELEASE_EVIDENCE" \
  || fail "release evidence must require RISC-V CT qualification"
grep -Fq 'Complete (release)' "$RELEASE_EVIDENCE" \
  || fail "release evidence must require the release-mode terminal gate"
grep -Fq '.event == "workflow_dispatch"' "$RELEASE_EVIDENCE" \
  || fail "scheduled runs must be ineligible for release evidence"
grep -Fq 'ct-raw-' "$RELEASE_EVIDENCE" \
  || fail "release evidence must require live raw CT artifacts"
[[ $(yq eval '.concurrency.group' "$RSA") == 'rsa-${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}' ]] \
  || fail "reusable RSA workflow concurrency must not collide with its caller"
grep -Fq 'pattern: ct-raw-*' "$RELEASE" \
  || fail "release must download the complete raw qualification evidence set"
grep -Fq 'scripts/ci/release-evidence-check.sh --commit "$RELEASE_COMMIT"' "$RELEASE" \
  || fail "release must require one exact-commit Qualification run"
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
[[ $(yq eval '.jobs | length' "$RELEASE") -eq 2 ]] \
  || fail "release must use one preflight and one publish job"
[[ $(yq eval '.concurrency.cancel-in-progress' "$RELEASE") == "false" ]] \
  || fail "release publication must not be canceled after side effects begin"
evidence_step=$(yq eval '.jobs.preflight.steps | to_entries | .[] | select(.value.id == "evidence") | .key' "$RELEASE")
setup_step=$(yq eval '.jobs.preflight.steps | to_entries | .[] | select(.value.name == "Setup") | .key' "$RELEASE")
[[ "$evidence_step" =~ ^[0-9]+$ && "$setup_step" =~ ^[0-9]+$ && "$evidence_step" -lt "$setup_step" ]] \
  || fail "release evidence must fail before expensive preflight setup"
grep -Fq 'run-id: ${{ needs.preflight.outputs.qualification_run_id }}' "$RELEASE" \
  || fail "release must consume all CT artifacts from the validated Qualification run"
grep -Fq 'run-id: ${{ needs.preflight.outputs.s390x_ct_run_id }}' "$RELEASE" \
  || fail "release must consume recovered s390x CT artifacts from the validated recovery run"
grep -Fq 'run-id: ${{ needs.preflight.outputs.x86_64_ct_run_id }}' "$RELEASE" \
  || fail "release must consume recovered x86_64 CT artifacts from the validated recovery run"
[[ $(yq eval '.on.workflow_dispatch.inputs.release_tag.type' "$CT") == "string" ]] \
  || fail "CT recovery release tag input must be a string"
ct_source_step=$(yq eval '.jobs.plan.steps[] | select(.name == "Resolve CT source") | .run' "$CT")
grep -Fq 'refs/heads/main' <<<"$ct_source_step" \
  || fail "release CT recovery must reject workflow code outside protected main"
grep -Fq 'checkout_ref=$RELEASE_TAG' <<<"$ct_source_step" \
  || fail "release CT recovery must bind execution to the immutable tag"
grep -Fq 'amd-zen4,intel-spr,intel-icl,amd-zen5)' <<<"$ct_source_step" \
  || fail "release CT recovery must require the complete x86_64 platform group"
grep -Fq 'ibm-s390x)' <<<"$ct_source_step" \
  || fail "release CT recovery must retain the complete s390x platform group"
if grep -Fq 'v0.9.0' <<<"$ct_source_step"; then
  fail "completed v0.9.0 recovery compatibility must not remain in the live workflow"
fi
grep -Fq 'DUDECT_TIMEOUT" != "1800"' <<<"$ct_source_step" \
  || fail "release CT recovery must preserve the release DudeCT timeout"
grep -Fq 'BINSEC_TIMEOUT" != "900"' <<<"$ct_source_step" \
  || fail "release CT recovery must preserve the release BINSEC timeout"
grep -Fq 'UPLOAD_RAW_ARTIFACTS" != "true"' <<<"$ct_source_step" \
  || fail "release CT recovery must retain raw evidence"
grep -Fq 'ARTIFACT_RETENTION_DAYS" != "90"' <<<"$ct_source_step" \
  || fail "release CT recovery must retain evidence for the release lifetime"
# shellcheck disable=SC2016 # GitHub expression is an intentional literal workflow contract.
[[ $(yq eval '.jobs.ct.with.checkout_ref' "$CT") == '${{ needs.plan.outputs.checkout_ref }}' ]] \
  || fail "release CT recovery must execute the resolved immutable tag source"
# shellcheck disable=SC2016 # GitHub expression is an intentional literal workflow contract.
[[ $(yq eval '.jobs.ct.with.rustflags' "$CT") == '${{ needs.plan.outputs.recovery_rustflags }}' ]] \
  || fail "release CT recovery must pass only plan-resolved rustflags"
for input_name in dudect_timeout binsec_timeout artifact_retention_days; do
  expected="\${{ fromJSON(format('{0}', inputs.${input_name})) }}"
  [[ $(yq eval ".jobs.ct.with.${input_name}" "$CT") == "$expected" ]] \
    || fail "CT must normalize manual $input_name before the typed reusable workflow"
done
if grep -Eq 'uses: ./\.github/workflows/(ct|rsa)\.yaml' "$RELEASE"; then
  fail "tag workflow must promote exact-commit evidence instead of rerunning CT or RSA"
fi
ci_musl=$(jq '[.variants[] | select(.dimensions.operation == "native" and ((.dimensions.target // "") | contains("musl")))] | length' "$RAIL_VARIANTS")
[[ "$ci_musl" -eq 0 ]] || fail "MUSL targets must not masquerade as native host jobs"

ci_linux=$(jq '[.variants[] | select(.dimensions.operation == "native" and ((.dimensions.target // "") | endswith("unknown-linux-gnu")))] | length' "$RAIL_VARIANTS")
[[ "$ci_linux" -eq 2 ]] || fail "native CI must contain exactly x86_64 and AArch64 GNU hosts"

group_musl=$(jq '[.groups.linux[] | select(contains("musl"))] | length' "$MANIFEST")
[[ "$group_musl" -eq 2 ]] || fail "the target manifest must retain both MUSL triples"

# shellcheck disable=SC2016 # `$target` is an intentional literal in the workflow contract regex.
[[ $(count_matches 'cargo (check|clippy|build) --locked --target "\$target"' "$CROSS_SCRIPT") -ge 3 ]] \
  || fail "MUSL evidence must pass the target triple explicitly to Cargo"

echo "CI ownership contract passed"
