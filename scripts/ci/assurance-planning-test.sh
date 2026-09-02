#!/usr/bin/env bash
# Prove affected assurance selection and its CI consumers without compiling product code.

set -euo pipefail
unset BASH_ENV

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

fail() {
  echo "assurance planning regression failure: $*" >&2
  exit 1
}

export GIT_INDEX_FILE="$TMP_ROOT/index"
export GIT_AUTHOR_NAME="rscrypto CI"
export GIT_AUTHOR_EMAIL="ci@rscrypto.invalid"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
export GIT_AUTHOR_DATE="2000-01-01T00:00:00Z"
export GIT_COMMITTER_DATE="$GIT_AUTHOR_DATE"

cd "$REPO_ROOT"
scripts/test/miri-contracts.sh list >/dev/null
scripts/test/fuzz-contracts.sh list >/dev/null
git read-tree HEAD
git add -A
base_tree=$(git write-tree)
base_commit=$(printf 'effective assurance planning fixture\n' | git commit-tree "$base_tree" -p HEAD)

plan_path() {
  local path=$1
  local label=$2
  local output=$3
  local mode blob tree commit

  git read-tree "$base_tree"
  mode=$(git ls-tree "$base_tree" -- "$path" | awk '{ print $1 }')
  [[ -n "$mode" ]] || fail "fixture path is absent: $path"
  blob=$(
    {
      git show "$base_commit:$path"
      case "$path" in
        *.lock | *.toml) printf '\n# assurance planning fixture: %s\n' "$label" ;;
        *) printf '\n// assurance planning fixture: %s\n' "$label" ;;
      esac
    } | git hash-object -w --stdin
  )
  git update-index --add --cacheinfo "$mode,$blob,$path"
  tree=$(git write-tree)
  commit=$(printf '%s assurance planning fixture\n' "$label" | git commit-tree "$tree" -p "$base_commit")
  cargo rail plan --quiet --from "$base_commit" --to "$commit" --json >"$output"
}

assert_decisions() {
  local plan=$1
  local ct_state=$2
  local rsa_state=$3
  jq -e \
    --arg ct "$ct_state" \
    --arg rsa "$rsa_state" \
    '.work["assurance.ct"].state == $ct and .work["assurance.rsa"].state == $rsa' \
    "$plan" >/dev/null || fail "unexpected CT/RSA decisions in $plan"
}

assert_state() {
  local plan=$1
  local work=$2
  local expected=$3
  jq -e --arg work "$work" --arg expected "$expected" '.work[$work].state == $expected' "$plan" >/dev/null \
    || fail "$work should be $expected in $plan"
}

assert_variants() {
  local plan=$1
  local work=$2
  local expected=$3

  if [[ "$expected" == skipped ]]; then
    jq -e --arg work "$work" '.work[$work].state == "skipped"' "$plan" >/dev/null \
      || fail "$work should be skipped in $plan"
    return
  fi
  if [[ "$expected" == all ]]; then
    jq -e --arg work "$work" '
      .work[$work].state == "required"
      and .work[$work].scope.kind == "variants"
      and .work[$work].scope.selection.kind == "all"
    ' "$plan" >/dev/null || fail "$work should select all variants in $plan"
    return
  fi

  local actual
  actual=$(jq -r --arg work "$work" '
    .work[$work]
    | select(.state == "required" and .scope.kind == "variants" and .scope.selection.kind == "selected")
    | [.scope.selection.variants[].id] | sort | join(",")
  ' "$plan")
  [[ "$actual" == "$expected" ]] \
    || fail "$work selected '${actual:-<invalid>}' instead of '$expected' in $plan"
}

rsa_plan="$TMP_ROOT/rsa.json"
plan_path src/auth/rsa_x86_64_asm.rs rsa "$rsa_plan"
assert_decisions "$rsa_plan" required required
assert_variants "$rsa_plan" assurance.miri skipped
assert_variants "$rsa_plan" assurance.fuzz rsa

rsa_core_plan="$TMP_ROOT/rsa-core.json"
plan_path src/auth/rsa.rs rsa-core "$rsa_core_plan"
assert_variants "$rsa_core_plan" assurance.miri rsa
assert_variants "$rsa_core_plan" assurance.fuzz rsa

aead_plan="$TMP_ROOT/aead.json"
plan_path src/aead/aes128gcm.rs aead "$aead_plan"
assert_decisions "$aead_plan" required skipped
assert_variants "$aead_plan" assurance.miri skipped
assert_variants "$aead_plan" assurance.fuzz aes-gcm

x25519_plan="$TMP_ROOT/x25519.json"
plan_path src/auth/x25519.rs x25519 "$x25519_plan"
assert_variants "$x25519_plan" assurance.miri portable
assert_variants "$x25519_plan" assurance.fuzz x25519

sha2_plan="$TMP_ROOT/sha2.json"
plan_path src/hashes/crypto/sha256/mod.rs sha2 "$sha2_plan"
assert_variants "$sha2_plan" assurance.miri skipped
assert_variants "$sha2_plan" assurance.fuzz sha2

crc32_plan="$TMP_ROOT/crc32.json"
plan_path src/checksum/crc32/mod.rs crc32 "$crc32_plan"
assert_variants "$crc32_plan" assurance.miri portable
assert_variants "$crc32_plan" assurance.fuzz crc32

scoped_fuzz_plan="$TMP_ROOT/scoped-fuzz.json"
plan_path fuzz-packages/auth-x25519/Cargo.toml scoped-fuzz "$scoped_fuzz_plan"
assert_variants "$scoped_fuzz_plan" assurance.miri skipped
assert_variants "$scoped_fuzz_plan" assurance.fuzz x25519

corpus_plan="$TMP_ROOT/corpus.json"
plan_path fuzz/corpus/auth_x25519/seed-basic corpus "$corpus_plan"
assert_variants "$corpus_plan" assurance.miri skipped
assert_variants "$corpus_plan" assurance.fuzz x25519

combined_checksum_plan="$TMP_ROOT/checksum-corpus.json"
plan_path fuzz/corpus/checksum_crc/seed-basic checksum-corpus "$combined_checksum_plan"
assert_variants "$combined_checksum_plan" assurance.miri skipped
assert_variants "$combined_checksum_plan" assurance.fuzz checksums-full

unattributed_plan="$TMP_ROOT/unattributed.json"
plan_path src/secret.rs unattributed "$unattributed_plan"
assert_variants "$unattributed_plan" assurance.miri skipped
assert_variants "$unattributed_plan" assurance.fuzz all

manifest_plan="$TMP_ROOT/manifest.json"
plan_path Cargo.toml manifest "$manifest_plan"
assert_decisions "$manifest_plan" required required
assert_state "$manifest_plan" contracts.cargo-graph required
assert_state "$manifest_plan" contracts.examples required
assert_variants "$manifest_plan" assurance.miri all
assert_variants "$manifest_plan" assurance.fuzz all

example_plan="$TMP_ROOT/example.json"
plan_path examples/aead_seal_open.rs example "$example_plan"
assert_state "$example_plan" contracts.cargo-graph skipped
assert_state "$example_plan" contracts.examples required

all_plan="$TMP_ROOT/all.json"
cargo rail plan --quiet --from "$base_commit" --to "$base_commit" --all --json >"$all_plan"
expected_all_work=$(
  printf '%s\n' \
    assurance.ct \
    assurance.fuzz \
    assurance.miri \
    assurance.rsa \
    cargo.build \
    cargo.clippy \
    cargo.doc \
    cargo.doctest \
    cargo.fmt \
    cargo.package \
    cargo.test \
    contracts.auxiliary \
    contracts.cargo-graph \
    contracts.examples \
    contracts.features \
    dependencies.auxiliary \
    dependency-policy \
    policy.actions \
    policy.repository \
    release.semver \
    surface \
    targets.platforms
)
actual_all_work=$(jq -r '.required[]' "$all_plan" | sort)
[[ "$actual_all_work" == "$expected_all_work" ]] \
  || fail "--all work inventory drifted"

lock_plan="$TMP_ROOT/lock.json"
plan_path Cargo.lock lock "$lock_plan"
assert_decisions "$lock_plan" skipped skipped
assert_state "$lock_plan" contracts.cargo-graph required
assert_state "$lock_plan" contracts.examples required
assert_variants "$lock_plan" assurance.miri skipped
assert_variants "$lock_plan" assurance.fuzz skipped

docs_plan="$TMP_ROOT/docs.json"
plan_path docs/features.md docs "$docs_plan"
assert_decisions "$docs_plan" skipped skipped
assert_state "$docs_plan" contracts.cargo-graph skipped
assert_state "$docs_plan" contracts.examples skipped

ruby - "$REPO_ROOT" <<'RUBY'
require "json"
require "yaml"

root = ARGV.fetch(0)
ci = YAML.safe_load(File.read(File.join(root, ".github/workflows/ci.yaml")), aliases: true)
jobs = ci.fetch("jobs")
plan_outputs = jobs.fetch("plan").fetch("outputs")
raise "missing CT plan output" unless plan_outputs.key?("ct")
raise "missing RSA plan output" unless plan_outputs.key?("rsa")
raise "missing test plan output" unless plan_outputs.key?("tests")
raise "missing examples plan output" unless plan_outputs.key?("examples")
raise "missing MSRV plan output" unless plan_outputs.key?("msrv")
%w[fuzz fuzz-rows miri miri-rows].each do |output|
  raise "missing #{output} plan output" unless plan_outputs.key?(output)
end

plan_steps = jobs.fetch("plan").fetch("steps")
select_run = plan_steps.find { |step| step["id"] == "select" }.fetch("run")
raise "CI plan does not select Actions policy setup" unless select_run.include?('echo "actions=$(required_any policy.actions)"')
raise "CI plan does not select minimum-feature examples" unless select_run.include?('echo "examples=$(required_any cargo.build contracts.examples)"')
raise "CI plan does not select MSRV from Cargo build impact" unless select_run.include?('echo "msrv=$(required_any cargo.build)"')
direct_policy_tools_install = plan_steps.find { |step| step["run"] == "scripts/ci/install-actions-policy-tools.sh" }
raise "CI does not install direct policy tools only for Actions policy" unless direct_policy_tools_install&.fetch("if") == "steps.select.outputs.actions == 'true'"
policy_tools_install = plan_steps.find do |step|
  step["uses"]&.start_with?("taiki-e/install-action@") && step.dig("with", "tool") == "just@1.58.0,zizmor@1.30.0"
end
raise "CI does not install pinned Just and Zizmor only for Actions policy" unless policy_tools_install&.fetch("if") == "steps.select.outputs.actions == 'true'"
raise "CI permits Actions policy tool source fallback" unless policy_tools_install.dig("with", "fallback") == "none"

actions_policy = File.read(File.join(root, "scripts/ci/actions-policy.sh"))
raise "Actions policy does not execute actionlint" unless actions_policy.match?(/^actionlint$/)
raise "Actions policy does not execute Zizmor" unless actions_policy.include?("zizmor .github/workflows .github/actions")

ct_runs = jobs.fetch("ct").fetch("steps").map { |step| step["run"] }.compact
rsa_runs = jobs.fetch("rsa").fetch("steps").map { |step| step["run"] }.compact
raise "CT job bypasses its plan decision" unless ct_runs.include?("scripts/ci/require-work.sh assurance.ct")
raise "CT job bypasses its repository command" unless ct_runs.include?("scripts/ct/structural.sh")
raise "RSA job bypasses its plan decision" unless rsa_runs.include?("scripts/ci/require-work.sh assurance.rsa")
raise "RSA job bypasses its repository command" unless rsa_runs.include?("scripts/test/test-rsa-linux-asm.sh")

miri_runs = jobs.fetch("miri").fetch("steps").map { |step| step["run"] }.compact
fuzz_runs = jobs.fetch("fuzz").fetch("steps").map { |step| step["run"] }.compact
raise "Miri job bypasses its plan decision" unless miri_runs.include?("scripts/ci/require-work.sh assurance.miri")
raise "Miri job bypasses selected rows" unless miri_runs.include?('scripts/test/miri-contracts.sh selected "$MIRI_ROWS"')
raise "Fuzz job bypasses its plan decision" unless fuzz_runs.include?("scripts/ci/require-work.sh assurance.fuzz")
raise "Fuzz job bypasses selected rows" unless fuzz_runs.include?('scripts/test/fuzz-contracts.sh selected "$FUZZ_ROWS"')

manifest = File.read(File.join(root, "Cargo.toml"))
unless manifest.match?(/^\[profile\.test\]\nopt-level = 1$/)
  raise "ordinary test execution lost its optimized Cargo profile"
end
miri_script = File.read(File.join(root, "scripts/test/test-miri.sh"))
unless miri_script.include?("export CARGO_PROFILE_TEST_OPT_LEVEL=0")
  raise "Miri no longer forces unoptimized test MIR"
end
coverage_script = File.read(File.join(root, "scripts/test/test-coverage.sh"))
unless coverage_script.include?("export CARGO_PROFILE_TEST_OPT_LEVEL=0")
  raise "coverage no longer forces unoptimized test code"
end

core_runs = jobs.fetch("core").fetch("steps").map { |step| step["run"] }.compact
raise "CI core lost minimum-feature examples" unless core_runs.include?("scripts/test/test-examples.sh")
msrv = jobs.fetch("msrv")
msrv_runs = msrv.fetch("steps").map { |step| step["run"] }.compact
raise "CI MSRV bypasses Cargo build selection" unless msrv_runs.include?("scripts/ci/require-work.sh cargo.build")
raise "CI MSRV bypasses its repository command" unless msrv_runs.include?("scripts/check/msrv.sh")
msrv_rust = msrv.fetch("steps").find { |step| step["uses"] == "$/.github/actions/rust" }
raise "CI MSRV does not install the declared contract" unless msrv_rust&.dig("with", "contract") == "msrv"

policy = File.read(File.join(root, "scripts/check/policy.sh"))
raise "repository policy does not own Cargo graph consistency" unless policy.include?("work_required contracts.cargo-graph") && policy.include?("cargo rail unify --check")

rust_action = YAML.safe_load(File.read(File.join(root, ".github/actions/rust/action.yaml")), aliases: true)
raise "Rust action does not expose cache activation" unless rust_action.fetch("outputs").key?("cache-enabled")

%w[core msrv features platforms rsa].each do |job_name|
  steps = jobs.fetch(job_name).fetch("steps")
  rust = steps.find { |step| step["id"] == "rust" && step["uses"] == "$/.github/actions/rust" }
  report = steps.find { |step| step["run"] == "scripts/ci/report-cache.sh" }
  raise "CI #{job_name} cache setup has no stable output identity" unless rust
  raise "CI #{job_name} lost post-run cache telemetry" unless report
end

needs = jobs.fetch("complete").fetch("needs")
raise "Complete omits CT" unless needs.include?("ct")
raise "Complete omits RSA" unless needs.include?("rsa")
raise "Complete omits Miri" unless needs.include?("miri")
raise "Complete omits fuzz" unless needs.include?("fuzz")
raise "Complete omits MSRV" unless needs.include?("msrv")

qualification = YAML.safe_load(File.read(File.join(root, ".github/workflows/qualification.yaml")), aliases: true)
ct_call = qualification.fetch("jobs").fetch("ct").fetch("with")
%w[head_commit plan_artifact plan_identity].each do |input|
  raise "Qualification CT omits #{input}" unless ct_call.key?(input)
end

release = YAML.safe_load(File.read(File.join(root, ".github/workflows/release.yaml")), aliases: true)
release_jobs = release.fetch("jobs")
expected_release_jobs = %w[qualification package publish]
unless release_jobs.keys == expected_release_jobs
  raise "Release DAG is not the minimal qualification/package join: #{release_jobs.keys.join(', ')}"
end
release_qualification = release_jobs.fetch("qualification")
unless release_qualification.fetch("uses") == "$/.github/workflows/qualification.yaml"
  raise "Release does not call exact-commit Qualification"
end
unless release_qualification.fetch("with") == {
  "head_commit" => "${{ github.sha }}",
  "mode" => "release",
}
  raise "Release Qualification inputs are not bound to the tag commit"
end
unless release_qualification.fetch("secrets").keys.sort == %w[
  CARGO_RAIL_R2_READ_ACCESS_KEY_ID
  CARGO_RAIL_R2_READ_SECRET_ACCESS_KEY
]
  raise "Release Qualification receives more than read-only cache credentials"
end
release_publish = release_jobs.fetch("publish")
unless release_publish.fetch("needs") == %w[qualification package]
  raise "Release publication does not join qualification and package results"
end
unless release_publish.fetch("environment") == "crates-io"
  raise "Release publication bypasses the crates-io environment"
end

qualification_miri = qualification.fetch("jobs").fetch("miri").fetch("steps").map { |step| step["run"] }.compact
raise "Qualification bypasses portable Miri row" unless qualification_miri.include?("scripts/test/miri-contracts.sh run portable")
raise "Qualification bypasses RSA Miri row" unless qualification_miri.include?("scripts/test/miri-contracts.sh run rsa")
qualification_fuzz = qualification.fetch("jobs").fetch("fuzz").fetch("steps").map { |step| step["run"] }.compact
raise "Qualification lost exhaustive fuzzing" unless qualification_fuzz.include?("scripts/test/test-fuzz.sh --all")
raise "Qualification lost exhaustive ASan replay" unless qualification_fuzz.include?("scripts/test/test-fuzz-asan.sh --all")

qualification_core_runs = qualification.fetch("jobs").fetch("core").fetch("steps").map { |step| step["run"] }.compact
raise "Qualification lost minimum-feature examples" unless qualification_core_runs.include?("scripts/test/test-examples.sh")
qualification_msrv = qualification.fetch("jobs").fetch("msrv")
qualification_msrv_runs = qualification_msrv.fetch("steps").map { |step| step["run"] }.compact
raise "Qualification lost MSRV execution" unless qualification_msrv_runs.include?("scripts/check/msrv.sh")

qualification_platforms = qualification.fetch("jobs").fetch("platforms").fetch("steps").map { |step| step["run"] }.compact
raise "Qualification lost deep native platform proof" unless qualification_platforms.include?('scripts/ci/target-contracts.sh run "$TARGET_ROW" deep')

target_catalog = JSON.parse(File.read(File.join(root, ".config/target-matrix.json")))
%w[aarch64-unknown-linux-gnu aarch64-apple-darwin].each do |id|
  row = target_catalog.fetch("variants").find { |candidate| candidate.fetch("id") == id }
  raise "missing native AArch64 proof row #{id}" unless row
  dimensions = row.fetch("dimensions")
  raise "#{id} is not native runtime proof" unless dimensions.fetch("operation") == "native"
  expected_paths = %w[tests/aead_kernel_equivalence.rs tests/portable_fallback.rs tests/vectored_dispatch.rs]
  missing_paths = expected_paths - row.fetch("external_paths")
  raise "#{id} lost differential proof paths: #{missing_paths.join(', ')}" unless missing_paths.empty?
end

{"CI" => ci, "Qualification" => qualification}.each do |name, workflow|
  installer = workflow.fetch("jobs").fetch("core").fetch("steps").find do |step|
    step["uses"]&.start_with?("taiki-e/install-action@")
  end
  raise "#{name} core does not install pinned Nextest" unless installer
  raise "#{name} core installs the wrong Nextest" unless installer.dig("with", "tool") == "cargo-nextest@0.9.143"
  raise "#{name} core permits source fallback" unless installer.dig("with", "fallback") == "none"
  if name == "CI"
    raise "CI installs Nextest for non-test plans" unless installer.fetch("if") == "needs.plan.outputs.tests == 'true'"
  end
end

coverage = qualification.fetch("jobs").fetch("coverage")
coverage_runs = coverage.fetch("steps").map { |step| step["run"] }.compact
raise "Coverage is not bound to cargo.test" unless coverage_runs.include?("scripts/ci/require-work.sh cargo.test")
raise "Coverage is not bound to assurance.fuzz" unless coverage_runs.include?("scripts/ci/require-work.sh assurance.fuzz")
raise "Qualification lost total coverage" unless coverage_runs.include?("scripts/test/test-coverage.sh")
coverage_installer = coverage.fetch("steps").find { |step| step["uses"]&.start_with?("taiki-e/install-action@") }
raise "Coverage tools are not installed from a pinned action" unless coverage_installer
unless coverage_installer.dig("with", "tool") == "cargo-llvm-cov@0.9.0,cargo-nextest@0.9.143"
  raise "Coverage installs tool versions that differ from the core test contract"
end
raise "Coverage tool fallback must remain disabled" unless coverage_installer.dig("with", "fallback") == "none"

zeroization = qualification.fetch("jobs").fetch("zeroization")
zeroization_runs = zeroization.fetch("steps").map { |step| step["run"] }.compact
raise "Zeroization is not bound to assurance.ct" unless zeroization_runs.include?("scripts/ci/require-work.sh assurance.ct")
raise "Qualification lost optimized zeroization" unless zeroization_runs.include?("scripts/check/zeroize-evidence.sh")
zeroization_rust = zeroization.fetch("steps").find { |step| step["uses"] == "$/.github/actions/rust" }
raise "Zeroization must not enable compiler reuse" unless zeroization_rust && !zeroization_rust.key?("with")

qualification_needs = qualification.fetch("jobs").fetch("complete").fetch("needs")
%w[coverage msrv zeroization].each do |job|
  raise "Qualification Complete omits #{job}" unless qualification_needs.include?(job)
end

rail_config = File.read(File.join(root, ".config/rail.toml"))
raise "Cargo Rail surface check must remain disabled" unless rail_config.match?(/\[surface\].*?enabled = false/m)
raise "pre-1.0 SemVer check must remain disabled" unless rail_config.match?(/\[release\].*?semver_check = "off"/m)
package_guard = File.read(File.join(root, "scripts/ci/release-package-guard.sh"))
raise "cargo.package lost its release-only executor" unless package_guard.include?("cargo package --locked")

mlkem_checkout = qualification.fetch("jobs").fetch("mlkem").fetch("steps").find do |step|
  step["uses"]&.start_with?("actions/checkout@")
end
expected_head = "${{ needs.plan.outputs.head-commit }}"
raise "Qualification ML-KEM is not pinned to the planned commit" unless mlkem_checkout&.dig("with", "ref") == expected_head

Dir[File.join(root, ".github/workflows/*.{yaml,yml}")].sort.each do |workflow_path|
  workflow = YAML.safe_load(File.read(workflow_path), aliases: true)
  workflow.fetch("jobs", {}).each do |job_name, job|
    next if job.key?("uses")
    raise "#{File.basename(workflow_path)} #{job_name} has no timeout" unless job.key?("timeout-minutes")
  end
end
RUBY

echo "Assurance planning regression tests passed"
