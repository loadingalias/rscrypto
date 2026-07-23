#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKER="$SCRIPT_DIR/check-ci-ownership.sh"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

make_fixture() {
  local fixture=$1
  mkdir -p "$fixture/.github" "$fixture/.config" "$fixture/scripts/check" "$fixture/scripts/lib" "$fixture/scripts/test"
  cp -R "$REPO_ROOT/.github/workflows" "$fixture/.github/workflows"
  cp -R "$REPO_ROOT/.github/actions" "$fixture/.github/actions"
  cp -R "$REPO_ROOT/.github/rulesets" "$fixture/.github/rulesets"
  cp -R "$REPO_ROOT/.github/repository-settings" "$fixture/.github/repository-settings"
  cp "$REPO_ROOT/.github/dependabot.yaml" "$fixture/.github/dependabot.yaml"
  cp "$REPO_ROOT/.config/target-matrix.json" "$fixture/.config/target-matrix.json"
  cp "$REPO_ROOT/.config/ci-tool-archives.tsv" "$fixture/.config/ci-tool-archives.tsv"
  cp -R "$REPO_ROOT/scripts/ci" "$fixture/scripts/ci"
  cp "$REPO_ROOT/scripts/lib/ci-tool-integrity.sh" "$REPO_ROOT/scripts/lib/common.sh" \
    "$fixture/scripts/lib/"
  cp "$REPO_ROOT/scripts/check/check-all.sh" "$REPO_ROOT/scripts/check/check-feature-matrix.sh" \
    "$REPO_ROOT/scripts/check/check.sh" "$fixture/scripts/check/"
  cp "$REPO_ROOT/scripts/test/test-feature-matrix.sh" "$fixture/scripts/test/"
}

expect_failure() {
  local fixture=$1
  local description=$2
  if "$CHECKER" --root "$fixture" >/dev/null 2>&1; then
    echo "expected ownership failure: $description" >&2
    exit 1
  fi
}

baseline="$TMP_ROOT/baseline"
make_fixture "$baseline"
"$CHECKER" --root "$baseline" >/dev/null

invalid_tool_digest="$TMP_ROOT/invalid-tool-digest"
make_fixture "$invalid_tool_digest"
sed -i.bak 's/ca1d64196d2d34771084afe76ea657d581bf628e31d993ff8e52ea09cc88a56d/not-a-digest/' \
  "$invalid_tool_digest/.config/ci-tool-archives.tsv"
rm -f "$invalid_tool_digest/.config/ci-tool-archives.tsv.bak"
expect_failure "$invalid_tool_digest" "direct tool digest is malformed"

mutable_tool_url="$TMP_ROOT/mutable-tool-url"
make_fixture "$mutable_tool_url"
sed -i.bak 's#/download/v46\.0\.1/#/download/Latest/#' \
  "$mutable_tool_url/.config/ci-tool-archives.tsv"
rm -f "$mutable_tool_url/.config/ci-tool-archives.tsv.bak"
expect_failure "$mutable_tool_url" "direct tool URL resolves a mutable release"

unexpected_tool_filename="$TMP_ROOT/unexpected-tool-filename"
make_fixture "$unexpected_tool_filename"
awk -F '\t' -v OFS='\t' '$1 == "codecov" { $5 = "codecov-substitute" } { print }' \
  "$unexpected_tool_filename/.config/ci-tool-archives.tsv" \
  >"$unexpected_tool_filename/.config/ci-tool-archives.tsv.tmp"
mv "$unexpected_tool_filename/.config/ci-tool-archives.tsv.tmp" \
  "$unexpected_tool_filename/.config/ci-tool-archives.tsv"
expect_failure "$unexpected_tool_filename" "direct tool URL and filename disagree"

unauthenticated_cargo_installer="$TMP_ROOT/unauthenticated-cargo-installer"
make_fixture "$unauthenticated_cargo_installer"
printf '\ncargo binstall cargo-nextest\n' \
  >>"$unauthenticated_cargo_installer/scripts/ci/install-tools.sh"
expect_failure "$unauthenticated_cargo_installer" "Cargo-binstall bypasses package integrity"

poisonable_tool_cache="$TMP_ROOT/poisonable-tool-cache"
make_fixture "$poisonable_tool_cache"
yq eval '.runs.steps += [{"name": "Restore poisonable tools", "uses": "actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9", "with": {"path": "~/.cargo/bin", "key": "known"}}]' -i \
  "$poisonable_tool_cache/.github/actions/setup/action.yaml"
expect_failure "$poisonable_tool_cache" "CI tool executables can be restored from a poisonable cache"

rust_cache_binaries="$TMP_ROOT/rust-cache-binaries"
make_fixture "$rust_cache_binaries"
yq eval '(.runs.steps[] | select(.name == "Setup Rust Cache") | .with."cache-bin") = true' -i \
  "$rust_cache_binaries/.github/actions/setup/action.yaml"
expect_failure "$rust_cache_binaries" "the Rust build cache can restore Cargo tool executables"

unauthenticated_rustup="$TMP_ROOT/unauthenticated-rustup"
make_fixture "$unauthenticated_rustup"
printf '\n    - uses: dtolnay/rust-toolchain@e97e2d8cc328f1b50210efc529dca0028893a2d9\n' \
  >>"$unauthenticated_rustup/.github/actions/setup-toolchain/action.yaml"
expect_failure "$unauthenticated_rustup" "toolchain setup can run a network bootstrap installer"

unpinned_scorecard="$TMP_ROOT/unpinned-scorecard"
make_fixture "$unpinned_scorecard"
sed -i.bak 's#@sha256:[0-9a-f]*#:v2.4.3#' \
  "$unpinned_scorecard/.github/actions/scorecard/action.yaml"
rm -f "$unpinned_scorecard/.github/actions/scorecard/action.yaml.bak"
expect_failure "$unpinned_scorecard" "Scorecard container uses a mutable tag"

floating_codecov="$TMP_ROOT/floating-codecov"
make_fixture "$floating_codecov"
sed -i.bak '/binary:.*steps\.codecov\.outputs\.binary/d' \
  "$floating_codecov/.github/workflows/weekly.yaml"
rm -f "$floating_codecov/.github/workflows/weekly.yaml.bak"
expect_failure "$floating_codecov" "Codecov action can download its floating default CLI"

unowned_download="$TMP_ROOT/unowned-download"
make_fixture "$unowned_download"
printf '\ncurl --output /tmp/tool https://example.invalid/tool\n' \
  >>"$unowned_download/scripts/ci/run-rust-job.sh"
expect_failure "$unowned_download" "a direct download exists outside the integrity owner"

unowned_package_install="$TMP_ROOT/unowned-package-install"
make_fixture "$unowned_package_install"
printf '\ncargo install ripgrep\n' \
  >>"$unowned_package_install/scripts/ci/run-rust-job.sh"
expect_failure "$unowned_package_install" "a package-manager install exists outside the integrity owner"

unowned_test_download="$TMP_ROOT/unowned-test-download"
make_fixture "$unowned_test_download"
printf '#!/usr/bin/env bash\ncurl --output /tmp/tool https://example.invalid/tool\n' \
  >"$unowned_test_download/scripts/ci/unreviewed-test.sh"
expect_failure "$unowned_test_download" "a test script bypasses downloader ownership"

missing_action_updates="$TMP_ROOT/missing-action-updates"
make_fixture "$missing_action_updates"
yq eval 'del(.updates[] | select(."package-ecosystem" == "github-actions"))' -i \
  "$missing_action_updates/.github/dependabot.yaml"
expect_failure "$missing_action_updates" "GitHub Actions updates are disabled"

push_ci="$TMP_ROOT/push-ci"
make_fixture "$push_ci"
yq eval '.on.push.branches = ["main"]' -i "$push_ci/.github/workflows/ci.yaml"
expect_failure "$push_ci" "duplicated post-merge CI"

missing_ready_event="$TMP_ROOT/missing-ready-event"
make_fixture "$missing_ready_event"
yq eval 'del(.on.pull_request.types[] | select(. == "ready_for_review"))' -i \
  "$missing_ready_event/.github/workflows/ci.yaml"
expect_failure "$missing_ready_event" "draft PR cannot start CI when marked ready"

draft_runs_suite="$TMP_ROOT/draft-runs-suite"
make_fixture "$draft_runs_suite"
yq eval '.jobs.suite."if" = "always()"' -i "$draft_runs_suite/.github/workflows/ci.yaml"
expect_failure "$draft_runs_suite" "draft PR can run the expensive suite"

planner_failure_skips_suite="$TMP_ROOT/planner-failure-skips-suite"
make_fixture "$planner_failure_skips_suite"
yq eval '.jobs.suite."if" = "github.event_name != '\''pull_request'\'' || !github.event.pull_request.draft"' -i \
  "$planner_failure_skips_suite/.github/workflows/ci.yaml"
expect_failure "$planner_failure_skips_suite" "planner failure can skip the CI suite"

missing_plan_resolver="$TMP_ROOT/missing-plan-resolver"
make_fixture "$missing_plan_resolver"
rm "$missing_plan_resolver/scripts/ci/resolve-rail-plan.sh"
expect_failure "$missing_plan_resolver" "workflow plan outputs bypass repository validation"

unchecked_suite_skip="$TMP_ROOT/unchecked-suite-skip"
make_fixture "$unchecked_suite_skip"
sed -i.bak '/if \[\[ "\$PLAN_VALID" != "true" || "\$PLAN_EMPTY" != "true" \]\]/d' \
  "$unchecked_suite_skip/.github/workflows/ci.yaml"
rm -f "$unchecked_suite_skip/.github/workflows/ci.yaml.bak"
expect_failure "$unchecked_suite_skip" "Complete accepts an unvalidated suite skip"

shell_fragment_input="$TMP_ROOT/shell-fragment-input"
make_fixture "$shell_fragment_input"
printf '\n      run_script: echo caller-controlled\n' \
  >>"$shell_fragment_input/.github/workflows/_ci-suite.yaml"
expect_failure "$shell_fragment_input" "reusable workflow accepts executable shell fragments"

missing_typed_operation="$TMP_ROOT/missing-typed-operation"
make_fixture "$missing_typed_operation"
sed -i.bak '/operation: quality/d' "$missing_typed_operation/.github/workflows/_ci-suite.yaml"
rm -f "$missing_typed_operation/.github/workflows/_ci-suite.yaml.bak"
expect_failure "$missing_typed_operation" "reusable Rust job caller omits its operation"

unsupported_typed_operation="$TMP_ROOT/unsupported-typed-operation"
make_fixture "$unsupported_typed_operation"
sed -i.bak 's/operation: quality/operation: arbitrary-shell/' \
  "$unsupported_typed_operation/.github/workflows/_ci-suite.yaml"
rm -f "$unsupported_typed_operation/.github/workflows/_ci-suite.yaml.bak"
expect_failure "$unsupported_typed_operation" "reusable Rust job caller selects an unsupported operation"

evaluated_workflow_input="$TMP_ROOT/evaluated-workflow-input"
make_fixture "$evaluated_workflow_input"
sed -i.bak 's#run: scripts/ci/run-rust-job.sh#run: echo "${{ inputs.operation }}"#' \
  "$evaluated_workflow_input/.github/workflows/_rust-job.yaml"
rm -f "$evaluated_workflow_input/.github/workflows/_rust-job.yaml.bak"
expect_failure "$evaluated_workflow_input" "workflow input is evaluated as shell code"

duplicate_matrix="$TMP_ROOT/duplicate-matrix"
make_fixture "$duplicate_matrix"
printf '\n# duplicate owner\n      run: just test-feature-matrix\n' >>"$duplicate_matrix/.github/workflows/weekly.yaml"
expect_failure "$duplicate_matrix" "duplicate feature matrix"

native_cross_sweep="$TMP_ROOT/native-cross-sweep"
make_fixture "$native_cross_sweep"
printf '\n# forbidden native sweep\n      run: just check --all\n' >>"$native_cross_sweep/.github/workflows/_ci-suite.yaml"
expect_failure "$native_cross_sweep" "comprehensive check in native workflow"

fake_musl="$TMP_ROOT/fake-musl"
make_fixture "$fake_musl"
jq '.ci += [{"name":"x86_64-unknown-linux-musl","type":"runson","pool":"linux-x64-ci"}]' \
  "$fake_musl/.config/target-matrix.json" >"$fake_musl/.config/target-matrix.json.tmp"
mv "$fake_musl/.config/target-matrix.json.tmp" "$fake_musl/.config/target-matrix.json"
expect_failure "$fake_musl" "MUSL label without a MUSL target invocation"

missing_cross_owner="$TMP_ROOT/missing-cross-owner"
make_fixture "$missing_cross_owner"
sed -i.bak '/operation: cross-targets/d' "$missing_cross_owner/.github/workflows/_ci-suite.yaml"
rm -f "$missing_cross_owner/.github/workflows/_ci-suite.yaml.bak"
expect_failure "$missing_cross_owner" "missing cross-target owner"

missing_graph_owner="$TMP_ROOT/missing-graph-owner"
make_fixture "$missing_graph_owner"
sed -i.bak '/operation: cargo-graph/d' "$missing_graph_owner/.github/workflows/_ci-suite.yaml"
rm -f "$missing_graph_owner/.github/workflows/_ci-suite.yaml.bak"
expect_failure "$missing_graph_owner" "missing Cargo graph assurance owner"

duplicate_release_graph="$TMP_ROOT/duplicate-release-graph"
make_fixture "$duplicate_release_graph"
printf '\n# duplicate release owner\ncargo rail unify --check --explain\n' \
  >>"$duplicate_release_graph/scripts/ci/release-preflight.sh"
expect_failure "$duplicate_release_graph" "duplicate release Cargo graph assurance owner"

missing_release_graph_gate="$TMP_ROOT/missing-release-graph-gate"
make_fixture "$missing_release_graph_gate"
sed -i.bak '/CI Suite (weekly) \/ Cargo Graph Assurance \/ run/d' \
  "$missing_release_graph_gate/scripts/ci/release-evidence-check.sh"
rm -f "$missing_release_graph_gate/scripts/ci/release-evidence-check.sh.bak"
expect_failure "$missing_release_graph_gate" "missing release Cargo graph assurance gate"

colliding_rsa_concurrency="$TMP_ROOT/colliding-rsa-concurrency"
make_fixture "$colliding_rsa_concurrency"
sed -i.bak 's/group: rsa-/group: /' "$colliding_rsa_concurrency/.github/workflows/rsa.yaml"
rm -f "$colliding_rsa_concurrency/.github/workflows/rsa.yaml.bak"
expect_failure "$colliding_rsa_concurrency" "reusable RSA workflow concurrency collision"

missing_release_evidence_gate="$TMP_ROOT/missing-release-evidence-gate"
make_fixture "$missing_release_evidence_gate"
sed -i.bak '/release-evidence-check\.sh/d' "$missing_release_evidence_gate/.github/workflows/release.yaml"
rm -f "$missing_release_evidence_gate/.github/workflows/release.yaml.bak"
expect_failure "$missing_release_evidence_gate" "missing exact-commit release evidence gate"

missing_repository_controls="$TMP_ROOT/missing-repository-controls"
make_fixture "$missing_repository_controls"
sed -i.bak '/repository-controls-evidence\.sh/d' "$missing_repository_controls/.github/workflows/release.yaml"
rm -f "$missing_repository_controls/.github/workflows/release.yaml.bak"
expect_failure "$missing_repository_controls" "release without repository controls evidence"

missing_repository_policy="$TMP_ROOT/missing-repository-policy"
make_fixture "$missing_repository_policy"
rm "$missing_repository_policy/.github/rulesets/protect-main.json"
expect_failure "$missing_repository_policy" "missing repository ruleset policy"

missing_release_tag_policy="$TMP_ROOT/missing-release-tag-policy"
make_fixture "$missing_release_tag_policy"
rm "$missing_release_tag_policy/.github/rulesets/protect-release-tags.json"
expect_failure "$missing_release_tag_policy" "missing release-tag ruleset policy"

missing_release_immutability_policy="$TMP_ROOT/missing-release-immutability-policy"
make_fixture "$missing_release_immutability_policy"
rm "$missing_release_immutability_policy/.github/repository-settings/release-immutability.json"
expect_failure "$missing_release_immutability_policy" "missing release immutability policy"

missing_source_archive="$TMP_ROOT/missing-source-archive"
make_fixture "$missing_source_archive"
sed -i.bak '/package-release-source\.sh/d' "$missing_source_archive/scripts/ci/release-preflight.sh"
rm -f "$missing_source_archive/scripts/ci/release-preflight.sh.bak"
expect_failure "$missing_source_archive" "release without deterministic source archive"

missing_release_manifest="$TMP_ROOT/missing-release-manifest"
make_fixture "$missing_release_manifest"
sed -i.bak '/write-release-manifest\.sh/d' "$missing_release_manifest/.github/workflows/release.yaml"
rm -f "$missing_release_manifest/.github/workflows/release.yaml.bak"
expect_failure "$missing_release_manifest" "release without identity manifest"

missing_riscv_workflow="$TMP_ROOT/missing-riscv-workflow"
make_fixture "$missing_riscv_workflow"
rm "$missing_riscv_workflow/.github/workflows/riscv.yaml"
expect_failure "$missing_riscv_workflow" "missing independent RISC-V workflow"

missing_riscv_release_artifact="$TMP_ROOT/missing-riscv-release-artifact"
make_fixture "$missing_riscv_release_artifact"
sed -i.bak '/needs\.evidence-gate\.outputs\.riscv_run_id/d' \
  "$missing_riscv_release_artifact/.github/workflows/release.yaml"
rm -f "$missing_riscv_release_artifact/.github/workflows/release.yaml.bak"
expect_failure "$missing_riscv_release_artifact" "release without validated RISC-V artifacts"

compact_weekly_ct="$TMP_ROOT/compact-weekly-ct"
make_fixture "$compact_weekly_ct"
sed -i.bak 's/upload_raw_artifacts: true/upload_raw_artifacts: false/' "$compact_weekly_ct/.github/workflows/weekly.yaml"
rm -f "$compact_weekly_ct/.github/workflows/weekly.yaml.bak"
expect_failure "$compact_weekly_ct" "Weekly without raw release CT evidence"

compact_riscv_ct="$TMP_ROOT/compact-riscv-ct"
make_fixture "$compact_riscv_ct"
sed -i.bak 's/upload_raw_artifacts: true/upload_raw_artifacts: false/' \
  "$compact_riscv_ct/.github/workflows/riscv.yaml"
rm -f "$compact_riscv_ct/.github/workflows/riscv.yaml.bak"
expect_failure "$compact_riscv_ct" "RISC-V without raw release CT evidence"

riscv_leaked_into_weekly="$TMP_ROOT/riscv-leaked-into-weekly"
make_fixture "$riscv_leaked_into_weekly"
printf '\n# riscv physical lane leaked back into Weekly\n' >>"$riscv_leaked_into_weekly/.github/workflows/weekly.yaml"
expect_failure "$riscv_leaked_into_weekly" "RISC-V coupling in Weekly"

broken_dependabot_grouping="$TMP_ROOT/broken-dependabot-grouping"
make_fixture "$broken_dependabot_grouping"
cat >>"$broken_dependabot_grouping/.github/dependabot.yaml" <<'EOF'
    groups:
      broken:
        group-by: dependency-name
EOF
expect_failure "$broken_dependabot_grouping" "broken cross-directory Dependabot grouping"

missing_fuzz_packages="$TMP_ROOT/missing-fuzz-packages"
make_fixture "$missing_fuzz_packages"
sed -i.bak '/fuzz-packages/d' "$missing_fuzz_packages/.github/dependabot.yaml"
rm -f "$missing_fuzz_packages/.github/dependabot.yaml.bak"
expect_failure "$missing_fuzz_packages" "incomplete Dependabot Cargo manifest coverage"

missing_tools="$TMP_ROOT/missing-tools"
make_fixture "$missing_tools"
sed -i.bak '/tools\/\*/d' "$missing_tools/.github/dependabot.yaml"
rm -f "$missing_tools/.github/dependabot.yaml.bak"
expect_failure "$missing_tools" "missing standalone tool dependency coverage"

duplicate_semver_owner="$TMP_ROOT/duplicate-semver-owner"
make_fixture "$duplicate_semver_owner"
printf '\n# duplicate owner\n      run: cargo semver-checks --package rscrypto --all-features\n' >>"$duplicate_semver_owner/.github/workflows/weekly.yaml"
expect_failure "$duplicate_semver_owner" "duplicate SemVer owner"

shrunk_matrix="$TMP_ROOT/shrunk-matrix"
make_fixture "$shrunk_matrix"
sed -i.bak '/  "crc16"/d' "$shrunk_matrix/scripts/test/test-feature-matrix.sh"
rm -f "$shrunk_matrix/scripts/test/test-feature-matrix.sh.bak"
expect_failure "$shrunk_matrix" "removed executable feature profile"

echo "CI ownership regression tests passed"
