#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKER="$SCRIPT_DIR/check-ci-ownership.sh"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

make_fixture() {
  local fixture=$1
  mkdir -p "$fixture/.cargo" "$fixture/.github" "$fixture/.config" \
    "$fixture/scripts/check" "$fixture/scripts/lib" "$fixture/scripts/test"
  cp "$REPO_ROOT/.cargo/config.toml" "$fixture/.cargo/config.toml"
  cp -R "$REPO_ROOT/.github/workflows" "$fixture/.github/workflows"
  cp -R "$REPO_ROOT/.github/actions" "$fixture/.github/actions"
  cp -R "$REPO_ROOT/.github/rulesets" "$fixture/.github/rulesets"
  cp -R "$REPO_ROOT/.github/repository-settings" "$fixture/.github/repository-settings"
  cp "$REPO_ROOT/.github/dependabot.yaml" "$fixture/.github/dependabot.yaml"
  cp "$REPO_ROOT/.github/runs-on.yml" "$fixture/.github/runs-on.yml"
  cp "$REPO_ROOT/.config/target-matrix.json" "$fixture/.config/target-matrix.json"
  cp "$REPO_ROOT/.config/ci-plan-variants.json" "$fixture/.config/ci-plan-variants.json"
  cp "$REPO_ROOT/.config/rail.toml" "$fixture/.config/rail.toml"
  cp "$REPO_ROOT/.config/ci-tool-archives.tsv" "$fixture/.config/ci-tool-archives.tsv"
  cp -R "$REPO_ROOT/scripts/ci" "$fixture/scripts/ci"
  cp "$REPO_ROOT/scripts/lib/ci-tool-integrity.sh" "$REPO_ROOT/scripts/lib/common.sh" \
    "$REPO_ROOT/scripts/lib/feature-profiles.sh" \
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

missing_ci_policy="$TMP_ROOT/missing-ci-policy"
make_fixture "$missing_ci_policy"
yq -oy -p toml eval 'del(.plan.work."ci-policy")' -i \
  "$missing_ci_policy/.config/rail.toml"
expect_failure "$missing_ci_policy" "shared CI infrastructure does not widen the Cargo Rail matrix"

zig_dependent_cross_targets="$TMP_ROOT/zig-dependent-cross-targets"
make_fixture "$zig_dependent_cross_targets"
printf '\nzig version\n' >>"$zig_dependent_cross_targets/scripts/ci/cross-targets.sh"
expect_failure "$zig_dependent_cross_targets" "cross-target CI depends on Zig"

missing_recovery_tag="$TMP_ROOT/missing-recovery-tag"
make_fixture "$missing_recovery_tag"
yq eval 'del(.on.workflow_dispatch.inputs.tag)' -i \
  "$missing_recovery_tag/.github/workflows/release.yaml"
expect_failure "$missing_recovery_tag" "release recovery has no explicit tag identity"

unprotected_recovery="$TMP_ROOT/unprotected-recovery"
make_fixture "$unprotected_recovery"
sed -i.bak 's#refs/heads/main#refs/heads/recovery#' \
  "$unprotected_recovery/.github/workflows/release.yaml"
rm -f "$unprotected_recovery/.github/workflows/release.yaml.bak"
expect_failure "$unprotected_recovery" "release recovery accepts unprotected workflow code"

missing_ct_recovery_tag="$TMP_ROOT/missing-ct-recovery-tag"
make_fixture "$missing_ct_recovery_tag"
yq eval 'del(.on.workflow_dispatch.inputs.release_tag)' -i \
  "$missing_ct_recovery_tag/.github/workflows/ct.yaml"
expect_failure "$missing_ct_recovery_tag" "CT recovery has no immutable tag identity"

unprotected_ct_recovery="$TMP_ROOT/unprotected-ct-recovery"
make_fixture "$unprotected_ct_recovery"
yq eval '(.jobs.plan.steps[] | select(.name == "Resolve CT source") | .run) |= sub("refs/heads/main"; "refs/heads/recovery")' -i \
  "$unprotected_ct_recovery/.github/workflows/ct.yaml"
expect_failure "$unprotected_ct_recovery" "CT recovery accepts unprotected workflow code"

mutable_ct_recovery_checkout="$TMP_ROOT/mutable-ct-recovery-checkout"
make_fixture "$mutable_ct_recovery_checkout"
yq eval '(.jobs.ct.with.checkout_ref) = "${{ github.sha }}"' -i \
  "$mutable_ct_recovery_checkout/.github/workflows/ct.yaml"
expect_failure "$mutable_ct_recovery_checkout" "CT recovery ignores the immutable tag source"

untyped_ct_dispatch_numbers="$TMP_ROOT/untyped-ct-dispatch-numbers"
make_fixture "$untyped_ct_dispatch_numbers"
yq eval '(.jobs.ct.with.dudect_timeout) = "${{ inputs.dudect_timeout }}"' -i \
  "$untyped_ct_dispatch_numbers/.github/workflows/ct.yaml"
expect_failure "$untyped_ct_dispatch_numbers" "manual CT timeout bypasses typed normalization"

missing_s390x_vector_environment="$TMP_ROOT/missing-s390x-vector-environment"
make_fixture "$missing_s390x_vector_environment"
yq eval 'del(.jobs.run.steps[] | select(.name == "Run") | .env.CARGO_TARGET_S390X_UNKNOWN_LINUX_GNU_RUSTFLAGS)' -i \
  "$missing_s390x_vector_environment/.github/workflows/_rust-job.yaml"
expect_failure "$missing_s390x_vector_environment" "s390x sibling CT processes lose the vector target environment"

unvalidated_s390x_recovery="$TMP_ROOT/unvalidated-s390x-recovery"
make_fixture "$unvalidated_s390x_recovery"
yq eval '(.jobs.preflight.steps[] | select(.name == "Verify s390x CT recovery evidence") | .run) = "true"' -i \
  "$unvalidated_s390x_recovery/.github/workflows/release.yaml"
expect_failure "$unvalidated_s390x_recovery" "release accepts unvalidated replacement s390x evidence"

unvalidated_x86_64_recovery="$TMP_ROOT/unvalidated-x86_64-recovery"
make_fixture "$unvalidated_x86_64_recovery"
yq eval '(.jobs.preflight.steps[] | select(.name == "Verify x86_64 CT recovery evidence") | .run) = "true"' -i \
  "$unvalidated_x86_64_recovery/.github/workflows/release.yaml"
expect_failure "$unvalidated_x86_64_recovery" "release accepts unvalidated replacement x86_64 evidence"

unplanned_recovery_rustflags="$TMP_ROOT/unplanned-recovery-rustflags"
make_fixture "$unplanned_recovery_rustflags"
yq eval '(.jobs.ct.with.rustflags) = "-A warnings"' -i \
  "$unplanned_recovery_rustflags/.github/workflows/ct.yaml"
expect_failure "$unplanned_recovery_rustflags" "CT recovery accepts unplanned compiler flags"

mutable_publish_checkout="$TMP_ROOT/mutable-publish-checkout"
make_fixture "$mutable_publish_checkout"
yq eval '(.jobs.publish.steps[] | select(.name == "Checkout") | .with.ref) = "${{ github.ref }}"' -i \
  "$mutable_publish_checkout/.github/workflows/release.yaml"
expect_failure "$mutable_publish_checkout" "release publication ignores the preflight-verified tag"

hosted_macos="$TMP_ROOT/hosted-macos"
make_fixture "$hosted_macos"
yq eval '.jobs.hosted_macos = {"runs-on": "macos-15", "steps": [{"run": "true"}]}' -i \
  "$hosted_macos/.github/workflows/rsa.yaml"
expect_failure "$hosted_macos" "macOS testing is delegated to a hosted runner"

apple_runner_alias="$TMP_ROOT/apple-runner-alias"
make_fixture "$apple_runner_alias"
yq eval '.jobs.apple_runner = {"uses": "./.github/workflows/_rust-job.yaml", "with": {"runner": "darwin", "operation": "check"}}' -i \
  "$apple_runner_alias/.github/workflows/rsa.yaml"
expect_failure "$apple_runner_alias" "Apple testing is delegated through a custom runner label"

apple_rustflags="$TMP_ROOT/apple-rustflags"
make_fixture "$apple_rustflags"
printf '\n[target.aarch64-apple-darwin]\nrustflags = ["-C", "target-cpu=native"]\n' \
  >>"$apple_rustflags/.cargo/config.toml"
expect_failure "$apple_rustflags" "normal Apple builds inherit host-specific rustflags"

invalid_tool_digest="$TMP_ROOT/invalid-tool-digest"
make_fixture "$invalid_tool_digest"
sed -i.bak 's/ca1d64196d2d34771084afe76ea657d581bf628e31d993ff8e52ea09cc88a56d/not-a-digest/' \
  "$invalid_tool_digest/.config/ci-tool-archives.tsv"
rm -f "$invalid_tool_digest/.config/ci-tool-archives.tsv.bak"
expect_failure "$invalid_tool_digest" "direct tool digest is malformed"

mutable_tool_url="$TMP_ROOT/mutable-tool-url"
make_fixture "$mutable_tool_url"
sed -i.bak 's#/download/v48\.0\.0/#/download/Latest/#' \
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

competing_rust_cache="$TMP_ROOT/competing-rust-cache"
make_fixture "$competing_rust_cache"
yq eval '.runs.steps += [{"name": "Competing Rust cache", "uses": "Swatinem/rust-cache@0123456789012345678901234567890123456789"}]' -i \
  "$competing_rust_cache/.github/actions/setup/action.yaml"
expect_failure "$competing_rust_cache" "a competing Rust compiler cache bypasses Cargo Rail"

magic_cache_extra="$TMP_ROOT/magic-cache-extra"
make_fixture "$magic_cache_extra"
yq eval '.runners.linux-x64-ci.extras = ["s3-cache"]' -i \
  "$magic_cache_extra/.github/runs-on.yml"
expect_failure "$magic_cache_extra" "RunsOn MagicCache intercepts Cargo Rail compiler results"

missing_coverage_telemetry="$TMP_ROOT/missing-coverage-telemetry"
make_fixture "$missing_coverage_telemetry"
yq eval 'del(.jobs.coverage.steps[] | select(.name == "Capture Cargo Rail Cache Status"))' -i \
  "$missing_coverage_telemetry/.github/workflows/weekly.yaml"
expect_failure "$missing_coverage_telemetry" "Qualification coverage omits Cargo Rail telemetry"

unauthenticated_rustup="$TMP_ROOT/unauthenticated-rustup"
make_fixture "$unauthenticated_rustup"
printf '\n    - uses: dtolnay/rust-toolchain@e97e2d8cc328f1b50210efc529dca0028893a2d9\n' \
  >>"$unauthenticated_rustup/.github/actions/setup-toolchain/action.yaml"
expect_failure "$unauthenticated_rustup" "toolchain setup can run a network bootstrap installer"

inactive_toolchain_contract="$TMP_ROOT/inactive-toolchain-contract"
make_fixture "$inactive_toolchain_contract"
sed -i.bak 's/ "\$GITHUB_ENV"$//' \
  "$inactive_toolchain_contract/.github/actions/setup-toolchain/action.yaml"
rm -f "$inactive_toolchain_contract/.github/actions/setup-toolchain/action.yaml.bak"
expect_failure "$inactive_toolchain_contract" "toolchain contract is installed but not activated"

floating_rail_action="$TMP_ROOT/floating-rail-action"
make_fixture "$floating_rail_action"
yq eval '(.jobs."rail-plan".steps[] | select(.id == "rail") | .uses) = "loadingalias/cargo-rail-action@v6"' -i \
  "$floating_rail_action/.github/workflows/ci.yaml"
expect_failure "$floating_rail_action" "cargo-rail-action is not commit-pinned"

floating_rail_cache="$TMP_ROOT/floating-rail-cache"
make_fixture "$floating_rail_cache"
yq eval '(.runs.steps[] | select(.name == "Setup Cargo Rail Cache") | .uses) = "loadingalias/cargo-rail-action/cache@v7"' -i \
  "$floating_rail_cache/.github/actions/setup/action.yaml"
expect_failure "$floating_rail_cache" "the Cargo Rail cache action is not commit-pinned with the planner"

writable_pr_cache="$TMP_ROOT/writable-pr-cache"
make_fixture "$writable_pr_cache"
yq eval '(.on.workflow_call.inputs.cache_mode.default) = "read-write"' -i \
  "$writable_pr_cache/.github/workflows/_ci-suite.yaml"
expect_failure "$writable_pr_cache" "untrusted pull requests can write shared compiler results"

untrusted_cache_seeder="$TMP_ROOT/untrusted-cache-seeder"
make_fixture "$untrusted_cache_seeder"
yq eval '(.jobs."cache-seed".if) = "github.event_name == '\''pull_request'\''"' -i \
  "$untrusted_cache_seeder/.github/workflows/ci.yaml"
expect_failure "$untrusted_cache_seeder" "an untrusted event can assume shared-cache write authority"

missing_cache_identity="$TMP_ROOT/missing-cache-identity"
make_fixture "$missing_cache_identity"
yq eval 'del(.runs.steps[] | select(.name == "Authenticate Cargo Rail Cache"))' -i \
  "$missing_cache_identity/.github/actions/setup/action.yaml"
expect_failure "$missing_cache_identity" "shared-cache mode is not enforced by provider identity"

ambient_session_token="$TMP_ROOT/ambient-session-token"
make_fixture "$ambient_session_token"
sed -i.bak '/AWS_SESSION_TOKEN=/d' "$ambient_session_token/.github/actions/setup/action.yaml"
rm -f "$ambient_session_token/.github/actions/setup/action.yaml.bak"
expect_failure "$ambient_session_token" "an ambient AWS session token can corrupt the selected R2 identity"

missing_strict_probe="$TMP_ROOT/missing-strict-probe"
make_fixture "$missing_strict_probe"
yq eval 'del(.runs.steps[] | select(.name == "Setup Cargo Rail Cache") | .with."strict-probe")' -i \
  "$missing_strict_probe/.github/actions/setup/action.yaml"
expect_failure "$missing_strict_probe" "compiler jobs can start before R2 and native-v6 are authenticated"

physical_ci_cache="$TMP_ROOT/physical-ci-cache"
make_fixture "$physical_ci_cache"
yq eval '(.inputs."cache-root-portability".default) = "physical"' -i \
  "$physical_ci_cache/.github/actions/setup/action.yaml"
expect_failure "$physical_ci_cache" "ephemeral CI cache cannot share across checkout roots"

untrusted_rail_reuse="$TMP_ROOT/untrusted-rail-reuse"
make_fixture "$untrusted_rail_reuse"
yq eval '(.runs.steps[] | select(.name == "Install Cargo Tools") | .env.RSCRYPTO_AUTHENTICATED_CARGO_RAIL) = "true"' -i \
  "$untrusted_rail_reuse/.github/actions/setup/action.yaml"
expect_failure "$untrusted_rail_reuse" "a runner-provided Cargo Rail binary is treated as authenticated"

mismatched_rail_version="$TMP_ROOT/mismatched-rail-version"
make_fixture "$mismatched_rail_version"
yq eval '(.jobs."rail-plan".steps[] | select(.id == "rail") | .with.version) = "0.19.1"' -i \
  "$mismatched_rail_version/.github/workflows/ci.yaml"
expect_failure "$mismatched_rail_version" "cargo-rail-action bypasses the authenticated Cargo Rail version"

missing_surface_component="$TMP_ROOT/missing-surface-component"
make_fixture "$missing_surface_component"
yq eval 'del(.jobs."rail-plan".steps[] | select(.id == "rail") | .with.components)' -i \
  "$missing_surface_component/.github/workflows/ci.yaml"
expect_failure "$missing_surface_component" "cargo-rail-action does not prepare Surface"

mutable_rail_base="$TMP_ROOT/mutable-rail-base"
make_fixture "$mutable_rail_base"
yq eval '(.jobs."rail-plan".steps[] | select(.id == "rail") | .with.since) = "origin/main"' -i \
  "$mutable_rail_base/.github/workflows/ci.yaml"
expect_failure "$mutable_rail_base" "cargo-rail-action plans from a mutable base"

indirect_scorecard="$TMP_ROOT/indirect-scorecard"
make_fixture "$indirect_scorecard"
yq eval '(.jobs.scorecard.steps[] | select(.name == "Run Scorecard") | .uses) = "./.github/actions/scorecard"' -i \
  "$indirect_scorecard/.github/workflows/scorecard.yaml"
expect_failure "$indirect_scorecard" "Scorecard publication does not call the official action directly"

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

missing_main_seed="$TMP_ROOT/missing-main-seed"
make_fixture "$missing_main_seed"
yq eval 'del(.on.push)' -i "$missing_main_seed/.github/workflows/ci.yaml"
expect_failure "$missing_main_seed" "main cannot seed affected compiler results"

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

cancelled_suite_runs="$TMP_ROOT/cancelled-suite-runs"
make_fixture "$cancelled_suite_runs"
yq eval '.jobs.suite."if" |= sub("!cancelled\\(\\)"; "always()")' -i \
  "$cancelled_suite_runs/.github/workflows/ci.yaml"
expect_failure "$cancelled_suite_runs" "workflow cancellation can start the expensive suite"

fork_release_bypasses_intent="$TMP_ROOT/fork-release-bypasses-intent"
make_fixture "$fork_release_bypasses_intent"
yq eval '(.jobs."rail-plan".steps[] | select(.name == "Check Release Intent Coverage") | .if) |= sub(" && github.event.pull_request.head.repo.full_name == github.repository"; "")' -i \
  "$fork_release_bypasses_intent/.github/workflows/ci.yaml"
expect_failure "$fork_release_bypasses_intent" "a fork can bypass release intent with a branch name"

shell_fragment_input="$TMP_ROOT/shell-fragment-input"
make_fixture "$shell_fragment_input"
printf '\n      run_script: echo caller-controlled\n' \
  >>"$shell_fragment_input/.github/workflows/_ci-suite.yaml"
expect_failure "$shell_fragment_input" "reusable workflow accepts executable shell fragments"

missing_typed_operation="$TMP_ROOT/missing-typed-operation"
make_fixture "$missing_typed_operation"
yq eval 'del(.jobs.selected.with.operation)' -i \
  "$missing_typed_operation/.github/workflows/_ci-suite.yaml"
expect_failure "$missing_typed_operation" "reusable Rust job caller omits its operation"

unsupported_typed_operation="$TMP_ROOT/unsupported-typed-operation"
make_fixture "$unsupported_typed_operation"
yq eval '(.jobs.selected.with.operation) = "arbitrary-shell"' -i \
  "$unsupported_typed_operation/.github/workflows/_ci-suite.yaml"
expect_failure "$unsupported_typed_operation" "reusable Rust job caller selects an unsupported operation"

evaluated_workflow_input="$TMP_ROOT/evaluated-workflow-input"
make_fixture "$evaluated_workflow_input"
yq eval '(.jobs.run.steps[] | select(.name == "Run") | .run) = "echo \"${{ inputs.operation }}\""' -i \
  "$evaluated_workflow_input/.github/workflows/_rust-job.yaml"
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
jq '.variants += [{"id":"native-linux-musl","dimensions":{"display_name":"fake","operation":"native","runner_type":"runson","runner":"linux-x64-ci","target":"x86_64-unknown-linux-musl","timeout_minutes":30,"tools_mode":"none","toolchain_contract":"development","toolchain_components":""},"paths":[],"config":[],"cargo":[]}]' \
  "$fake_musl/.config/ci-plan-variants.json" >"$fake_musl/.config/ci-plan-variants.json.tmp"
mv "$fake_musl/.config/ci-plan-variants.json.tmp" "$fake_musl/.config/ci-plan-variants.json"
expect_failure "$fake_musl" "MUSL label without a MUSL target invocation"

missing_cross_owner="$TMP_ROOT/missing-cross-owner"
make_fixture "$missing_cross_owner"
jq '.variants |= map(select(.dimensions.operation != "cross-targets"))' \
  "$missing_cross_owner/.config/ci-plan-variants.json" >"$missing_cross_owner/.config/ci-plan-variants.json.tmp"
mv "$missing_cross_owner/.config/ci-plan-variants.json.tmp" "$missing_cross_owner/.config/ci-plan-variants.json"
expect_failure "$missing_cross_owner" "missing cross-target owner"

missing_graph_owner="$TMP_ROOT/missing-graph-owner"
make_fixture "$missing_graph_owner"
jq '.variants |= map(select(.dimensions.operation != "cargo-graph"))' \
  "$missing_graph_owner/.config/ci-plan-variants.json" >"$missing_graph_owner/.config/ci-plan-variants.json.tmp"
mv "$missing_graph_owner/.config/ci-plan-variants.json.tmp" "$missing_graph_owner/.config/ci-plan-variants.json"
expect_failure "$missing_graph_owner" "missing Cargo graph assurance owner"

duplicate_release_graph="$TMP_ROOT/duplicate-release-graph"
make_fixture "$duplicate_release_graph"
printf '\n# duplicate release owner\ncargo rail unify --check --explain\n' \
  >>"$duplicate_release_graph/scripts/ci/release-preflight.sh"
expect_failure "$duplicate_release_graph" "duplicate release Cargo graph assurance owner"

missing_release_graph_gate="$TMP_ROOT/missing-release-graph-gate"
make_fixture "$missing_release_graph_gate"
sed -i.bak '/CI Suite (release) \/ Cargo Graph Assurance \/ run/d' \
  "$missing_release_graph_gate/scripts/ci/release-evidence-check.sh"
rm -f "$missing_release_graph_gate/scripts/ci/release-evidence-check.sh.bak"
expect_failure "$missing_release_graph_gate" "missing release Cargo graph assurance gate"

scheduled_release_mode="$TMP_ROOT/scheduled-release-mode"
make_fixture "$scheduled_release_mode"
yq eval '(.jobs.mode.steps[] | select(.id == "mode") | .run) |= sub("mode=assurance"; "mode=release")' -i \
  "$scheduled_release_mode/.github/workflows/weekly.yaml"
expect_failure "$scheduled_release_mode" "scheduled Qualification run resolves to release mode"

release_by_default="$TMP_ROOT/release-by-default"
make_fixture "$release_by_default"
yq eval '.on.workflow_dispatch.inputs.mode.default = "release"' -i \
  "$release_by_default/.github/workflows/weekly.yaml"
expect_failure "$release_by_default" "manual Qualification run defaults to release evidence"

generic_weekly_gate="$TMP_ROOT/generic-weekly-gate"
make_fixture "$generic_weekly_gate"
yq eval '.jobs.complete.name = "Complete (weekly)"' -i \
  "$generic_weekly_gate/.github/workflows/weekly.yaml"
expect_failure "$generic_weekly_gate" "Qualification terminal gate omits the resolved mode"

fixed_weekly_retention="$TMP_ROOT/fixed-weekly-retention"
make_fixture "$fixed_weekly_retention"
yq eval '.jobs.ct.with.artifact_retention_days = 90' -i \
  "$fixed_weekly_retention/.github/workflows/weekly.yaml"
expect_failure "$fixed_weekly_retention" "assurance CT artifacts retain release lifetime"

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
sed -i.bak '/RISC-V CT Evidence (release) \/ Complete (CT)/d' \
  "$missing_riscv_release_artifact/scripts/ci/release-evidence-check.sh"
rm -f "$missing_riscv_release_artifact/scripts/ci/release-evidence-check.sh.bak"
expect_failure "$missing_riscv_release_artifact" "release without validated RISC-V evidence"

compact_weekly_ct="$TMP_ROOT/compact-weekly-ct"
make_fixture "$compact_weekly_ct"
yq eval '.jobs.ct.with.upload_raw_artifacts = false' -i "$compact_weekly_ct/.github/workflows/weekly.yaml"
expect_failure "$compact_weekly_ct" "Qualification without raw release CT evidence"

compact_riscv_ct="$TMP_ROOT/compact-riscv-ct"
make_fixture "$compact_riscv_ct"
yq eval '.jobs.riscv-ct.with.upload_raw_artifacts = false' -i \
  "$compact_riscv_ct/.github/workflows/weekly.yaml"
expect_failure "$compact_riscv_ct" "Qualification without raw RISC-V release CT evidence"

missing_qualification_riscv="$TMP_ROOT/missing-qualification-riscv"
make_fixture "$missing_qualification_riscv"
yq eval 'del(.jobs.riscv-native)' -i "$missing_qualification_riscv/.github/workflows/weekly.yaml"
expect_failure "$missing_qualification_riscv" "Qualification without RISC-V native evidence"

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

reintroduced_semver_owner="$TMP_ROOT/reintroduced-semver-owner"
make_fixture "$reintroduced_semver_owner"
printf '\n# pre-1.0 SemVer enforcement reintroduced\n      run: cargo semver-checks --package rscrypto --all-features\n' >>"$reintroduced_semver_owner/.github/workflows/weekly.yaml"
expect_failure "$reintroduced_semver_owner" "reintroduced SemVer owner"

shrunk_matrix="$TMP_ROOT/shrunk-matrix"
make_fixture "$shrunk_matrix"
sed -i.bak '/  "crc16"/d' "$shrunk_matrix/scripts/lib/feature-profiles.sh"
rm -f "$shrunk_matrix/scripts/lib/feature-profiles.sh.bak"
expect_failure "$shrunk_matrix" "removed required compile feature profile"

uncompiled_execution="$TMP_ROOT/uncompiled-execution"
make_fixture "$uncompiled_execution"
sed -i.bak '/EXECUTABLE_FEATURE_SETS=(/,/^)/ s/  "full"/  "std,full,uncompiled-fixture"/' \
  "$uncompiled_execution/scripts/lib/feature-profiles.sh"
rm -f "$uncompiled_execution/scripts/lib/feature-profiles.sh.bak"
expect_failure "$uncompiled_execution" "executable feature profile without compile coverage"

echo "CI ownership regression tests passed"
