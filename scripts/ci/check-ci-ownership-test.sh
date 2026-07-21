#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKER="$SCRIPT_DIR/check-ci-ownership.sh"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

make_fixture() {
  local fixture=$1
  mkdir -p "$fixture/.github" "$fixture/.config" "$fixture/scripts/check" "$fixture/scripts/test"
  cp -R "$REPO_ROOT/.github/workflows" "$fixture/.github/workflows"
  cp -R "$REPO_ROOT/.github/rulesets" "$fixture/.github/rulesets"
  cp -R "$REPO_ROOT/.github/repository-settings" "$fixture/.github/repository-settings"
  cp "$REPO_ROOT/.github/dependabot.yaml" "$fixture/.github/dependabot.yaml"
  cp "$REPO_ROOT/.config/target-matrix.json" "$fixture/.config/target-matrix.json"
  cp -R "$REPO_ROOT/scripts/ci" "$fixture/scripts/ci"
  cp "$REPO_ROOT/scripts/check/check-all.sh" "$REPO_ROOT/scripts/check/check-feature-matrix.sh" "$fixture/scripts/check/"
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
yq eval '.jobs.suite.if = "always()"' -i "$draft_runs_suite/.github/workflows/ci.yaml"
expect_failure "$draft_runs_suite" "draft PR can run the expensive suite"

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
sed -i.bak '/scripts\/ci\/cross-targets\.sh/d' "$missing_cross_owner/.github/workflows/_ci-suite.yaml"
rm -f "$missing_cross_owner/.github/workflows/_ci-suite.yaml.bak"
expect_failure "$missing_cross_owner" "missing cross-target owner"

missing_graph_owner="$TMP_ROOT/missing-graph-owner"
make_fixture "$missing_graph_owner"
sed -i.bak '/cargo rail unify --check/d' "$missing_graph_owner/.github/workflows/_ci-suite.yaml"
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
