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
sed -i.bak '/CI Suite \/ Cargo Graph Assurance \/ run/d' \
  "$missing_release_graph_gate/.github/workflows/release.yaml"
rm -f "$missing_release_graph_gate/.github/workflows/release.yaml.bak"
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

compact_weekly_ct="$TMP_ROOT/compact-weekly-ct"
make_fixture "$compact_weekly_ct"
sed -i.bak 's/upload_raw_artifacts: true/upload_raw_artifacts: false/' "$compact_weekly_ct/.github/workflows/weekly.yaml"
rm -f "$compact_weekly_ct/.github/workflows/weekly.yaml.bak"
expect_failure "$compact_weekly_ct" "Weekly without raw release CT evidence"

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
