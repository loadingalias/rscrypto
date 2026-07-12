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

shrunk_matrix="$TMP_ROOT/shrunk-matrix"
make_fixture "$shrunk_matrix"
sed -i.bak '/  "crc16"/d' "$shrunk_matrix/scripts/test/test-feature-matrix.sh"
rm -f "$shrunk_matrix/scripts/test/test-feature-matrix.sh.bak"
expect_failure "$shrunk_matrix" "removed executable feature profile"

echo "CI ownership regression tests passed"
