#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/pin-actions.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

make_fixture() {
  local fixture=$1
  mkdir -p "$fixture/.github/workflows" "$fixture/.github/actions"
  cp "$REPO_ROOT/.github/actions-lock.yaml" "$fixture/.github/actions-lock.yaml"
  cp "$REPO_ROOT/.github/workflows/scorecard.yaml" "$fixture/.github/workflows/scorecard.yaml"
}

expect_failure() {
  local fixture=$1
  local description=$2
  if "$CHECKER" --verify-only --root "$fixture" >/dev/null 2>&1; then
    echo "expected action-pin failure: $description" >&2
    exit 1
  fi
}

baseline="$TMP_ROOT/baseline"
make_fixture "$baseline"
"$CHECKER" --verify-only --root "$baseline" >/dev/null

sha_mismatch="$TMP_ROOT/sha-mismatch"
make_fixture "$sha_mismatch"
sed -i.bak -E \
  's|github/codeql-action/upload-sarif@[0-9a-f]{40}|github/codeql-action/upload-sarif@0000000000000000000000000000000000000000|' \
  "$sha_mismatch/.github/workflows/scorecard.yaml"
rm -f "$sha_mismatch/.github/workflows/scorecard.yaml.bak"
expect_failure "$sha_mismatch" "workflow SHA differs from lock"

ref_mismatch="$TMP_ROOT/ref-mismatch"
make_fixture "$ref_mismatch"
sed -i.bak -E 's/# v[0-9][^[:space:]]*/# v0.0.0/' "$ref_mismatch/.github/workflows/scorecard.yaml"
rm -f "$ref_mismatch/.github/workflows/scorecard.yaml.bak"
expect_failure "$ref_mismatch" "workflow ref differs from lock"

missing_lock="$TMP_ROOT/missing-lock"
make_fixture "$missing_lock"
yq eval -i 'del(."github/codeql-action/upload-sarif")' "$missing_lock/.github/actions-lock.yaml"
expect_failure "$missing_lock" "workflow action missing from lock"

unpinned="$TMP_ROOT/unpinned"
make_fixture "$unpinned"
sed -i.bak -E \
  's|github/codeql-action/upload-sarif@[0-9a-f]{40}|github/codeql-action/upload-sarif@v4|' \
  "$unpinned/.github/workflows/scorecard.yaml"
rm -f "$unpinned/.github/workflows/scorecard.yaml.bak"
expect_failure "$unpinned" "workflow action is not SHA-pinned"

nested_action="$TMP_ROOT/nested-action"
fake_bin="$TMP_ROOT/bin"
mkdir -p "$nested_action/.github/workflows" "$nested_action/.github/actions" "$fake_bin"
cat >"$nested_action/.github/actions-lock.yaml" <<'YAML'
github/codeql-action/upload-sarif:
  ref: v4.37.0
  sha: "0000000000000000000000000000000000000000"
YAML
cat >"$nested_action/.github/workflows/scorecard.yaml" <<'YAML'
jobs:
  scorecard:
    steps:
      - uses: github/codeql-action/upload-sarif@0000000000000000000000000000000000000000 # v4.37.0
YAML
cat >"$fake_bin/gh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

if [[ "$1" == "auth" && "$2" == "status" ]]; then
  exit 0
fi

[[ "$1" == "api" ]]
[[ "$2" == "repos/github/codeql-action/commits/v4.37.0" ]]
printf '%s\n' 99df26d4f13ea111d4ec1a7dddef6063f76b97e9
SH
chmod +x "$fake_bin/gh"

PATH="$fake_bin:$PATH" "$CHECKER" --update-lock --root "$nested_action" >/dev/null
[[ $(yq eval -r '."github/codeql-action/upload-sarif".sha' "$nested_action/.github/actions-lock.yaml") \
  == "99df26d4f13ea111d4ec1a7dddef6063f76b97e9" ]]
grep -Fq \
  'uses: github/codeql-action/upload-sarif@99df26d4f13ea111d4ec1a7dddef6063f76b97e9  # v4.37.0' \
  "$nested_action/.github/workflows/scorecard.yaml"
yq eval -i '."github/codeql-action/upload-sarif".updated = "unchanged"' "$nested_action/.github/actions-lock.yaml"
PATH="$fake_bin:$PATH" "$CHECKER" --update-lock --root "$nested_action" >/dev/null
[[ $(yq eval -r '."github/codeql-action/upload-sarif".updated' "$nested_action/.github/actions-lock.yaml") == "unchanged" ]]

echo "Action pin regression tests passed"
