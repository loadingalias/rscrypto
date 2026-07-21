#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/check-action-pins.sh"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

expected_sha=7188fc363630916deb702c7fdcf4e481b751f97a
fake_bin="$TMP_ROOT/bin"
mkdir -p "$fake_bin"

cat >"$fake_bin/git" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == "ls-remote" ]]
ref=""
for argument in "$@"; do
  case "$argument" in
    refs/tags/*)
      [[ "$argument" == *'^{}' ]] || ref=${argument#refs/tags/}
      ;;
  esac
done
printf '%s\trefs/tags/%s\n' \
  "${FAKE_RESOLVED_SHA:-7188fc363630916deb702c7fdcf4e481b751f97a}" "$ref"
SH

cat >"$fake_bin/curl" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
output=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      output=$2
      shift 2
      ;;
    *) shift ;;
  esac
done
[[ -n "$output" ]]
[[ ${FAKE_MISSING_DEFINITION:-0} != 1 ]] || exit 22
printf 'runs:\n  using: %s\n' "${FAKE_RUNTIME:-node24}" >"$output"
SH
chmod +x "$fake_bin/git" "$fake_bin/curl"

make_fixture() {
  local fixture=$1
  mkdir -p "$fixture/.github/workflows" "$fixture/.github/actions"
  cat >"$fixture/.github/workflows/scorecard.yaml" <<YAML
jobs:
  scorecard:
    steps:
      - uses: github/codeql-action/upload-sarif@$expected_sha  # v4.37.1
YAML
}

run_checker() {
  PATH="$fake_bin:$PATH" "$CHECKER" --root "$1"
}

expect_failure() {
  local fixture=$1
  local description=$2
  if run_checker "$fixture" >/dev/null 2>&1; then
    echo "expected action-pin failure: $description" >&2
    exit 1
  fi
}

baseline="$TMP_ROOT/baseline"
make_fixture "$baseline"
run_checker "$baseline" >/dev/null

unpinned="$TMP_ROOT/unpinned"
make_fixture "$unpinned"
sed -i.bak "s/@$expected_sha/@v4/" "$unpinned/.github/workflows/scorecard.yaml"
rm -f "$unpinned/.github/workflows/scorecard.yaml.bak"
expect_failure "$unpinned" "action is not SHA-pinned"

missing_ref="$TMP_ROOT/missing-ref"
make_fixture "$missing_ref"
sed -i.bak 's/  # v4.37.1//' "$missing_ref/.github/workflows/scorecard.yaml"
rm -f "$missing_ref/.github/workflows/scorecard.yaml.bak"
expect_failure "$missing_ref" "semantic ref comment is missing"

inconsistent="$TMP_ROOT/inconsistent"
make_fixture "$inconsistent"
cat >"$inconsistent/.github/actions/nested.yaml" <<'YAML'
runs:
  using: composite
  steps:
    - uses: github/codeql-action/upload-sarif@99df26d4f13ea111d4ec1a7dddef6063f76b97e9  # v4.37.0
YAML
expect_failure "$inconsistent" "one action has inconsistent pins"

wrong_tag="$TMP_ROOT/wrong-tag"
make_fixture "$wrong_tag"
if FAKE_RESOLVED_SHA=0000000000000000000000000000000000000000 \
  run_checker "$wrong_tag" >/dev/null 2>&1; then
  echo "expected action-pin failure: semantic ref resolves to another SHA" >&2
  exit 1
fi

unsupported_runtime="$TMP_ROOT/unsupported-runtime"
make_fixture "$unsupported_runtime"
if FAKE_RUNTIME=node20 run_checker "$unsupported_runtime" >/dev/null 2>&1; then
  echo "expected action-pin failure: unsupported action runtime" >&2
  exit 1
fi

missing_definition="$TMP_ROOT/missing-definition"
make_fixture "$missing_definition"
if FAKE_MISSING_DEFINITION=1 run_checker "$missing_definition" >/dev/null 2>&1; then
  echo "expected action-pin failure: missing action definition" >&2
  exit 1
fi

echo "Action pin regression tests passed"
