#!/usr/bin/env bash
set -euo pipefail
unset BASH_ENV

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

fail() {
  echo "Cargo Rail matrix materialization regression failure: $*" >&2
  exit 1
}

READER="$TMP_ROOT/read.py"
cat >"$READER" <<'PY'
#!/usr/bin/env python3
import os
import sys

command = sys.argv[1]
if command == "validate":
    raise SystemExit(0)
if command == "is-required" and sys.argv[3] == "ci-policy":
    print(os.environ["MOCK_POLICY_REQUIRED"])
    raise SystemExit(0)
if command == "is-required" and sys.argv[3] == "ci-suite":
    print(os.environ["MOCK_SUITE_REQUIRED"])
    raise SystemExit(0)
if command == "matrix" and sys.argv[3] == "ci-suite":
    if os.environ["MOCK_POLICY_REQUIRED"] == "true" or os.environ["MOCK_SUITE_REQUIRED"] != "true":
        raise SystemExit(3)
    print(os.environ["MOCK_MATRIX"])
    raise SystemExit(0)
raise SystemExit(2)
PY
chmod +x "$READER"

PLAN="$TMP_ROOT/plan.json"
printf '{}\n' >"$PLAN"
SELECTED=$(jq -c '{include: [.variants[] | select(.id == "quality") | ({id} + .dimensions)]}' \
  "$REPO_ROOT/.config/ci-plan-variants.json")

materialize() {
  local policy=$1
  local matrix=$2
  local suite=${3:-true}
  local output="$TMP_ROOT/output"
  : >"$output"
  MOCK_POLICY_REQUIRED=$policy \
    MOCK_SUITE_REQUIRED=$suite \
    MOCK_MATRIX=$matrix \
    "$REPO_ROOT/scripts/ci/materialize-rail-plan.sh" \
      "$PLAN" "$READER" "$output" "$REPO_ROOT/.config/ci-plan-variants.json"
  sed -n 's/^matrix=//p' "$output"
}

selected=$(materialize false "$SELECTED")
[[ $(jq '.include | length' <<<"$selected") -eq 1 ]] \
  || fail "a source-selected matrix did not stay narrow"
[[ $(jq -r '.include[0].work.id' <<<"$selected") == quality ]] \
  || fail "the selected row identity changed during lowering"

policy_full=$(materialize true "$SELECTED")
[[ $(jq '.include | length' <<<"$policy_full") -eq 14 ]] \
  || fail "shared CI policy did not widen to the complete catalog"

forced_full=$(materialize false all)
[[ $(jq '.include | length' <<<"$forced_full") -eq 14 ]] \
  || fail "Cargo Rail's all selection did not materialize the complete catalog"

empty=$(materialize false unused false)
[[ $(jq '.include | length' <<<"$empty") -eq 0 ]] \
  || fail "skipped Cargo Rail work did not produce an empty matrix"

echo "Cargo Rail matrix materialization regression tests passed"
