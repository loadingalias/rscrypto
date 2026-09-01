#!/usr/bin/env bash
# Prove exact, shared-policy, and fail-closed Cargo Rail feature selection
# without changing the repository index or compiling product code.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

fail() {
  echo "feature planning regression failure: $*" >&2
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
git read-tree HEAD
git add -A
base_tree=$(git write-tree)
base_commit=$(printf 'effective feature planning fixture\n' | git commit-tree "$base_tree" -p HEAD)

plan_existing_path() {
  local path=$1
  local label=$2
  local output=$3
  local mode blob tree commit
  git read-tree "$base_tree"
  mode=$(git ls-tree "$base_tree" -- "$path" | awk '{print $1}')
  [[ -n "$mode" ]] || fail "fixture path is absent: $path"
  blob=$(
    {
      git show "$base_commit:$path"
      case "$path" in
        *.toml) printf '\n# feature planning fixture: %s\n' "$label" ;;
        *) printf '\n// feature planning fixture: %s\n' "$label" ;;
      esac
    } | git hash-object -w --stdin
  )
  git update-index --add --cacheinfo "$mode,$blob,$path"
  tree=$(git write-tree)
  commit=$(printf '%s feature planning fixture\n' "$label" | git commit-tree "$tree" -p "$base_commit")
  cargo rail plan --quiet --from "$base_commit" --to "$commit" --json >"$output"
}

plan_new_path() {
  local path=$1
  local output=$2
  local blob tree commit
  git read-tree "$base_tree"
  blob=$(printf '// unattributed feature planning fixture\n' | git hash-object -w --stdin)
  git update-index --add --cacheinfo "100644,$blob,$path"
  tree=$(git write-tree)
  commit=$(printf 'unattributed feature planning fixture\n' | git commit-tree "$tree" -p "$base_commit")
  cargo rail plan --quiet --from "$base_commit" --to "$commit" --json >"$output"
}

rsa_plan="$TMP_ROOT/rsa.json"
plan_existing_path src/auth/rsa.rs rsa "$rsa_plan"
jq -e '
  .work["contracts.features"].scope.selection
  | .kind == "selected"
    and [.variants[].id] == ["signatures"]
' "$rsa_plan" >/dev/null || fail "RSA source did not select only the signature feature group"

policy_plan="$TMP_ROOT/policy.json"
plan_existing_path Cargo.toml policy "$policy_plan"
jq -e '
  .work["contracts.features"].scope.selection
  | .kind == "selected"
    and [.variants[].id] == ["feature-policy"]
' "$policy_plan" >/dev/null || fail "feature policy did not select the full-profile policy group"

unknown_plan="$TMP_ROOT/unknown.json"
plan_new_path tests/unattributed_feature_contract.rs "$unknown_plan"
jq -e '
  .work["contracts.features"].scope.selection.kind == "all"
' "$unknown_plan" >/dev/null || fail "unattributed feature input did not widen to every group"

echo "Feature planning regression tests passed"
