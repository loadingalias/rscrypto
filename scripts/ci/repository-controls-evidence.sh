#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: repository-controls-evidence.sh --commit SHA --output PATH [--repo OWNER/REPO] [--root PATH] [--allow-redacted-bypass]" >&2
  exit 2
}

commit=""
output=""
repo="${GITHUB_REPOSITORY:-loadingalias/rscrypto}"
root="$(git rev-parse --show-toplevel)"
allow_redacted_bypass=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit)
      commit=${2:-}
      shift 2
      ;;
    --output)
      output=${2:-}
      shift 2
      ;;
    --repo)
      repo=${2:-}
      shift 2
      ;;
    --root)
      root=${2:-}
      shift 2
      ;;
    --allow-redacted-bypass)
      allow_redacted_bypass=true
      shift
      ;;
    *) usage ;;
  esac
done

[[ "$commit" =~ ^[0-9a-fA-F]{40}$ ]] || usage
[[ -n "$output" ]] || usage
[[ "$repo" == */* ]] || usage
git -C "$root" cat-file -e "$commit^{commit}" 2>/dev/null || {
  echo "repository controls error: commit is not present in the local repository: $commit" >&2
  exit 1
}

policy_path="$root/.github/rulesets/protect-main.json"
[[ -f "$policy_path" ]] || {
  echo "repository controls error: missing policy $policy_path" >&2
  exit 1
}
jq -e 'type == "object"' "$policy_path" >/dev/null || {
  echo "repository controls error: policy must be a JSON object" >&2
  exit 1
}

api() {
  gh api -H "X-GitHub-Api-Version: 2022-11-28" "$1"
}

canonical_filter='
  def canonical:
    if type == "object" then
      to_entries | sort_by(.key) | map(.value |= canonical) | from_entries
    elif type == "array" then
      map(canonical) | sort_by(tojson)
    else
      .
    end;
  canonical
'
full_policy_filter="{name, target, enforcement, conditions, bypass_actors, rules} | $canonical_filter"
public_policy_filter="{name, target, enforcement, conditions, rules} | $canonical_filter"

repository=$(api "repos/$repo")
default_branch=$(jq -er '.default_branch | select(type == "string" and length > 0)' <<<"$repository")
ruleset_name=$(jq -er '.name | select(type == "string" and length > 0)' "$policy_path")
rulesets=$(api "repos/$repo/rulesets?includes_parents=true&per_page=100")
ruleset_id=$(jq -er --arg name "$ruleset_name" --arg source "$repo" '
  [.[] | select(.name == $name and .target == "branch" and .source_type == "Repository" and .source == $source)]
  | if length == 1 then .[0].id else error("expected exactly one repository branch ruleset named " + $name) end
' <<<"$rulesets")

ruleset=$(api "repos/$repo/rulesets/$ruleset_id")
branch_path=$(jq -rn --arg branch "$default_branch" '$branch | @uri')
effective_rules=$(api "repos/$repo/rules/branches/$branch_path?per_page=100")
default_branch_commit=$(api "repos/$repo/commits/$branch_path")
default_branch_sha=$(jq -er '.sha | select(test("^[0-9a-fA-F]{40}$"))' <<<"$default_branch_commit")

bypass_evidence="verified_empty"
policy_filter=$full_policy_filter
if ! jq -e 'has("bypass_actors") and (.bypass_actors | type == "array")' <<<"$ruleset" >/dev/null; then
  if [[ "$allow_redacted_bypass" != true ]]; then
    echo "repository controls error: GitHub redacted bypass actors; rerun with repository-rules write access" >&2
    exit 1
  fi
  jq -e '
    (.bypass_actors // null) == null
    and ((.current_user_can_bypass // null) == null or .current_user_can_bypass == "never")
  ' <<<"$ruleset" >/dev/null || {
    echo "repository controls error: partial bypass data is not acceptable" >&2
    exit 1
  }
  bypass_evidence="redacted_by_github_api"
  policy_filter=$public_policy_filter
fi

if ! diff -u \
  <(jq -S "$policy_filter" "$policy_path") \
  <(jq -S "$policy_filter" <<<"$ruleset"); then
  echo "repository controls error: live ruleset differs from $policy_path" >&2
  exit 1
fi

jq -e --arg repo "$repo" --arg bypass_evidence "$bypass_evidence" '
  .source_type == "Repository"
  and .source == $repo
  and (
    if $bypass_evidence == "verified_empty" then
      .bypass_actors == [] and .current_user_can_bypass == "never"
    else
      true
    end
  )
' <<<"$ruleset" >/dev/null || {
  echo "repository controls error: the live ruleset is not repository-owned or permits this actor to bypass it" >&2
  exit 1
}

if ! diff -u \
  <(jq -S ".rules | $canonical_filter" "$policy_path") \
  <(jq -S --argjson id "$ruleset_id" \
    "[.[] | select(.ruleset_id == \$id) | {type, parameters: (.parameters // null)} | if .parameters == null then del(.parameters) else . end] | $canonical_filter" \
    <<<"$effective_rules"); then
  echo "repository controls error: effective rules on $default_branch differ from the repository ruleset" >&2
  exit 1
fi

captured_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
policy_sha256=$(sha256sum "$policy_path" | awk '{print $1}')
output_dir=$(dirname "$output")
mkdir -p "$output_dir"
output_tmp="${output}.tmp.$$"
trap 'rm -f "$output_tmp"' EXIT

jq -nS \
  --arg captured_at "$captured_at" \
  --arg commit "$commit" \
  --arg default_branch "$default_branch" \
  --arg default_branch_sha "$default_branch_sha" \
  --arg bypass_evidence "$bypass_evidence" \
  --arg policy_path ".github/rulesets/protect-main.json" \
  --arg policy_sha256 "$policy_sha256" \
  --arg repo "$repo" \
  --argjson repository "$repository" \
  --argjson policy "$(jq -S "$canonical_filter" "$policy_path")" \
  --argjson ruleset "$ruleset" \
  --argjson effective_rules "$effective_rules" '
  {
    schema_version: 1,
    kind: "rscrypto.repository-controls",
    captured_at: $captured_at,
    release_commit: $commit,
    repository: {
      id: $repository.id,
      full_name: $repo,
      visibility: $repository.visibility,
      default_branch: $default_branch,
      default_branch_sha: $default_branch_sha
    },
    policy: {
      path: $policy_path,
      sha256: $policy_sha256,
      expected: $policy
    },
    validation: {
      bypass_actors: {
        expected: [],
        status: $bypass_evidence,
        current_user_can_bypass: ($ruleset.current_user_can_bypass // null)
      }
    },
    live: {
      ruleset: $ruleset,
      effective_rules: $effective_rules
    }
  }
' >"$output_tmp"
mv "$output_tmp" "$output"
trap - EXIT

evidence_sha256=$(sha256sum "$output" | awk '{print $1}')
if [[ -n ${GITHUB_OUTPUT:-} ]]; then
  {
    echo "evidence_name=$(basename "$output")"
    echo "evidence_path=$output"
    echo "evidence_sha256=$evidence_sha256"
  } >>"$GITHUB_OUTPUT"
fi

echo "Repository controls match $policy_path"
echo "Evidence: $output"
echo "SHA-256: $evidence_sha256"
