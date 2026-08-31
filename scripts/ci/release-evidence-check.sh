#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: release-evidence-check.sh --commit SHA [--repo OWNER/REPO] [--root PATH]" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
commit=""
repo="${GITHUB_REPOSITORY:-loadingalias/rscrypto}"
root="$(git rev-parse --show-toplevel)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit)
      commit=${2:-}
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
    *) usage ;;
  esac
done

[[ "$commit" =~ ^[0-9a-fA-F]{40}$ ]] || usage
[[ "$repo" == */* ]] || usage

qualification_runs=$(gh run list \
  --repo "$repo" \
  --workflow weekly.yaml \
  --status success \
  --json databaseId,headSha,status,conclusion,url,createdAt,event \
  --limit 100)

select_qualification_run() {
  local candidate=$1
  local selected
  local run_id
  local jobs
  while IFS= read -r selected; do
    run_id=$(jq -r '.databaseId' <<<"$selected")
    jobs=$(gh run view "$run_id" --repo "$repo" --json jobs)
    if jq -e '
      [.jobs[] | select(.name == "Complete (release)" and .conclusion == "success")]
      | length == 1
    ' <<<"$jobs" >/dev/null; then
      echo "$selected"
      return 0
    fi
  done < <(jq -c --arg commit "$candidate" '
    map(select(
      .headSha == $commit
      and .event == "workflow_dispatch"
      and .status == "completed"
      and .conclusion == "success"
    ))
    | sort_by(.createdAt)
    | reverse
    | .[]
  ' <<<"$qualification_runs")
  return 1
}

evidence_commit="$commit"
evidence_mode="exact_commit"
selected_qualification=$(select_qualification_run "$commit") || selected_qualification=""
if [[ -z "$selected_qualification" ]]; then
  echo "No release-mode qualification is valid for release commit $commit." >&2
  echo "The Qualification workflow must be manually dispatched for the exact commit; scheduled or ancestor evidence cannot be promoted." >&2
  exit 1
fi

qualification_run_id=$(jq -r '.databaseId' <<<"$selected_qualification")
qualification_run_url=$(jq -r '.url' <<<"$selected_qualification")
evidence_version=$(git -C "$root" show "$evidence_commit:Cargo.toml" | "$SCRIPT_DIR/../lib/python.sh" -c \
  'import sys, tomllib; print(tomllib.loads(sys.stdin.read())["package"]["version"])')
qualification_jobs=$(gh run view "$qualification_run_id" --repo "$repo" --json jobs)
qualification_artifacts=$(gh api --method GET "repos/$repo/actions/runs/$qualification_run_id/artifacts?per_page=100")

require_job() {
  local workflow=$1
  local run_id=$2
  local jobs=$3
  local name=$4
  local conclusion
  conclusion=$(jq -r --arg name "$name" '[.jobs[] | select(.name == $name)] | if length == 1 then .[0].conclusion else "missing" end' <<<"$jobs")
  if [[ "$conclusion" != "success" ]]; then
    echo "Required $workflow job '$name' is $conclusion in run $run_id." >&2
    exit 1
  fi
}

require_raw_ct_artifacts() {
  local workflow=$1
  local run_id=$2
  local jobs=$3
  local artifacts=$4
  local expected
  local returned
  local total
  local raw_total
  local valid
  local unique
  expected=$(jq -r '
    [.jobs[] | select(
      (
        (.name | startswith("Constant-Time Evidence (release) / CT Full ("))
        or (.name | startswith("RISC-V CT Evidence (release) / CT Full ("))
      )
      and .conclusion == "success"
    )]
    | length
  ' <<<"$jobs")
  returned=$(jq -r '.artifacts | length' <<<"$artifacts")
  total=$(jq -r '.total_count' <<<"$artifacts")
  raw_total=$(jq -r '[.artifacts[] | select(.name | startswith("ct-raw-"))] | length' <<<"$artifacts")
  valid=$(jq -r '[.artifacts[] | select(
    (.name | startswith("ct-raw-"))
    and (.expired == false)
    and (.size_in_bytes > 0)
  )] | length' <<<"$artifacts")
  unique=$(jq -r '[.artifacts[] | select(.name | startswith("ct-raw-")) | .name] | unique | length' <<<"$artifacts")
  if [[ "$expected" -lt 1 ]]; then
    echo "Required $workflow CT jobs are missing from run $run_id." >&2
    exit 1
  fi
  if [[ "$returned" -ne "$total" ]]; then
    echo "$workflow run $run_id returned only $returned of $total artifacts." >&2
    exit 1
  fi
  if [[ "$raw_total" -ne "$expected" || "$valid" -ne "$expected" || "$unique" -ne "$expected" ]]; then
    echo "$workflow run $run_id has $valid valid raw CT artifacts for $expected successful CT jobs." >&2
    exit 1
  fi
}

require_job Qualification "$qualification_run_id" "$qualification_jobs" "Constant-Time Evidence (release) / Complete (CT)"
require_job Qualification "$qualification_run_id" "$qualification_jobs" "RSA Evidence (release) / Complete (RSA)"
require_job Qualification "$qualification_run_id" "$qualification_jobs" "RISC-V Native Evidence / run"
require_job Qualification "$qualification_run_id" "$qualification_jobs" "RISC-V CT Evidence (release) / Complete (CT)"
require_job Qualification "$qualification_run_id" "$qualification_jobs" "CI Suite (release) / Cargo Graph Assurance / run"
require_job Qualification "$qualification_run_id" "$qualification_jobs" "Complete (release)"
require_raw_ct_artifacts Qualification "$qualification_run_id" "$qualification_jobs" "$qualification_artifacts"

if [[ -n ${GITHUB_OUTPUT:-} ]]; then
  {
    echo "qualification_run_id=$qualification_run_id"
    echo "qualification_run_url=$qualification_run_url"
    echo "qualification_commit=$evidence_commit"
    echo "qualification_version=$evidence_version"
    echo "qualification_evidence_mode=$evidence_mode"
  } >>"$GITHUB_OUTPUT"
fi

echo "Exact-commit release qualification passed: $qualification_run_url"
