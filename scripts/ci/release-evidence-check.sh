#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: release-evidence-check.sh --commit SHA [--repo OWNER/REPO]" >&2
  exit 2
}

commit=""
repo="${GITHUB_REPOSITORY:-loadingalias/rscrypto}"
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
    *) usage ;;
  esac
done

[[ "$commit" =~ ^[0-9a-fA-F]{40}$ ]] || usage
[[ "$repo" == */* ]] || usage

runs=$(gh run list \
  --repo "$repo" \
  --workflow weekly.yaml \
  --commit "$commit" \
  --status success \
  --json databaseId,headSha,status,conclusion,url,createdAt \
  --limit 20)
selected=$(jq -c --arg commit "$commit" '
  map(select(.headSha == $commit and .status == "completed" and .conclusion == "success"))
  | sort_by(.createdAt)
  | last // empty
' <<<"$runs")

if [[ -z "$selected" ]]; then
  echo "No successful Weekly run exists for exact release commit $commit." >&2
  echo "Dispatch weekly.yaml on that commit and do not create a tag until it passes." >&2
  exit 1
fi

run_id=$(jq -r '.databaseId' <<<"$selected")
run_url=$(jq -r '.url' <<<"$selected")
jobs=$(gh run view "$run_id" --repo "$repo" --json jobs)

require_job() {
  local name=$1
  local conclusion
  conclusion=$(jq -r --arg name "$name" '[.jobs[] | select(.name == $name)] | if length == 1 then .[0].conclusion else "missing" end' <<<"$jobs")
  if [[ "$conclusion" != "success" ]]; then
    echo "Required Weekly job '$name' is $conclusion in run $run_id." >&2
    exit 1
  fi
}

require_job "Constant-Time Evidence (weekly) / CT Full (RISE RISC-V riscv64) / run"
require_job "Constant-Time Evidence (weekly) / Complete (CT)"
require_job "RSA Evidence (weekly) / Complete (RSA)"
require_job "Complete (weekly)"

if [[ -n ${GITHUB_OUTPUT:-} ]]; then
  {
    echo "weekly_run_id=$run_id"
    echo "weekly_run_url=$run_url"
  } >>"$GITHUB_OUTPUT"
fi

echo "Exact-commit release evidence passed: $run_url"
