#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: release-ci-check.sh --commit SHA [--repo OWNER/REPO] [--wait]" >&2
  exit 2
}

commit=""
repo="${GITHUB_REPOSITORY:-loadingalias/rscrypto}"
wait=false

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
    --wait)
      wait=true
      shift
      ;;
    *) usage ;;
  esac
done

[[ "$commit" =~ ^[0-9a-fA-F]{40}$ ]] || usage
[[ "$repo" == */* ]] || usage

attempts=1
if [[ "$wait" == true ]]; then
  attempts=360
fi

for ((attempt = 1; attempt <= attempts; attempt++)); do
  runs=$(gh run list \
    --repo "$repo" \
    --workflow CI \
    --branch main \
    --commit "$commit" \
    --event push \
    --json databaseId,status,conclusion,url,createdAt \
    --limit 10)

  selected=$(jq -c '
    map(select(.status == "completed" and .conclusion == "success"))
    | sort_by(.createdAt)
    | last // empty
  ' <<<"$runs")

  if [[ -n "$selected" ]]; then
    run_id=$(jq -r '.databaseId' <<<"$selected")
    run_url=$(jq -r '.url' <<<"$selected")
    jobs=$(gh run view "$run_id" --repo "$repo" --json jobs)
    graph_result=$(jq -r '
      [.jobs[] | select(.name == "CI Suite / Cargo Graph Assurance / run")]
      | if length == 1 then .[0].conclusion else "missing" end
    ' <<<"$jobs")

    if [[ "$graph_result" != "success" ]]; then
      echo "Cargo Graph Assurance result for $commit: $graph_result" >&2
      exit 1
    fi

    echo "Exact-commit CI passed: $run_url"
    exit 0
  fi

  latest=$(jq -c 'sort_by(.createdAt) | last // empty' <<<"$runs")
  if [[ -n "$latest" ]]; then
    run_status=$(jq -r '.status' <<<"$latest")
    run_conclusion=$(jq -r '.conclusion // ""' <<<"$latest")
    run_url=$(jq -r '.url' <<<"$latest")
    echo "CI for $commit: status=$run_status conclusion=${run_conclusion:-none}"
    echo "$run_url"

    if [[ "$run_status" == "completed" ]]; then
      echo "CI did not pass for $commit: $run_conclusion" >&2
      exit 1
    fi
  else
    echo "No push CI run for $commit on main."
  fi

  if [[ "$wait" != true ]]; then
    echo "Wait for exact-commit CI to pass before creating the release tag." >&2
    exit 1
  fi

  echo "Retry $attempt/$attempts in 30 seconds."
  sleep 30
done

echo "Timed out waiting for CI to complete for $commit" >&2
exit 1
