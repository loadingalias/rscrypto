#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: release-ct-recovery-check.sh --run-id ID --workflow-commit SHA [--repo OWNER/REPO]" >&2
  exit 2
}

run_id=""
workflow_commit=""
repo="${GITHUB_REPOSITORY:-loadingalias/rscrypto}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) run_id=${2:-}; shift 2 ;;
    --workflow-commit) workflow_commit=${2:-}; shift 2 ;;
    --repo) repo=${2:-}; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$run_id" =~ ^[1-9][0-9]*$ ]] || usage
[[ "$workflow_commit" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$repo" == */* ]] || usage

run=$(gh api "repos/$repo/actions/runs/$run_id")
if ! jq -e \
  --arg repo "$repo" \
  --arg workflow_commit "$workflow_commit" '
    .event == "workflow_dispatch"
    and .status == "completed"
    and .conclusion == "success"
    and .head_branch == "main"
    and .head_sha == $workflow_commit
    and .path == ".github/workflows/ct.yaml"
    and .head_repository.full_name == $repo
  ' <<<"$run" >/dev/null; then
  echo "CT recovery run $run_id is not a successful protected-main ct.yaml dispatch for $workflow_commit." >&2
  exit 1
fi

jobs=$(gh run view "$run_id" --repo "$repo" --json jobs)
for name in \
  "Resolve CT Matrix" \
  "CT Full (IBM Z s390x) / run" \
  "Complete (CT)"; do
  count=$(jq -r --arg name "$name" \
    '[.jobs[] | select(.name == $name and .conclusion == "success")] | length' <<<"$jobs")
  if [[ "$count" -ne 1 ]]; then
    echo "CT recovery run $run_id lacks one successful '$name' job." >&2
    exit 1
  fi
done

artifacts=$(gh api "repos/$repo/actions/runs/$run_id/artifacts?per_page=100")
if ! jq -e '
  .total_count == 1
  and (.artifacts | length) == 1
  and .artifacts[0].name == "ct-raw-ibm-s390x"
  and .artifacts[0].expired == false
  and .artifacts[0].size_in_bytes > 0
' <<<"$artifacts" >/dev/null; then
  echo "CT recovery run $run_id does not contain exactly one live raw s390x artifact." >&2
  exit 1
fi

run_url=$(jq -r '.html_url' <<<"$run")
if [[ -n ${GITHUB_OUTPUT:-} ]]; then
  {
    echo "s390x_run_id=$run_id"
    echo "s390x_run_url=$run_url"
  } >>"$GITHUB_OUTPUT"
fi

echo "Validated s390x CT recovery run: $run_url"
