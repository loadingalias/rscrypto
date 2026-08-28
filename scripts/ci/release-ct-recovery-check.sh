#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: release-ct-recovery-check.sh --run-id ID --platform-group GROUP --workflow-commit SHA [--repo OWNER/REPO]" >&2
  exit 2
}

run_id=""
platform_group=""
workflow_commit=""
repo="${GITHUB_REPOSITORY:-loadingalias/rscrypto}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) run_id=${2:-}; shift 2 ;;
    --platform-group) platform_group=${2:-}; shift 2 ;;
    --workflow-commit) workflow_commit=${2:-}; shift 2 ;;
    --repo) repo=${2:-}; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$run_id" =~ ^[1-9][0-9]*$ ]] || usage
[[ "$workflow_commit" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$repo" == */* ]] || usage

required_jobs=("Resolve CT Matrix")
required_artifacts=()
case "$platform_group" in
  s390x)
    required_jobs+=("CT Full (IBM Z s390x) / run")
    required_artifacts+=("ct-raw-ibm-s390x")
    ;;
  x86_64)
    required_jobs+=(
      "CT Full (AMD Zen4) / run"
      "CT Full (AMD Zen5) / run"
      "CT Full (Intel Ice Lake) / run"
      "CT Full (Intel Sapphire Rapids) / run"
    )
    required_artifacts+=(
      "ct-raw-amd-zen4"
      "ct-raw-amd-zen5"
      "ct-raw-intel-icl"
      "ct-raw-intel-spr"
    )
    ;;
  *) usage ;;
esac
required_jobs+=("Complete (CT)")

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
for name in "${required_jobs[@]}"; do
  count=$(jq -r --arg name "$name" \
    '[.jobs[] | select(.name == $name and .conclusion == "success")] | length' <<<"$jobs")
  if [[ "$count" -ne 1 ]]; then
    echo "CT recovery run $run_id lacks one successful '$name' job." >&2
    exit 1
  fi
done

artifacts=$(gh api "repos/$repo/actions/runs/$run_id/artifacts?per_page=100")
expected_artifacts=$(printf '%s\n' "${required_artifacts[@]}" | jq -Rsc 'split("\n") | map(select(length > 0))')
if ! jq -e --argjson expected "$expected_artifacts" '
  .total_count == ($expected | length)
  and (.artifacts | length) == ($expected | length)
  and ([.artifacts[].name] | sort) == ($expected | sort)
  and all(.artifacts[]; .expired == false and .size_in_bytes > 0)
' <<<"$artifacts" >/dev/null; then
  echo "CT recovery run $run_id does not contain exactly the live raw $platform_group artifact set." >&2
  exit 1
fi

run_url=$(jq -r '.html_url' <<<"$run")
if [[ -n ${GITHUB_OUTPUT:-} ]]; then
  {
    echo "recovery_run_id=$run_id"
    echo "recovery_run_url=$run_url"
  } >>"$GITHUB_OUTPUT"
fi

echo "Validated $platform_group CT recovery run: $run_url"
