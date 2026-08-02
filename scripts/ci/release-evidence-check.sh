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

weekly_runs=$(gh run list \
  --repo "$repo" \
  --workflow weekly.yaml \
  --status success \
  --json databaseId,headSha,status,conclusion,url,createdAt,event \
  --limit 100)
riscv_runs=$(gh run list \
  --repo "$repo" \
  --workflow riscv.yaml \
  --status success \
  --json databaseId,headSha,status,conclusion,url,createdAt,event \
  --limit 100)

select_weekly_release_run() {
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
  ' <<<"$weekly_runs")
  return 1
}

select_riscv_evidence_run() {
  local candidate=$1
  local selected
  local run_id
  local jobs
  while IFS= read -r selected; do
    run_id=$(jq -r '.databaseId' <<<"$selected")
    jobs=$(gh run view "$run_id" --repo "$repo" --json jobs)
    if jq -e '
      [
        "Native CI / run",
        "Constant-Time Evidence (RISC-V) / CT Full (RISE RISC-V riscv64) / run",
        "Constant-Time Evidence (RISC-V) / Complete (CT)",
        "Complete (RISC-V)"
      ] as $required
      | .jobs as $jobs
      | all($required[]; . as $name | [$jobs[] | select(.name == $name and .conclusion == "success")] | length == 1)
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
  ' <<<"$riscv_runs")
  return 1
}

selected_weekly=""
selected_riscv=""
select_pair() {
  local candidate=$1
  selected_weekly=$(select_weekly_release_run "$candidate") || selected_weekly=""
  selected_riscv=$(select_riscv_evidence_run "$candidate") || selected_riscv=""
  [[ -n "$selected_weekly" && -n "$selected_riscv" ]]
}

evidence_commit="$commit"
evidence_mode="exact_commit"
if ! select_pair "$commit"; then
  selected_weekly=""
  selected_riscv=""
fi

if [[ -z "$selected_weekly" || -z "$selected_riscv" ]]; then
  echo "No paired release-mode Weekly and RISC-V evidence is valid for release commit $commit." >&2
  echo "Both workflows must be manually dispatched for the exact commit; scheduled or ancestor evidence cannot be promoted." >&2
  exit 1
fi

weekly_run_id=$(jq -r '.databaseId' <<<"$selected_weekly")
weekly_run_url=$(jq -r '.url' <<<"$selected_weekly")
riscv_run_id=$(jq -r '.databaseId' <<<"$selected_riscv")
riscv_run_url=$(jq -r '.url' <<<"$selected_riscv")
evidence_version=$(git -C "$root" show "$evidence_commit:Cargo.toml" | "$SCRIPT_DIR/../ct/python.sh" -c \
  'import sys, tomllib; print(tomllib.loads(sys.stdin.read())["package"]["version"])')
weekly_jobs=$(gh run view "$weekly_run_id" --repo "$repo" --json jobs)
riscv_jobs=$(gh run view "$riscv_run_id" --repo "$repo" --json jobs)
weekly_artifacts=$(gh api --method GET "repos/$repo/actions/runs/$weekly_run_id/artifacts?per_page=100")
riscv_artifacts=$(gh api --method GET "repos/$repo/actions/runs/$riscv_run_id/artifacts?per_page=100")

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
  local job_prefix=$4
  local artifacts=$5
  local expected
  local returned
  local total
  local raw_total
  local valid
  local unique
  expected=$(jq -r --arg prefix "$job_prefix" \
    '[.jobs[] | select((.name | startswith($prefix)) and .conclusion == "success")] | length' <<<"$jobs")
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

require_job Weekly "$weekly_run_id" "$weekly_jobs" "Constant-Time Evidence (release) / Complete (CT)"
require_job Weekly "$weekly_run_id" "$weekly_jobs" "RSA Evidence (release) / Complete (RSA)"
require_job Weekly "$weekly_run_id" "$weekly_jobs" "CI Suite (release) / Cargo Graph Assurance / run"
require_job Weekly "$weekly_run_id" "$weekly_jobs" "Complete (release)"
require_job RISC-V "$riscv_run_id" "$riscv_jobs" "Native CI / run"
require_job RISC-V "$riscv_run_id" "$riscv_jobs" "Constant-Time Evidence (RISC-V) / CT Full (RISE RISC-V riscv64) / run"
require_job RISC-V "$riscv_run_id" "$riscv_jobs" "Constant-Time Evidence (RISC-V) / Complete (CT)"
require_job RISC-V "$riscv_run_id" "$riscv_jobs" "Complete (RISC-V)"
require_raw_ct_artifacts Weekly "$weekly_run_id" "$weekly_jobs" \
  "Constant-Time Evidence (release) / CT Full (" "$weekly_artifacts"
require_raw_ct_artifacts RISC-V "$riscv_run_id" "$riscv_jobs" \
  "Constant-Time Evidence (RISC-V) / CT Full (" "$riscv_artifacts"

if [[ -n ${GITHUB_OUTPUT:-} ]]; then
  {
    echo "weekly_run_id=$weekly_run_id"
    echo "weekly_run_url=$weekly_run_url"
    echo "riscv_run_id=$riscv_run_id"
    echo "riscv_run_url=$riscv_run_url"
    echo "weekly_commit=$evidence_commit"
    echo "weekly_version=$evidence_version"
    echo "weekly_evidence_mode=$evidence_mode"
  } >>"$GITHUB_OUTPUT"
fi

echo "Exact-commit release evidence passed: Weekly $weekly_run_url; RISC-V $riscv_run_url"
