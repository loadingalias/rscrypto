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

runs=$(gh run list \
  --repo "$repo" \
  --workflow weekly.yaml \
  --status success \
  --json databaseId,headSha,status,conclusion,url,createdAt \
  --limit 100)
selected=$(jq -c --arg commit "$commit" '
  map(select(.headSha == $commit and .status == "completed" and .conclusion == "success"))
  | sort_by(.createdAt)
  | last // empty
' <<<"$runs")

release_only_delta() {
  local evidence_commit=$1
  local release_commit=$2
  local path

  git -C "$root" merge-base --is-ancestor "$evidence_commit" "$release_commit" || return 1

  while IFS= read -r path; do
    case "$path" in
      CHANGELOG.md|Cargo.toml|Cargo.lock|docs/release.md|.changes/*|.github/workflows/release.yaml|scripts/ci/*|\
        scripts/ct/validate_release_evidence.py|\
        tools/ct-binsec-harness/Cargo.lock|tools/ct-dudect/Cargo.lock|tools/ct-harness/Cargo.lock) ;;
      *) return 1 ;;
    esac
  done < <(git -C "$root" diff --name-only "$evidence_commit..$release_commit")

  (
    cd "$root"
    "$SCRIPT_DIR/../ct/python.sh" - "$evidence_commit" "$release_commit" <<'PY'
import copy
import subprocess
import sys
import tomllib

evidence_commit, release_commit = sys.argv[1:]
lockfiles = (
  "Cargo.lock",
  "tools/ct-binsec-harness/Cargo.lock",
  "tools/ct-dudect/Cargo.lock",
  "tools/ct-harness/Cargo.lock",
)


def load(commit: str, path: str) -> dict:
  raw = subprocess.check_output(["git", "show", f"{commit}:{path}"], text=True)
  return tomllib.loads(raw)


def normalized_manifest(commit: str) -> dict:
  manifest = copy.deepcopy(load(commit, "Cargo.toml"))
  manifest["package"]["version"] = "<release-version>"
  return manifest


def normalized_lock(commit: str, path: str) -> dict:
  lock = copy.deepcopy(load(commit, path))
  packages = [package for package in lock.get("package", []) if package.get("name") == "rscrypto" and "source" not in package]
  if len(packages) != 1:
    raise SystemExit(f"{path} must contain exactly one local rscrypto package, found {len(packages)}")
  packages[0]["version"] = "<release-version>"
  return lock


if normalized_manifest(evidence_commit) != normalized_manifest(release_commit):
  raise SystemExit("Cargo.toml changed beyond package.version")

for path in lockfiles:
  if normalized_lock(evidence_commit, path) != normalized_lock(release_commit, path):
    raise SystemExit(f"{path} changed beyond the local rscrypto package version")
PY
  )
}

evidence_mode="exact_commit"
if [[ -z "$selected" ]]; then
  while IFS= read -r candidate; do
    [[ "$candidate" =~ ^[0-9a-fA-F]{40}$ ]] || continue
    git -C "$root" cat-file -e "$candidate^{commit}" 2>/dev/null || continue
    if release_only_delta "$candidate" "$commit"; then
      selected=$(jq -c --arg commit "$candidate" '
        map(select(.headSha == $commit and .status == "completed" and .conclusion == "success"))
        | sort_by(.createdAt)
        | last // empty
      ' <<<"$runs")
      evidence_mode="release_only_delta"
      break
    fi
  done < <(jq -r 'sort_by(.createdAt) | reverse | .[].headSha' <<<"$runs")
fi

if [[ -z "$selected" ]]; then
  echo "No successful Weekly evidence is valid for release commit $commit." >&2
  echo "A prior run may be promoted only when the intervening delta is provably release-only." >&2
  exit 1
fi

run_id=$(jq -r '.databaseId' <<<"$selected")
run_url=$(jq -r '.url' <<<"$selected")
evidence_commit=$(jq -r '.headSha' <<<"$selected")
evidence_version=$(git -C "$root" show "$evidence_commit:Cargo.toml" | "$SCRIPT_DIR/../ct/python.sh" -c \
  'import sys, tomllib; print(tomllib.loads(sys.stdin.read())["package"]["version"])')
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
    echo "weekly_commit=$evidence_commit"
    echo "weekly_version=$evidence_version"
    echo "weekly_evidence_mode=$evidence_mode"
  } >>"$GITHUB_OUTPUT"
fi

if [[ "$evidence_mode" == "exact_commit" ]]; then
  echo "Exact-commit release evidence passed: $run_url"
else
  echo "Release-only delta proven; promoted Weekly evidence from $evidence_commit: $run_url"
fi
