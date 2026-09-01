#!/usr/bin/env bash
# Validate one transported plan and export it for later GitHub Actions steps.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 PLAN_DIRECTORY PLAN_IDENTITY HEAD_COMMIT" >&2
  exit 2
fi

plan_dir=$(cd "$1" && pwd)
identity=$2
head_commit=$3
plan_file="$plan_dir/plan.json"
plan_reader="$plan_dir/read.py"
cargo_rail="$plan_dir/cargo-rail"

[[ -f "$plan_file" && -f "$plan_reader" && -f "$cargo_rail" ]] || {
  echo "Transported Cargo Rail plan artifact is incomplete" >&2
  exit 2
}

chmod 700 "$plan_reader" "$cargo_rail"
export PATH="$plan_dir:$PATH"
export RAIL_PLAN_FILE="$plan_file"
export RAIL_PLAN_READER="$plan_reader"
export RAIL_PLAN_IDENTITY="$identity"
export RAIL_PLAN_HEAD_COMMIT="$head_commit"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/rail-plan.sh
source "$SCRIPT_DIR/../lib/rail-plan.sh"
rail_prime_plan
printf 'Cargo Rail plan: %s\n' "$identity"
printf 'Required work: %s\n' \
  "$(jq -r '.required | if length == 0 then "none" else join(", ") end' "$plan_file")"

{
  printf 'RAIL_PLAN_FILE=%s\n' "$plan_file"
  printf 'RAIL_PLAN_READER=%s\n' "$plan_reader"
  printf 'RAIL_PLAN_IDENTITY=%s\n' "$identity"
  printf 'RAIL_PLAN_HEAD_COMMIT=%s\n' "$head_commit"
} >>"${GITHUB_ENV:?GITHUB_ENV is required}"
printf '%s\n' "$plan_dir" >>"${GITHUB_PATH:?GITHUB_PATH is required}"
