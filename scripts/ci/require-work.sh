#!/usr/bin/env bash
# Assert that a direct executor is authorized by the active saved plan.

set -euo pipefail

mode=all
if [[ ${1:-} == --any ]]; then
  mode=any
  shift
fi
[[ $# -gt 0 ]] || { echo "Usage: $0 [--any] WORK_ID..." >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/rail-plan.sh
source "$SCRIPT_DIR/../lib/rail-plan.sh"
rail_prime_plan

required_count=0
for work_id in "$@"; do
  status=0
  rail_work_required "$work_id" || status=$?
  case "$status" in
    0) required_count=$((required_count + 1)) ;;
    1)
      if [[ "$mode" == all ]]; then
        echo "Cargo Rail did not authorize $work_id" >&2
        exit 2
      fi
      ;;
    *) exit "$status" ;;
  esac
done

if [[ "$mode" == any && "$required_count" -eq 0 ]]; then
  echo "Cargo Rail did not authorize any requested work" >&2
  exit 2
fi
