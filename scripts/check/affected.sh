#!/usr/bin/env bash
# Create one immutable plan and share it across the normal local proof set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

force_all=false
with_tests=false
for arg in "$@"; do
  case "$arg" in
    --all) force_all=true ;;
    --with-tests) with_tests=true ;;
    *) echo "Usage: $0 [--all] [--with-tests]" >&2; exit 2 ;;
  esac
done

plan_file=$(mktemp "${TMPDIR:-/tmp}/rscrypto-plan-v8.XXXXXX")
trap 'rm -f "$plan_file"' EXIT

plan_args=(rail plan --quiet --json)
[[ "$force_all" == true ]] && plan_args+=(--all)
cargo "${plan_args[@]}" >"$plan_file"
cargo rail plan --verify "$plan_file"
printf 'Cargo Rail plan: %s\n' "$(jq -r '.identity' "$plan_file")"
printf 'Required work: %s\n' "$(jq -r '.required | if length == 0 then "none" else join(", ") end' "$plan_file")"

export RAIL_PLAN_FILE="$plan_file"
export RAIL_PLAN_LOCAL=true

"$SCRIPT_DIR/policy.sh"
"$SCRIPT_DIR/check.sh"

feature_matrix=$("$SCRIPT_DIR/feature-contracts.sh" matrix)
feature_count=$(jq -r '.include | length' <<<"$feature_matrix")
if [[ "$feature_count" -eq 0 ]]; then
  echo "Feature contracts: not required by Cargo Rail"
else
  while IFS=$'\t' read -r domain shard profiles; do
    if [[ "$domain" == runtime && "$with_tests" == false ]]; then
      continue
    fi
    "$SCRIPT_DIR/feature-contracts.sh" selected "$domain" "$shard" "$profiles"
  done < <(jq -r '.include[] | [.domain, .shard, .profiles] | @tsv' <<<"$feature_matrix")
fi

if [[ "$with_tests" == true ]]; then
  "$SCRIPT_DIR/../test/test.sh"

  examples_required=false
  if rail_work_required cargo.build || rail_work_required contracts.examples; then
    examples_required=true
  fi
  if [[ "$examples_required" == true ]]; then
    "$SCRIPT_DIR/../test/test-examples.sh"
  else
    echo "Examples: not required by Cargo Rail"
  fi

  miri_rows=$("$SCRIPT_DIR/../test/miri-contracts.sh" rows)
  if [[ -n "$miri_rows" ]]; then
    "$SCRIPT_DIR/../test/miri-contracts.sh" selected "$miri_rows"
  else
    echo "Miri contracts: not required by Cargo Rail"
  fi

  fuzz_rows=$("$SCRIPT_DIR/../test/fuzz-contracts.sh" rows)
  if [[ -n "$fuzz_rows" ]]; then
    "$SCRIPT_DIR/../test/fuzz-contracts.sh" selected "$fuzz_rows"
  else
    echo "Fuzz contracts: not required by Cargo Rail"
  fi
fi
