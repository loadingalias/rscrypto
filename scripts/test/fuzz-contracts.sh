#!/usr/bin/env bash
# Materialize and execute Cargo Rail-selected fuzz target groups.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CATALOG="$REPO_ROOT/.config/fuzz-matrix.json"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

usage() {
  echo "Usage: $0 rows [PLAN READER] | selected ROWS | run ROW | list" >&2
  exit 2
}

validate_catalog() {
  jq -e '
    .variant_catalog_version == 2
    and .work == "assurance.fuzz"
    and (.variants | type == "array" and length > 0)
    and ([.variants[].id] | length == (unique | length))
    and ([.variants[].dimensions.targets | split(",")[]]
      | length == (unique | length))
    and all(.variants[];
      (.id | test("^[a-z][a-z0-9.-]*$"))
      and (.dimensions | keys | sort) == ["duration_seconds", "name", "targets"]
      and (.dimensions.name | type == "string" and length > 0)
      and (.dimensions.targets | test("^[a-z0-9_]+(,[a-z0-9_]+)*$"))
      and (.dimensions.duration_seconds | type == "number" and . > 0 and floor == .)
      and (.external_paths | type == "array" and length > 0 and length == (unique | length))
      and all(.external_paths[]; type == "string" and length > 0)
    )
  ' "$CATALOG" >/dev/null || {
    echo "fuzz variant catalog is malformed" >&2
    return 2
  }

  local known_targets catalog_targets
  known_targets=$(mktemp "${TMPDIR:-/tmp}/rscrypto-fuzz-targets.XXXXXX")
  catalog_targets="$known_targets.catalog"
  awk '
    /^\[\[bin\]\]$/ { in_bin = 1; next }
    in_bin && /^name = "/ {
      value = $0
      sub(/^name = "/, "", value)
      sub(/".*$/, "", value)
      print value
      in_bin = 0
    }
  ' "$REPO_ROOT/fuzz/Cargo.toml" "$REPO_ROOT"/fuzz-packages/*/Cargo.toml \
    | sort -u >"$known_targets"

  jq -r '.variants[].dimensions.targets | split(",")[]' "$CATALOG" \
    | sort -u >"$catalog_targets"
  if ! cmp -s "$known_targets" "$catalog_targets"; then
    echo "fuzz variant catalog and declared fuzz targets differ:" >&2
    comm -3 "$known_targets" "$catalog_targets" >&2
    rm -f "$known_targets" "$catalog_targets"
    return 2
  fi
  rm -f "$known_targets" "$catalog_targets"
}

row_exists() {
  jq -e --arg id "$1" 'any(.variants[]; .id == $id)' "$CATALOG" >/dev/null
}

parse_rows() {
  local value=$1
  local row seen=,
  [[ -n "$value" ]] || {
    echo "selected fuzz rows must not be empty" >&2
    return 2
  }
  IFS=',' read -r -a FUZZ_ROWS <<<"$value"
  for row in "${FUZZ_ROWS[@]}"; do
    if [[ ! "$row" =~ ^[a-z][a-z0-9.-]*$ ]] || ! row_exists "$row"; then
      echo "unknown fuzz row: ${row:-<empty>}" >&2
      return 2
    fi
    [[ "$seen" != *",$row,"* ]] || {
      echo "duplicate fuzz row: $row" >&2
      return 2
    }
    seen+="$row,"
  done
}

selected_rows() {
  [[ $# -eq 0 || $# -eq 2 ]] || usage
  if [[ $# -eq 2 ]]; then
    [[ -f "$1" && -f "$2" ]] || usage
    export RAIL_PLAN_FILE=$1
    export RAIL_PLAN_READER=$2
  fi

  local matrix
  matrix=$(rail_variant_matrix assurance.fuzz)
  if [[ "$matrix" == all ]]; then
    jq -r '[.variants[].id] | join(",")' "$CATALOG"
  else
    jq -er '[.include[].id] | join(",")' <<<"$matrix"
  fi
}

run_selected() {
  [[ $# -eq 1 ]] || usage
  parse_rows "$1"

  local selection
  selection=$(jq -cer --arg rows "$1" '
    ($rows | split(",")) as $wanted
    | [.variants[] | select(.id as $id | $wanted | index($id))]
    | {
        duration: ([.[].dimensions.duration_seconds] | max),
        targets: ([.[].dimensions.targets | split(",")[]]
          | reduce .[] as $target ([];
              if index($target) then . else . + [$target] end)
          | join(","))
      }
  ' "$CATALOG")
  local duration targets
  duration=$(jq -r '.duration' <<<"$selection")
  targets=$(jq -r '.targets' <<<"$selection")
  RSCRYPTO_FUZZ_DURATION_SECS="$duration" "$SCRIPT_DIR/test-fuzz.sh" --targets "$targets"
}

run_row() {
  [[ $# -eq 1 ]] || usage
  row_exists "$1" || {
    echo "unknown fuzz row: $1" >&2
    exit 2
  }
  run_selected "$1"
}

validate_catalog
case "${1:-}" in
  rows)
    shift
    selected_rows "$@"
    ;;
  selected)
    shift
    run_selected "$@"
    ;;
  run)
    shift
    run_row "$@"
    ;;
  list)
    [[ $# -eq 1 ]] || usage
    jq -r '.variants[] | [.id, .dimensions.targets, (.dimensions.duration_seconds | tostring), .dimensions.name] | @tsv' "$CATALOG"
    ;;
  *) usage ;;
esac
