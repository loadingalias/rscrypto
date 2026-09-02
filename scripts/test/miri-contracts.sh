#!/usr/bin/env bash
# Materialize and execute Cargo Rail-selected Miri proof modes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATALOG="$SCRIPT_DIR/../../.config/miri-matrix.json"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

usage() {
  echo "Usage: $0 rows [PLAN READER] | selected ROWS | run ROW | list" >&2
  exit 2
}

validate_catalog() {
  jq -e '
    .variant_catalog_version == 2
    and .work == "assurance.miri"
    and (.variants | type == "array" and length > 0)
    and ([.variants[].id] | length == (unique | length))
    and ([.variants[].dimensions.mode] | length == (unique | length))
    and all(.variants[];
      (.id | test("^[a-z][a-z0-9.-]*$"))
      and (.dimensions | keys | sort) == ["mode", "name"]
      and (.dimensions.name | type == "string" and length > 0)
      and (.dimensions.mode == "focused" or .dimensions.mode == "rsa")
      and (.external_paths | type == "array" and length > 0 and length == (unique | length))
      and all(.external_paths[]; type == "string" and length > 0)
    )
  ' "$CATALOG" >/dev/null || {
    echo "Miri variant catalog is malformed" >&2
    return 2
  }
}

row_exists() {
  jq -e --arg id "$1" 'any(.variants[]; .id == $id)' "$CATALOG" >/dev/null
}

parse_rows() {
  local value=$1
  local row seen=,
  [[ -n "$value" ]] || {
    echo "selected Miri rows must not be empty" >&2
    return 2
  }
  IFS=',' read -r -a MIRI_ROWS <<<"$value"
  for row in "${MIRI_ROWS[@]}"; do
    if [[ ! "$row" =~ ^[a-z][a-z0-9.-]*$ ]] || ! row_exists "$row"; then
      echo "unknown Miri row: ${row:-<empty>}" >&2
      return 2
    fi
    [[ "$seen" != *",$row,"* ]] || {
      echo "duplicate Miri row: $row" >&2
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
  matrix=$(rail_variant_matrix assurance.miri)
  if [[ "$matrix" == all ]]; then
    jq -r '[.variants[].id] | join(",")' "$CATALOG"
  else
    jq -er '[.include[].id] | join(",")' <<<"$matrix"
  fi
}

run_row() {
  [[ $# -eq 1 ]] || usage
  local row=$1
  row_exists "$row" || {
    echo "unknown Miri row: $row" >&2
    exit 2
  }
  local mode
  mode=$(jq -r --arg id "$row" '.variants[] | select(.id == $id) | .dimensions.mode' "$CATALOG")
  "$SCRIPT_DIR/test-miri.sh" "--$mode"
}

run_selected() {
  [[ $# -eq 1 ]] || usage
  parse_rows "$1"
  local row
  for row in "${MIRI_ROWS[@]}"; do
    run_row "$row"
  done
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
    jq -r '.variants[] | [.id, .dimensions.mode, .dimensions.name] | @tsv' "$CATALOG"
    ;;
  *) usage ;;
esac
