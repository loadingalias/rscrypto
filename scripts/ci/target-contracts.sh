#!/usr/bin/env bash
# Materialize and execute Cargo Rail-selected platform proof rows.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CATALOG="$REPO_ROOT/.config/target-matrix.json"
VALIDATOR="$SCRIPT_DIR/../lib/target-matrix.sh"

usage() {
  echo "Usage: $0 matrix PLAN READER | run ROW [shallow|deep]" >&2
  exit 2
}

full_matrix() {
  jq -c '{include: [.variants[]
    | select(.dimensions.operation != "core")
    | {id: .id} + .dimensions]}' "$CATALOG"
}

materialize_matrix() {
  [[ $# -eq 2 && -f "$1" && -f "$2" ]] || usage
  local plan=$1
  local reader=$2
  local selected
  selected=$(python3 "$reader" matrix "$plan" targets.platforms)
  if [[ "$selected" == all ]]; then
    full_matrix
    return
  fi

  jq -ce '
    .include |= map(select(.operation != "core"))
    | select((.include | type) == "array")
  ' <<<"$selected"
}

run_row() {
  [[ $# -ge 1 && $# -le 2 ]] || usage
  local id=$1
  local depth=${2:-deep}
  [[ "$id" =~ ^[a-z][a-z0-9.-]*$ ]] || usage
  [[ "$depth" == shallow || "$depth" == deep ]] || usage

  local row operation target platform contract toolchain
  row=$(jq -ce --arg id "$id" '.variants[] | select(.id == $id)' "$CATALOG") || {
    echo "unknown platform proof row: $id" >&2
    exit 2
  }
  operation=$(jq -r '.dimensions.operation' <<<"$row")
  target=$(jq -r '.dimensions.target' <<<"$row")
  platform=$(jq -r '.dimensions.platform' <<<"$row")
  contract=$(jq -r '.dimensions.contract' <<<"$row")
  case "$contract" in
    development) toolchain=$("$SCRIPT_DIR/../lib/toolchain.sh") ;;
    nightly) toolchain=$("$SCRIPT_DIR/../lib/toolchain.sh" --nightly) ;;
    *)
      echo "unsupported platform toolchain contract: $contract" >&2
      exit 2
      ;;
  esac
  export RUSTUP_TOOLCHAIN="$toolchain"

  case "$operation" in
    cross) "$SCRIPT_DIR/cross-targets.sh" "$target" "$depth" ;;
    native) "$SCRIPT_DIR/native-platform.sh" "$platform" "$target" "$depth" ;;
    amx) "$SCRIPT_DIR/native-platform.sh" amx "$target" "$depth" ;;
    core)
      echo "$id is owned by the core Rust job and has no separate executor" >&2
      exit 2
      ;;
    *)
      echo "unsupported platform proof operation: $operation" >&2
      exit 2
      ;;
  esac
}

"$VALIDATOR" --validate
case "${1:-}" in
  matrix)
    shift
    materialize_matrix "$@"
    ;;
  run)
    shift
    run_row "$@"
    ;;
  *) usage ;;
esac
