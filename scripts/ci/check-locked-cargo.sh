#!/usr/bin/env bash
set -euo pipefail

ROOT=""
if [[ ${1:-} == --root ]]; then
  ROOT=${2:?missing path after --root}
  shift 2
fi
if [[ $# -ne 0 ]]; then
  echo "usage: check-locked-cargo.sh [--root PATH]" >&2
  exit 2
fi

if [[ -z "$ROOT" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

status=0
check_statement() {
  local file=$1
  local line_number=$2
  local statement=$3
  local trimmed=${statement#"${statement%%[![:space:]]*}"}

  case "$trimmed" in
    echo\ * | printf\ * | step\ *) return ;;
  esac

  if [[ "$statement" =~ (^|[[:space:]])cargo[[:space:]]+(build|check|clippy|test|bench|rustc|run|doc|miri|nextest)($|[[:space:]]) ]] \
    && [[ "$statement" != *"--locked"* ]]; then
    echo "$file:$line_number: routine Cargo command must use --locked" >&2
    echo "  $statement" >&2
    status=1
  fi

  if [[ "$statement" =~ (^|[[:space:]])cargo[[:space:]]+llvm-cov[[:space:]]+(nextest|test)($|[[:space:]]) ]] \
    && [[ "$statement" != *"--locked"* ]]; then
    echo "$file:$line_number: cargo-llvm-cov execution must use --locked" >&2
    echo "  $statement" >&2
    status=1
  fi
}

while IFS= read -r file; do
  statement=""
  statement_line=0
  line_number=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line_number=$((line_number + 1))
    if [[ -z "$statement" ]]; then
      [[ "$line" =~ ^[[:space:]]*# ]] && continue
      statement_line=$line_number
    fi
    statement+=" ${line%\\}"
    if [[ "$line" == *\\ ]]; then
      continue
    fi
    check_statement "${file#"$ROOT"/}" "$statement_line" "$statement"
    statement=""
  done <"$file"
  if [[ -n "$statement" ]]; then
    check_statement "${file#"$ROOT"/}" "$statement_line" "$statement"
  fi
done < <(
  {
    [[ -f "$ROOT/justfile" ]] && printf '%s\n' "$ROOT/justfile"
    find "$ROOT/scripts" -type f -name '*.sh' \
      ! -name '*-test.sh' ! -name '*-scheduler-test.sh' 2>/dev/null || true
    find "$ROOT/.github" -type f \( -name '*.yaml' -o -name '*.yml' \) 2>/dev/null || true
  } | LC_ALL=C sort
)

if [[ -f "$ROOT/.zed/tasks.json" ]]; then
  while IFS= read -r task; do
    command=$(jq -r '.command' <<<"$task")
    args=$(jq -r '(.args // []) | join(" ")' <<<"$task")
    check_statement ".zed/tasks.json" 1 "$command $args"
  done < <(jq -c '.[] | select((.command // "") == "cargo" or ((.command // "") | startswith("cargo ")))' \
    "$ROOT/.zed/tasks.json")
fi

exit "$status"
