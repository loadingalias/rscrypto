#!/usr/bin/env bash
# Build one Criterion benchmark and record its profiling window with samply.

set -euo pipefail

BENCH=${1:?usage: profile.sh <bench> [filter] [seconds]}
FILTER=${2:-}
PROFILE_SECONDS=${3:-10}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CATALOG="$SCRIPT_DIR/benchmark_catalog.py"
PYTHON="$("$SCRIPT_DIR/../lib/python.sh" --print)"

if [[ ! "$PROFILE_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || ! awk -v seconds="$PROFILE_SECONDS" 'BEGIN { exit !(seconds >= 1) }'; then
  echo "error: profile duration must be numeric and at least one second" >&2
  exit 2
fi

command -v samply >/dev/null 2>&1 || {
  echo "error: samply is required; install the repository-pinned profile tool set" >&2
  exit 1
}
command -v jq >/dev/null 2>&1 || {
  echo "error: jq is required to resolve the benchmark executable" >&2
  exit 1
}

cd "$REPO_ROOT"
"$PYTHON" "$CATALOG" require-kind "$BENCH" criterion
FEATURES=$("$PYTHON" "$CATALOG" features "$BENCH")
BINARY=$("$PYTHON" "$CATALOG" binary "$BENCH")

ARTIFACT=$(
  cargo bench --locked --profile bench --features "$FEATURES" --bench "$BINARY" \
    --no-run --message-format=json \
    | jq -r --arg name "$BINARY" '
        select(.reason == "compiler-artifact")
        | select(.target.name == $name)
        | select(.target.kind | index("bench"))
        | .executable // empty
      ' \
    | tail -n 1
)

[[ -n "$ARTIFACT" && -x "$ARTIFACT" ]] || {
  echo "error: Cargo did not report an executable for benchmark $BINARY" >&2
  exit 1
}

COMMAND=("$ARTIFACT" --bench)
[[ -n "$FILTER" ]] && COMMAND+=("$FILTER")
COMMAND+=(--profile-time "$PROFILE_SECONDS" --noplot)

PROFILE_DIR="$REPO_ROOT/target/profiles"
PROFILE_PATH="$PROFILE_DIR/${BENCH}-$(date -u +%Y%m%dT%H%M%SZ).json.gz"
mkdir -p "$PROFILE_DIR"

echo "Profiling: ${COMMAND[*]}"
samply record --save-only --output "$PROFILE_PATH" "${COMMAND[@]}"
echo "Profile: $PROFILE_PATH"
