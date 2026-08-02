#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/feature-profiles.sh
source "$SCRIPT_DIR/../lib/feature-profiles.sh"

maybe_disable_sccache

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT
TOTAL=${#COMPILE_FEATURE_SETS[@]}
STARTED_AT=$SECONDS

echo "Compilation feature matrix ($TOTAL profiles)"

for i in "${!COMPILE_FEATURE_SETS[@]}"; do
  feature_set=${COMPILE_FEATURE_SETS[$i]}
  profile=$((i + 1))
  profile_started_at=$SECONDS
  display=${feature_set:-no-features}
  log_path="$LOG_DIR/${display//,/_}.log"

  step "[$profile/$TOTAL] cargo check --lib --tests --no-default-features --features $display"
  args=(check --locked --workspace --lib --tests --no-default-features)
  if [[ -n "$feature_set" ]]; then
    args+=(--features "$feature_set")
  fi

  if ! cargo "${args[@]}" >"$log_path" 2>&1; then
    fail
    show_error "$log_path"
    exit 1
  fi

  ok
  echo "    elapsed: $((SECONDS - profile_started_at))s"
done

echo "${GREEN}✓${RESET} Compilation feature matrix passed: $TOTAL/$TOTAL profiles in $((SECONDS - STARTED_AT))s"
