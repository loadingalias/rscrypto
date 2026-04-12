#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

maybe_disable_sccache

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT

FEATURE_SETS=(
  "checksums"
  "alloc,checksums"
  "hashes"
  "alloc,hashes"
  "auth"
  "alloc,auth"
)

echo "Executable rscrypto feature matrix"

for feature_set in "${FEATURE_SETS[@]}"; do
  log_path="$LOG_DIR/$(echo "$feature_set" | tr ',' '_').log"
  step "--no-default-features --features $feature_set"
  if ! cargo test --workspace --lib --tests --no-default-features --features "$feature_set" >"$log_path" 2>&1; then
    fail
    show_error "$log_path"
    exit 1
  fi
  ok
done

echo "${GREEN}✓${RESET} Executable feature matrix passed"
