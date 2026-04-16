#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

maybe_disable_sccache

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT
TARGET_DIR="$LOG_DIR/target"

FEATURE_SETS=(
  "crc16"
  "crc24"
  "crc32"
  "crc64"
  "checksums"
  "alloc,checksums"
  "std,checksums"
  "hashes"
  "alloc,hashes"
  "std,hashes"
  "macs"
  "kdfs"
  "signatures"
  "key-exchange"
  "auth"
  "alloc,auth"
  "std,auth"
  "aead"
  "alloc,aead"
  "std,aead"
  "std,checksums,hashes"
  "std,checksums,aead"
  "std,hashes,aead"
  "std,full"
)

# RISC-V: nightly rustc crashes (SIGABRT in glibc allocator) when linking
# per-feature test binaries on the RISE runner. Downgrade to cargo-check so
# we still verify every feature combination compiles; full test coverage is
# provided by the --all-features test job.
ARCH=$(uname -m)
if [[ "$ARCH" == "riscv64" ]]; then
  CARGO_CMD="cargo check --workspace --lib --tests"
  echo "Compilation rscrypto feature matrix (riscv64: check-only)"
else
  CARGO_CMD="cargo test --workspace --lib --tests"
  echo "Executable rscrypto feature matrix"
fi

for feature_set in "${FEATURE_SETS[@]}"; do
  log_path="$LOG_DIR/$(echo "$feature_set" | tr ',' '_').log"
  step "--no-default-features --features $feature_set"
  # Isolate reduced-feature test builds from the workspace target dir. The
  # commit lane runs full-feature and no_std checks first, and sharing the same
  # restored target cache has produced flaky matrix failures in CI.
  if ! CARGO_TARGET_DIR="$TARGET_DIR" $CARGO_CMD --no-default-features --features "$feature_set" >"$log_path" 2>&1; then
    fail
    show_error "$log_path"
    exit 1
  fi
  ok
done

echo "${GREEN}✓${RESET} Executable feature matrix passed"
