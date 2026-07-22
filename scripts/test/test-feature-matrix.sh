#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

maybe_disable_sccache

LOG_DIR=$(mktemp -d)
TARGET_DIR="$REPO_ROOT/target/feature-matrix"

cleanup() {
  rm -rf "$LOG_DIR" "$TARGET_DIR"
}
trap cleanup EXIT
rm -rf "$TARGET_DIR"

cleanup_feature_artifacts() {
  cargo clean --target-dir "$TARGET_DIR" -p rscrypto >/dev/null 2>&1 || true
}

show_feature_matrix_disk() {
  if [[ "${CI:-}" == "true" || -n "${GITHUB_ACTIONS:-}" ]]; then
    df -h "$LOG_DIR" | sed 's/^/    /'
  fi
}

FEATURE_SETS=(
  "crc16"
  "crc24"
  "crc32"
  "crc64"
  "blake2b"
  "blake2s"
  "pbkdf2"
  "checksums"
  "alloc,checksums"
  "std,checksums"
  "std,checksums,diag"
  "hashes"
  "alloc,hashes"
  "std,hashes"
  "std,hashes,diag"
  "macs"
  "kdfs"
  "rsa"
  "rsa,getrandom"
  "signatures"
  "key-exchange"
  "auth"
  "aegis256"
  "alloc,auth"
  "std,auth"
  "std,password-hashing,getrandom"
  "aead"
  "alloc,aead"
  "std,aead"
  "std,aead,diag"
  "std,checksums,hashes"
  "std,checksums,aead"
  "std,hashes,aead"
  "std,full"
  "std,full,serde"
  "std,full,serde-secrets"
  "std,parallel"
  "std,full,portable-only"
)

TOTAL=${#FEATURE_SETS[@]}
STARTED_AT=$SECONDS

# RISC-V: nightly rustc crashes (SIGABRT in glibc allocator) when linking
# per-feature test binaries on the RISE runner. Downgrade to cargo-check so
# we still verify every feature combination compiles; full test coverage is
# provided by the --all-features test job.
ARCH=$(uname -m)
if [[ "$ARCH" == "riscv64" ]]; then
  CARGO_CMD="cargo check --locked --workspace --lib --tests"
  COMMAND_CLASS="cargo check"
  echo "Compilation rscrypto feature matrix ($TOTAL profiles; riscv64: check-only)"
else
  CARGO_CMD="cargo test --locked --workspace --lib --tests"
  COMMAND_CLASS="cargo test"
  echo "Executable rscrypto feature matrix ($TOTAL profiles)"
fi

for i in "${!FEATURE_SETS[@]}"; do
  feature_set=${FEATURE_SETS[$i]}
  profile=$((i + 1))
  profile_started_at=$SECONDS
  log_path="$LOG_DIR/$(echo "$feature_set" | tr ',' '_').log"
  step "[$profile/$TOTAL] $COMMAND_CLASS --no-default-features --features $feature_set"
  # Isolate reduced-feature test builds from the workspace target dir. The
  # commit lane runs full-feature and no_std checks first, and sharing the same
  # restored target cache has produced flaky matrix failures in CI.
  if ! CARGO_TARGET_DIR="$TARGET_DIR" $CARGO_CMD --no-default-features --features "$feature_set" >"$log_path" 2>&1; then
    fail
    show_error "$log_path"
    show_feature_matrix_disk
    exit 1
  fi
  cleanup_feature_artifacts
  ok
  echo "    elapsed: $((SECONDS - profile_started_at))s"
done

echo "${GREEN}✓${RESET} Feature matrix passed: $TOTAL/$TOTAL profiles in $((SECONDS - STARTED_AT))s"
