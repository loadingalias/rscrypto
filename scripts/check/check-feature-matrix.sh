#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

maybe_disable_sccache

FEATURE_SETS=(
  ""
  "alloc"
  "crc16"
  "crc24"
  "crc32"
  "crc64"
  "alloc,crc32"
  "sha2"
  "sha3"
  "xxh3"
  "hmac"
  "hmac-sha3"
  "kmac"
  "hkdf"
  "poly1305"
  "rsa"
  "rsa,getrandom"
  "x25519"
  "chacha20poly1305"
  "ascon-aead"
  "checksums"
  "hashes"
  "macs"
  "kdfs"
  "signatures"
  "key-exchange"
  "auth"
  "aead"
  "full"
)

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT
TOTAL=${#FEATURE_SETS[@]}
STARTED_AT=$SECONDS

echo "Compilation feature matrix ($TOTAL profiles)"

for i in "${!FEATURE_SETS[@]}"; do
  feature_set=${FEATURE_SETS[$i]}
  profile=$((i + 1))
  profile_started_at=$SECONDS
  display=${feature_set:-no-features}
  log_path="$LOG_DIR/${display//,/_}.log"

  step "[$profile/$TOTAL] cargo check --no-default-features --features $display"
  args=(check -p rscrypto --no-default-features --lib)
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
