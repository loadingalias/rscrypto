#!/usr/bin/env bash
# Extract toolchain channel from rust-toolchain.toml.
#
# Usage:
#   scripts/lib/toolchain.sh         # prints channel
#   RUST_TOOLCHAIN=$(scripts/lib/toolchain.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOOLCHAIN_FILE="$REPO_ROOT/rust-toolchain.toml"

if [ ! -f "$TOOLCHAIN_FILE" ]; then
  echo "ERROR: rust-toolchain.toml not found at $TOOLCHAIN_FILE" >&2
  exit 1
fi

RUST_TOOLCHAIN=$(awk -F'"' '/^channel/ {print $2}' "$TOOLCHAIN_FILE")

if [ -z "$RUST_TOOLCHAIN" ]; then
  echo "ERROR: Could not extract toolchain from $TOOLCHAIN_FILE" >&2
  exit 1
fi

echo "$RUST_TOOLCHAIN"
