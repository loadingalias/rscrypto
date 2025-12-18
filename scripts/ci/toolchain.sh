#!/usr/bin/env bash
# Extract toolchain version from rust-toolchain.toml
#
# Usage:
#   source scripts/ci/toolchain.sh  # Sets RUST_TOOLCHAIN env var
#   scripts/ci/toolchain.sh         # Prints toolchain version

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOOLCHAIN_FILE="$REPO_ROOT/rust-toolchain.toml"

if [ ! -f "$TOOLCHAIN_FILE" ]; then
  echo "ERROR: rust-toolchain.toml not found at $TOOLCHAIN_FILE" >&2
  exit 1
fi

# Extract channel value (e.g., "nightly-2025-12-14")
# Uses awk for cross-platform compatibility (macOS sed differs from GNU sed)
RUST_TOOLCHAIN=$(awk -F'"' '/^channel/ {print $2}' "$TOOLCHAIN_FILE")

if [ -z "$RUST_TOOLCHAIN" ]; then
  echo "ERROR: Could not extract toolchain from $TOOLCHAIN_FILE" >&2
  exit 1
fi

export RUST_TOOLCHAIN
echo "$RUST_TOOLCHAIN"
