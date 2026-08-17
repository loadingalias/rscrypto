#!/usr/bin/env bash
# Extract toolchain channel from rust-toolchain.toml.
#
# Usage:
#   scripts/lib/toolchain.sh          # prints the development channel
#   scripts/lib/toolchain.sh --msrv   # prints the package MSRV
#   scripts/lib/toolchain.sh --nightly # prints the pinned nightly channel

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOOLCHAIN_FILE="$REPO_ROOT/rust-toolchain.toml"
CARGO_MANIFEST="$REPO_ROOT/Cargo.toml"
TOOLCHAIN_CONTRACTS="$REPO_ROOT/.config/toolchains.toml"

case "${1:-}" in
  "")
    SOURCE_FILE="$TOOLCHAIN_FILE"
    FIELD_PATTERN='^channel[[:space:]]*='
    ;;
  --msrv)
    SOURCE_FILE="$CARGO_MANIFEST"
    FIELD_PATTERN='^rust-version[[:space:]]*='
    ;;
  --nightly)
    SOURCE_FILE="$TOOLCHAIN_CONTRACTS"
    FIELD_PATTERN='^nightly[[:space:]]*='
    ;;
  *)
    echo "Usage: $0 [--msrv|--nightly]" >&2
    exit 2
    ;;
esac

if [[ ! -f "$SOURCE_FILE" ]]; then
  echo "ERROR: toolchain authority not found: $SOURCE_FILE" >&2
  exit 1
fi

RUST_TOOLCHAIN=$(awk -F'"' -v pattern="$FIELD_PATTERN" '$0 ~ pattern { print $2; exit }' "$SOURCE_FILE")

if [[ -z "$RUST_TOOLCHAIN" ]]; then
  echo "ERROR: could not extract toolchain from $SOURCE_FILE" >&2
  exit 1
fi

echo "$RUST_TOOLCHAIN"
