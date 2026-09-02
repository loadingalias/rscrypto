#!/usr/bin/env bash
# Compile the public library contract with the exact Cargo.toml MSRV.

set -euo pipefail

[[ $# -eq 0 ]] || { echo "Usage: $0" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
toolchain=$("$SCRIPT_DIR/../lib/toolchain.sh" --msrv)

if ! rustup run "$toolchain" rustc --version >/dev/null 2>&1; then
  echo "MSRV toolchain $toolchain is not installed" >&2
  echo "Install it with: rustup toolchain install $toolchain --profile minimal --no-self-update" >&2
  exit 1
fi

cd "$REPO_ROOT"
export RUSTUP_TOOLCHAIN="$toolchain"
rustc --version --verbose
cargo check --locked --workspace --lib --no-default-features
cargo check --locked --workspace --lib --all-features
