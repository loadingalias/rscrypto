#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# uv runs every Python command with the interpreter and packages that the repository pins.
export UV_PROJECT="$PWD" UV_LOCKED=1
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
  echo 'macOS CI requires an Apple Silicon runner' >&2
  exit 1
}
uv run python scripts/lib/toolchain.py --install
channel=$(scripts/lib/toolchain.sh)
while IFS= read -r tool; do
  version=$(uv run python scripts/tooling/catalog.py get cargo "$tool")
  cargo "+$channel" install --locked --version "$version" "$tool"
done < <(uv run python scripts/tooling/catalog.py get ci cargo)
while IFS= read -r tool; do
  case "$tool" in
    cargo-*) cargo "+$channel" "${tool#cargo-}" --version ;;
    ripgrep) rg --version ;;
    *) "$tool" --version ;;
  esac
done < <(uv run python scripts/tooling/catalog.py get ci cargo)
