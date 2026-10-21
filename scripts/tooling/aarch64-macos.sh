#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
  echo 'macOS CI requires an Apple Silicon runner' >&2
  exit 1
}
# The catalog's pinned uv runs every Python command with the interpreter and packages that the
# repository pins; the first command installs it.
python=scripts/lib/python.sh
"$python" scripts/lib/toolchain.py --install
channel=$(scripts/lib/toolchain.sh)
while IFS= read -r tool; do
  version=$("$python" scripts/tooling/catalog.py get cargo "$tool")
  cargo "+$channel" install --locked --version "$version" "$tool"
done < <("$python" scripts/tooling/catalog.py get ci cargo)
while IFS= read -r tool; do
  case "$tool" in
    cargo-*) cargo "+$channel" "${tool#cargo-}" --version ;;
    ripgrep) rg --version ;;
    *) "$tool" --version ;;
  esac
done < <("$python" scripts/tooling/catalog.py get ci cargo)
