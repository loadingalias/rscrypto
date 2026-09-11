#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
  echo 'macOS CI requires an Apple Silicon runner' >&2
  exit 1
}
python3 scripts/lib/toolchain.py --install aarch64-apple-darwin
channel=$(scripts/lib/toolchain.sh)
while IFS= read -r tool; do
  version=$(python3 scripts/tooling/catalog.py get cargo "$tool")
  cargo "+$channel" install --locked --version "$version" "$tool"
done < <(python3 scripts/tooling/catalog.py get ci cargo)
