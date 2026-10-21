#!/usr/bin/env bash
# Print the zig that cross-links rscrypto's reviewed binaries. The Apple Silicon workstation runs the
# catalog's pinned archive, installed on first use and kept off PATH; other hosts run the zig on PATH,
# which their installers provide. Exit 127 when there is none.
set -euo pipefail

if [[ "$(uname -sm)" != "Darwin arm64" ]]; then
  command -v zig || exit 127
  exit 0
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
directory="$("$ROOT/scripts/lib/python.sh" "$ROOT/scripts/tooling/catalog.py" install-archive aarch64-macos zig \
  "$HOME/.local/share/rscrypto-tooling")"
printf '%s/zig\n' "$directory"
