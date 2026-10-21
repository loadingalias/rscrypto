#!/usr/bin/env bash
# Run repository Python through uv, with the interpreter (.python-version) and packages (uv.lock)
# that the repository pins, from any working directory. A stale uv.lock refuses instead of changing.
# Leading `--group <name>` options add uv dependency groups; `--print` prints the interpreter.

set -euo pipefail

# Always this checkout: `uv run` exports UV_PROJECT, and an inherited value would run another
# checkout's lock and environment, which may since have been removed.
UV_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export UV_PROJECT UV_LOCKED=1
groups=()
while [[ "${1:-}" == --group ]]; do
  groups+=(--group "${2:?--group requires a dependency group}")
  shift 2
done
print_only=false
if [[ "${1:-}" == "--print" ]]; then
  print_only=true
  shift
fi

# Tests pin the interpreter that runs them, which uv selected.
if [[ -n "${PYTHON:-}" ]]; then
  if [[ "$print_only" == true ]]; then printf '%s\n' "$PYTHON"; exit 0; fi
  exec "$PYTHON" -X utf8 "$@"
fi
if [[ "$(uname -sm)" == "Darwin arm64" ]]; then
  # The Mac runs the catalog's pinned uv, kept off PATH in the layout of scripts/tooling/uv.sh,
  # which installs it on first use.
  sha="$(sed -n '/^\[aarch64-macos\.assets\.uv\]$/,/^\[/s/^sha256 = "\([0-9a-f]*\)"$/\1/p' "$UV_PROJECT/.config/tooling.toml")"
  uv="$HOME/.local/share/rscrypto-tooling/uv/${sha:0:16}/uv"
  [[ -f "${uv%/uv}/.rscrypto-installed" ]] || "$UV_PROJECT/scripts/tooling/uv.sh" aarch64-macos >/dev/null
else
  uv=uv
  command -v uv >/dev/null 2>&1 || {
    echo 'rscrypto scripts run Python through uv; the scripts/tooling installers provide it.' >&2
    exit 127
  }
fi
if [[ "$print_only" == true ]]; then
  exec "$uv" run ${groups[@]+"${groups[@]}"} python -c 'import sys; print(sys.executable)'
fi
exec "$uv" run ${groups[@]+"${groups[@]}"} python -X utf8 "$@"
