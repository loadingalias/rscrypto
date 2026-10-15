#!/usr/bin/env bash
# Run repository Python through uv, with the interpreter (.python-version) and packages (uv.lock)
# that the repository pins, from any working directory. A stale uv.lock refuses instead of changing.
# Leading `--group <name>` options add uv dependency groups; `--print` prints the interpreter.

set -euo pipefail

export UV_PROJECT="${UV_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export UV_LOCKED=1
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
command -v uv >/dev/null 2>&1 || {
  echo 'rscrypto scripts run Python through uv; scripts/tooling installs it, or install it with Homebrew.' >&2
  exit 127
}
if [[ "$print_only" == true ]]; then
  exec uv run ${groups[@]+"${groups[@]}"} python -c 'import sys; print(sys.executable)'
fi
exec uv run ${groups[@]+"${groups[@]}"} python -X utf8 "$@"
