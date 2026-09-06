#!/usr/bin/env bash
# Build and validate the bounded constant-time structure gate.

set -euo pipefail

[[ $# -eq 0 ]] || {
  echo "Usage: $0" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

target=$(rustc -vV | awk -F': ' '/^host:/ { print $2 }')
[[ "$target" == x86_64-unknown-linux-gnu ]] || {
  echo "CT structural gate requires an x86_64-unknown-linux-gnu host; found $target" >&2
  exit 2
}

"$SCRIPT_DIR/artifacts.sh" --target "$target" --profile release
"$REPO_ROOT/scripts/lib/python.sh" "$SCRIPT_DIR/validate.py" \
  --target "$target" \
  --profile release \
  --strict-coverage
