#!/usr/bin/env bash
# Regression tests for cross-platform Python executable discovery.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fixture="$(mktemp -d)"
trap 'rm -rf "$fixture"' EXIT

mkdir -p "$fixture/bin"
printf '%s\n' '#!/bin/sh' 'exit 0' >"$fixture/bin/python.exe"
chmod +x "$fixture/bin/python.exe"

resolved="$(BASH_ENV= PYTHON= PATH="$fixture/bin" /bin/bash "$SCRIPT_DIR/python.sh" --print)"
[[ "$resolved" == "$fixture/bin/python.exe" ]] || {
  echo "python resolver did not select the Windows executable name: $resolved" >&2
  exit 1
}

printf '%s\n' '#!/bin/sh' 'exit 1' >"$fixture/bin/python.exe"
if BASH_ENV= PYTHON= PATH="$fixture/bin" /bin/bash "$SCRIPT_DIR/python.sh" --print >/dev/null 2>&1; then
  echo 'python resolver accepted an unsupported Python runtime' >&2
  exit 1
fi

echo 'python resolver test: pass'
