#!/usr/bin/env bash
set -euo pipefail

PRINT=0
if [[ "${1:-}" == "--print" ]]; then
  PRINT=1
  shift
fi

CANDIDATES=()
if [[ -n "${PYTHON:-}" ]]; then
  CANDIDATES+=("$PYTHON")
fi
CANDIDATES+=(python3.14 python3.13 python3.12 python3.11 python3)

SEEN=":"
for candidate in "${CANDIDATES[@]}"; do
  if [[ "$candidate" == */* ]]; then
    resolved="$candidate"
    [[ -x "$resolved" ]] || continue
  else
    resolved="$(command -v "$candidate" 2>/dev/null || true)"
    [[ -n "$resolved" ]] || continue
  fi

  case "$SEEN" in
    *":$resolved:"*) continue ;;
  esac
  SEEN="$SEEN$resolved:"

  if "$resolved" - <<'PY' >/dev/null 2>&1; then
import sys

if sys.version_info < (3, 10):
  raise SystemExit(1)

try:
  import tomllib  # noqa: F401
except ModuleNotFoundError:
  import tomli  # noqa: F401
PY
    if [[ "$PRINT" == "1" ]]; then
      printf '%s\n' "$resolved"
    else
      export PYTHON="$resolved"
      exec "$resolved" "$@"
    fi
    exit 0
  fi
done

cat >&2 <<'EOF'
rscrypto CT scripts require Python 3.10+ with TOML support.

Use Python 3.11+ for stdlib tomllib, install tomli for Python 3.10,
or set PYTHON=/path/to/python3.11.
EOF
exit 1
