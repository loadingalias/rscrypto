#!/usr/bin/env bash
# Resolve the supported Python runtime for repository tooling.

set -euo pipefail

print_only=false
if [[ "${1:-}" == "--print" ]]; then
  print_only=true
  shift
fi

candidates=()
if [[ -n "${PYTHON:-}" ]]; then
  candidates+=("$PYTHON")
fi
candidates+=(python3.14 python3.13 python3.12 python3.11 python3)

seen=":"
for candidate in "${candidates[@]}"; do
  if [[ "$candidate" == */* ]]; then
    resolved="$candidate"
    [[ -x "$resolved" ]] || continue
  else
    resolved="$(command -v "$candidate" 2>/dev/null || true)"
    [[ -n "$resolved" ]] || continue
  fi

  case "$seen" in
    *":$resolved:"*) continue ;;
  esac
  seen="$seen$resolved:"

  if "$resolved" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' \
    >/dev/null 2>&1; then
    if [[ "$print_only" == true ]]; then
      printf '%s\n' "$resolved"
    else
      export PYTHON="$resolved"
      exec "$resolved" "$@"
    fi
    exit 0
  fi
done

cat >&2 <<'EOF'
rscrypto scripts require Python 3.11 or newer.

Install a current Python or set PYTHON=/path/to/python3.11-or-newer.
EOF
exit 1
