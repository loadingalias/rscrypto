#!/usr/bin/env bash
set -euo pipefail
unset BASH_ENV

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

fail() {
  echo "remote cache recipe test failure: $*" >&2
  exit 1
}

mkdir -p "$TMP_ROOT/bin"
cat >"$TMP_ROOT/bin/cargo" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$MOCK_CARGO_LOG"
if [[ ${MOCK_FAIL_APPLY:-0} == 1 && "$*" == "rail cache setup --remote "* ]]; then
  exit 40
fi
if [[ ${MOCK_FAIL_PREVIEW:-0} == 1 && "$*" == "rail cache setup --check "* ]]; then
  exit 41
fi
SH
chmod +x "$TMP_ROOT/bin/cargo"

export PATH="$TMP_ROOT/bin:$PATH"
export MOCK_CARGO_LOG="$TMP_ROOT/cargo.log"
export CARGO_RAIL_CACHE_REMOTE='r2://rscrypto-cache.example/rscrypto/shared'
export CARGO_RAIL_CACHE_MODE=read-write

just --justfile "$REPO_ROOT/justfile" rail-cache-setup --max-size 10GiB
cat >"$TMP_ROOT/expected-setup.log" <<'EOF'
rail cache setup --check --remote r2://rscrypto-cache.example/rscrypto/shared --remote-mode read-write --root-portability remap --max-size 10GiB
rail cache setup --remote r2://rscrypto-cache.example/rscrypto/shared --remote-mode read-write --root-portability remap --max-size 10GiB
rail cache probe --json
EOF
cmp "$TMP_ROOT/expected-setup.log" "$MOCK_CARGO_LOG" \
  || fail "setup recipe did not preserve the canonical preview/apply/probe transaction"

: >"$MOCK_CARGO_LOG"
just --justfile "$REPO_ROOT/justfile" cache-status
[[ $(<"$MOCK_CARGO_LOG") == 'rail cache status --scope local --format json' ]] \
  || fail "status recipe did not request local JSON telemetry"

: >"$MOCK_CARGO_LOG"
if MOCK_FAIL_PREVIEW=1 just --justfile "$REPO_ROOT/justfile" rail-cache-setup --max-size 10GiB \
  >/dev/null 2>&1; then
  fail "setup recipe continued after preview failed"
fi
[[ $(wc -l <"$MOCK_CARGO_LOG" | tr -d ' ') == 1 ]] \
  || fail "setup recipe continued after preview failed"

: >"$MOCK_CARGO_LOG"
if MOCK_FAIL_APPLY=1 just --justfile "$REPO_ROOT/justfile" rail-cache-setup --max-size 10GiB \
  >/dev/null 2>&1; then
  fail "setup recipe accepted a failed installation"
fi
[[ $(wc -l <"$MOCK_CARGO_LOG" | tr -d ' ') == 2 ]] \
  || fail "setup recipe probed a policy that failed installation"

echo "remote cache recipe tests passed"
