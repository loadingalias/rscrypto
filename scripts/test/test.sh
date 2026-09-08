#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/rail-plan.sh
source "$SCRIPT_DIR/../lib/rail-plan.sh"

usage() {
  cat <<USAGE
Usage: $0 [--all] [--native | --portable] [--] [NEXTEST_ARGS...]

Repository options precede runner arguments. After --, arguments go unchanged
into cargo nextest run. Any runner arguments select explicit work and skip
doctests. With no runner arguments, Cargo Rail selects tests and doctests.
Example: $0 --portable -- --release --lib -- --skip slow_test
USAGE
}

force_all=false
dispatch_profile=native
selected_dispatch=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) force_all=true ;;
    --native | --portable)
      [[ -z "$selected_dispatch" ]] || { usage >&2; exit 2; }
      selected_dispatch=${1#--}
      dispatch_profile=$selected_dispatch
      ;;
    -h | --help) usage; exit 0 ;;
    --) shift; break ;;
    *) break ;;
  esac
  shift
done
focused=false
if [[ $# -gt 0 ]]; then
  focused=true
  force_all=true
fi

if [[ "$dispatch_profile" == portable ]]; then
  feature_args=(--all-features)
  echo "Dispatch profile: portable (portable-only enabled; accelerated dispatch disabled)"
else
  PYTHON="$("$SCRIPT_DIR/../lib/python.sh" --print)"
  native_features=$("$PYTHON" - "$SCRIPT_DIR/../../Cargo.toml" <<'PYTHON'
import sys, tomllib
with open(sys.argv[1], 'rb') as source:
    features = tomllib.load(source)['features']
selected = set(features) - {'portable-only'}
if any('portable-only' in features[name] for name in selected):
    raise SystemExit('native test features indirectly enable portable-only; fix the feature graph')
print(','.join(sorted(selected)))
PYTHON
  )
  feature_args=(--no-default-features --features "$native_features")
  echo "Dispatch profile: native (all crate features except portable-only; runtime capability detection enabled)"
fi

export RUSTUP_TOOLCHAIN
RUSTUP_TOOLCHAIN=$("$SCRIPT_DIR/../lib/toolchain.sh" --host)

echo "Running tests..."

if ! command -v cargo-nextest >/dev/null 2>&1; then
  echo "cargo-nextest is required; install the repository-pinned tooling" >&2
  exit 127
fi
if [[ -n "${RSCRYPTO_TEST_THREADS:-}" ]]; then
  export NEXTEST_TEST_THREADS="$RSCRYPTO_TEST_THREADS"
  echo "Test threads: $NEXTEST_TEST_THREADS (Nextest CLI options take precedence)"
fi

skip_doctests=false
case "${RSCRYPTO_SKIP_DOCTESTS:-}" in
  1 | true | TRUE | yes | YES)
    skip_doctests=true
    echo "Doctests disabled by RSCRYPTO_SKIP_DOCTESTS"
    ;;
esac

if [[ "$focused" == true ]]; then
  skip_doctests=true
  echo "Doctests disabled for explicit Nextest arguments"
fi

scope_status=0
select_cargo_scope cargo.test "$force_all" || scope_status=$?
if [[ "$scope_status" -gt 1 ]]; then
  exit "$scope_status"
fi

if [[ "$scope_status" -eq 0 ]]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Testing $SCOPE_DESC"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  cargo nextest run --locked "${CARGO_ARGS[@]:+${CARGO_ARGS[@]}}" "${feature_args[@]}" \
    --config-file .config/nextest.toml "$@"
else
  echo "No unit or integration test targets selected by Cargo Rail"
fi

if [[ "$skip_doctests" == true ]]; then
  echo "Doctests skipped"
  exit 0
fi

scope_status=0
select_cargo_scope cargo.doctest "$force_all" || scope_status=$?
if [[ "$scope_status" -gt 1 ]]; then
  exit "$scope_status"
fi
if [[ "$scope_status" -eq 1 ]]; then
  echo "No doctest targets selected by Cargo Rail"
  exit 0
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running doctests for $SCOPE_DESC"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cargo test --locked "${CARGO_ARGS[@]:+${CARGO_ARGS[@]}}" --doc "${feature_args[@]}"
