#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/rail-plan.sh
source "$SCRIPT_DIR/../lib/rail-plan.sh"

usage() {
  cat <<USAGE
Usage: $0 [--all] [--release] [--native | --portable] [--] [NEXTEST_ARGS...]

Repository options precede runner arguments. After --, arguments go unchanged
into cargo nextest run. Any runner arguments select explicit work and skip
doctests. With no runner arguments, Cargo Rail selects tests and doctests.
Example: $0 --portable -- --release --lib -- --skip slow_test
USAGE
}

profile_args=()
force_all=false
dispatch_profile=native
selected_dispatch=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) force_all=true ;;
    --release) profile_args=(--release) ;;
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

# Cargo feature selection belongs to this wrapper. Allowing forwarded feature
# flags would let a command labelled native enable `portable-only` after the
# dispatch profile was resolved. Explicit package selection is honored: the
# wrapper then adds no --workspace, and native features name only the selected
# packages, because Cargo rejects features of unselected packages.
explicit_selection=()
explicit_packages=false
explicit_excludes=false
explicit_workspace=false
forwarded=("$@")
for ((index = 0; index < ${#forwarded[@]}; index++)); do
  arg=${forwarded[index]}
  [[ "$arg" != -- ]] || break
  case "$arg" in
    --all-features | --features | --features=* | -F | -F?*)
      echo "Cargo feature selection must use the repository dispatch profile, not Nextest arguments: $arg" >&2
      exit 2
      ;;
    -p | --package | --exclude)
      ((index + 1 < ${#forwarded[@]})) || { echo "$arg requires a package name" >&2; exit 2; }
      index=$((index + 1))
      if [[ "$arg" == --exclude ]]; then
        explicit_excludes=true
        explicit_selection+=(--exclude "${forwarded[index]}")
      else
        explicit_packages=true
        explicit_selection+=(--package "${forwarded[index]}")
      fi
      ;;
    -p?*) explicit_packages=true; explicit_selection+=(--package "${arg#-p}") ;;
    --package=*) explicit_packages=true; explicit_selection+=(--package "${arg#--package=}") ;;
    --exclude=*) explicit_excludes=true; explicit_selection+=(--exclude "${arg#--exclude=}") ;;
    --workspace | --all) explicit_workspace=true ;;
  esac
done
if [[ "$explicit_packages" == true && ( "$explicit_excludes" == true || "$explicit_workspace" == true ) ]]; then
  echo "Select packages with --package, or the workspace with --workspace and --exclude, not both" >&2
  exit 2
fi

if [[ "$dispatch_profile" == portable ]]; then
  feature_args=(--all-features)
  echo "Dispatch profile: portable-only (all Cargo features; accelerated dispatch disabled)"
else
  PYTHON="$("$SCRIPT_DIR/../lib/python.sh" --print)"
  native_features=$("$PYTHON" "$SCRIPT_DIR/../lib/workspace_features.py" \
    "${explicit_selection[@]:+${explicit_selection[@]}}")
  feature_args=(--no-default-features --features "$native_features")
  echo "Dispatch profile: production-auto (all feature roots except portable-only; production dispatch enabled)"
fi

export RUSTUP_TOOLCHAIN
RUSTUP_TOOLCHAIN=$("$SCRIPT_DIR/../lib/toolchain.sh")

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
if [[ "$explicit_packages" == true || "$explicit_workspace" == true ]]; then
  CARGO_ARGS=()
  SCOPE_DESC="explicit packages"
elif [[ "$explicit_excludes" == true ]]; then
  # Cargo accepts --exclude only together with --workspace.
  CARGO_ARGS=(--workspace)
  SCOPE_DESC="workspace with explicit exclusions"
fi

if [[ "$scope_status" -eq 0 ]]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Testing $SCOPE_DESC"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  cargo nextest run --locked "${CARGO_ARGS[@]:+${CARGO_ARGS[@]}}" "${feature_args[@]}" "${profile_args[@]:+${profile_args[@]}}" \
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
cargo test --locked "${CARGO_ARGS[@]:+${CARGO_ARGS[@]}}" --doc "${feature_args[@]}" "${profile_args[@]:+${profile_args[@]}}"
