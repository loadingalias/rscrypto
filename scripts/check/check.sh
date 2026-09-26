#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

[[ $# -ge 1 && $# -le 2 ]] || { echo 'usage: scripts/check/check.sh {check|fix|local|native|target TRIPLE}' >&2; exit 2; }
mode=$1
case "$mode" in
  check|fix|local|native) [[ $# -eq 1 ]] || exit 2 ;;
  target) [[ $# -eq 2 ]] || exit 2; python3 scripts/lib/cross_build.py "$2" >/dev/null ;;
  *) echo 'usage: scripts/check/check.sh {check|fix|local|native}' >&2; exit 2 ;;
esac

host=$(scripts/lib/toolchain.sh --print-host)
if [[ "$mode" == target ]]; then
  host=$2
  # Cargo keeps build scripts/proc macros on the build host; all checked product
  # targets and independent workspaces use the same target cfg as native CI.
  export CARGO_BUILD_TARGET="$host"
fi
[[ -n "$host" ]] || { echo 'cannot determine Rust host' >&2; exit 1; }
# rust-toolchain.toml owns the one compiler for every checked target.
channel=$(scripts/lib/toolchain.sh)
export RUSTUP_TOOLCHAIN="$channel"
python=$(scripts/lib/python.sh --print)
# Resolve feature roots from Cargo, rejecting aliases that enable portable-only.
native_features=$("$python" - <<'PY'
import tomllib
with open('Cargo.toml', 'rb') as source:
    features = tomllib.load(source)['features']
selected = set(features) - {'portable-only'}
if any('portable-only' in features[name] for name in selected):
    raise SystemExit('native check features indirectly enable portable-only')
print(','.join(sorted(selected)))
PY
)
targets=("$host")
if [[ "$mode" != native && "$mode" != target ]]; then
  catalog=$(jq -er '.targets | if length > 0 and length == (unique | length) then .[] else error("invalid target catalog") end' .config/target-matrix.json)
  while IFS= read -r target; do
    [[ "$target" == "$host" ]] || targets+=("$target")
  done <<<"$catalog"
fi

# No implicit installation or skipped lanes: report prerequisites before editing.
missing=false
installed=$(rustup target list --toolchain "$channel" --installed)
components=$(rustup component list --toolchain "$channel" --installed)
for target in "${targets[@]}"; do
  if ! grep -qx "$target" <<<"$installed"; then
    echo "missing check prerequisite: rustup target add --toolchain $channel $target" >&2
    missing=true
  fi
done
if ! grep -q '^clippy-' <<<"$components"; then
  echo "missing check prerequisite: rustup component add --toolchain $channel clippy" >&2
  missing=true
fi
[[ "$missing" == false ]] || exit 1

run_checks() {
  local mode=$1
  repair=()
  if [[ "$mode" == fix ]]; then
    cargo "+$channel" fmt --all
    repair=(--fix --allow-dirty --allow-staged)
  else
    cargo "+$channel" fmt --all -- --check
  fi

  for target in "${targets[@]}"; do
    scope=(--lib)
    [[ "$target" != "$host" ]] || scope=(--all-targets)
    features=$native_features
    case "$target" in
      *-none*|wasm32-unknown-unknown)
        # No OS entropy, threads, or std; full retains alloc-backed primitives.
        features=full,serde,serde-secrets,websocket-sha1
        ;;
      wasm32-wasip1)
        features=std,full,diag,serde,serde-secrets,websocket-sha1,getrandom
        ;;
    esac
    args=(--workspace --locked --target "$target" "${scope[@]}" --no-default-features)
    echo "Clippy ($mode): $target / $channel / release native"
    cargo "+$channel" clippy "${args[@]}" --release --features "$features" "${repair[@]:+${repair[@]}}"
    echo "Clippy ($mode): $target / $channel / debug portable"
    cargo "+$channel" clippy "${args[@]}" --features "$features,portable-only" "${repair[@]:+${repair[@]}}"
  done

  if [[ "$mode" == fix ]]; then
    cargo "+$channel" fmt --all
  fi
}
if [[ "$mode" == check ]]; then
  run_checks fix
  run_checks local
else
  run_checks "$mode"
fi
[[ "$mode" != fix ]] || exit 0

# Native CI keeps architecture-sensitive compilation and documentation here.
scripts/check/lint-independent-workspaces.sh
if [[ "$mode" != native && "$mode" != target ]]; then scripts/check/dependencies.sh; fi
RUSTDOCFLAGS="${RUSTDOCFLAGS:+$RUSTDOCFLAGS }-D warnings" cargo doc --workspace --no-deps --all-features --locked
