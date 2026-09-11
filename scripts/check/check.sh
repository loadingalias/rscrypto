#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

[[ $# -ge 1 && $# -le 2 ]] || { echo 'usage: scripts/check/check.sh {check|fix|local|native|target TRIPLE}' >&2; exit 2; }
mode=$1
case "$mode" in
  check|fix|local|native) [[ $# -eq 1 ]] || exit 2 ;;
  target) [[ $# -eq 2 && "$2" == riscv64gc-unknown-linux-gnu ]] || exit 2 ;;
  *) echo 'usage: scripts/check/check.sh {check|fix|local|native}' >&2; exit 2 ;;
esac

host=$(scripts/lib/toolchain.sh --print-host)
if [[ "$mode" == target ]]; then
  host=$2
  # Cargo keeps build scripts/proc macros on the build host; all checked product
  # targets and independent workspaces use the same RISC-V cfg as native CI.
  export CARGO_BUILD_TARGET="$host"
fi
[[ -n "$host" ]] || { echo 'cannot determine Rust host' >&2; exit 1; }
stable=$(scripts/lib/toolchain.sh)
export RUSTUP_TOOLCHAIN
RUSTUP_TOOLCHAIN=$(scripts/lib/toolchain.sh --target "$host")
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
inventory_toolchains=()
inventory_targets=()
inventory_components=()
for target in "${targets[@]}"; do
  toolchain=$(scripts/lib/toolchain.sh --target "$target")
  index=0
  while [[ "$index" -lt "${#inventory_toolchains[@]}" && "${inventory_toolchains[$index]}" != "$toolchain" ]]; do
    index=$((index + 1))
  done
  if [[ "$index" -eq "${#inventory_toolchains[@]}" ]]; then
    installed=$(rustup target list --toolchain "$toolchain" --installed)
    components=$(rustup component list --toolchain "$toolchain" --installed)
    inventory_toolchains+=("$toolchain")
    inventory_targets+=("$installed")
    inventory_components+=("$components")
  fi
  installed=${inventory_targets[$index]}
  if ! grep -qx "$target" <<<"$installed"; then
    echo "missing check prerequisite: rustup target add --toolchain $toolchain $target" >&2
    missing=true
  fi
  components=${inventory_components[$index]}
  if ! grep -q '^clippy-' <<<"$components"; then
    echo "missing check prerequisite: rustup component add --toolchain $toolchain clippy" >&2
    missing=true
  fi
done
[[ "$missing" == false ]] || exit 1

run_checks() {
  local mode=$1
  repair=()
  if [[ "$mode" == fix ]]; then
    cargo "+$stable" fmt --all
    repair=(--fix --allow-dirty --allow-staged)
  else
    cargo "+$stable" fmt --all -- --check
  fi

  for target in "${targets[@]}"; do
    toolchain=$(scripts/lib/toolchain.sh --target "$target")
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
    echo "Clippy ($mode): $target / $toolchain / release native"
    cargo "+$toolchain" clippy "${args[@]}" --release --features "$features" "${repair[@]:+${repair[@]}}"
    echo "Clippy ($mode): $target / $toolchain / debug portable"
    cargo "+$toolchain" clippy "${args[@]}" --features "$features,portable-only" "${repair[@]:+${repair[@]}}"
  done

  if [[ "$mode" == fix ]]; then
    cargo "+$stable" fmt --all
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
