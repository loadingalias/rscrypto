#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# check: local development. Repair (rustfmt and Clippy fixes), then validate this host.
# native: CI. Validate the runner's host without repairs.
# target TRIPLE: CI cross build. Validate one cross-built Linux target as its own host.
# targets: CI. Run Clippy for each catalog target that no other job checks: not this host, not Apple
# (the Mac checks it), and not a cross-built target (its cross build checks it).
usage='usage: scripts/check/check.sh {check|native|targets|target TRIPLE}'
mode=${1:-}
case "$mode" in
  check|native|targets) [[ $# -eq 1 ]] || { echo "$usage" >&2; exit 2; } ;;
  target) [[ $# -eq 2 ]] || { echo "$usage" >&2; exit 2; }
          scripts/lib/python.sh scripts/lib/cross_build.py "$2" >/dev/null ;;
  *) echo "$usage" >&2; exit 2 ;;
esac

host=$(scripts/lib/toolchain.sh --print-host)
if [[ "$mode" == target ]]; then
  host=$2
  # Cargo keeps build scripts and proc macros on the build host; the checked product
  # targets and independent workspaces use the target cfg.
  export CARGO_BUILD_TARGET="$host"
fi
[[ -n "$host" ]] || { echo 'cannot determine Rust host' >&2; exit 1; }
# rust-toolchain.toml owns the one compiler for every checked target.
channel=$(scripts/lib/toolchain.sh)
export RUSTUP_TOOLCHAIN="$channel"
python=$(scripts/lib/python.sh --print)
"$python" scripts/test/fixtures.py
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
catalog=$(jq -er '.targets | if length == (unique | length) and
  any(.[]; startswith("x86_64-")) and any(.[]; startswith("aarch64-"))
  then .[] else error("invalid target catalog") end' .config/target-matrix.json)
targets=("$host")
if [[ "$mode" == targets ]]; then
  targets=()
  cross=$("$python" -c 'import sys; sys.path.insert(0, "scripts/lib"); from cross_build import TARGETS; print(*TARGETS)')
  while IFS= read -r target; do
    [[ "$target" == "$host" || "$target" == *-apple-darwin || " $cross " == *" $target "* ]] || targets+=("$target")
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
        features=full,rand_core,serde,serde-secrets,websocket-sha1
        ;;
      wasm32-wasip1)
        features=std,full,diag,rand_core,serde,serde-secrets,websocket-sha1,getrandom
        ;;
    esac
    args=(-p rscrypto --locked --target "$target" "${scope[@]}" --no-default-features)
    echo "Clippy ($mode): $target / $channel / release native"
    cargo "+$channel" clippy "${args[@]}" --release --features "$features" "${repair[@]:+${repair[@]}}"
    echo "Clippy ($mode): $target / $channel / debug portable"
    cargo "+$channel" clippy "${args[@]}" --features "$features,portable-only" "${repair[@]:+${repair[@]}}"
    case "$target" in
      *-none*|wasm32-unknown-unknown) ;;
      *)
        provider_args=(--target "$target")
        [[ "$target" != "$host" ]] || provider_args+=(--all-targets)
        [[ "$mode" != fix ]] || provider_args+=(--fix)
        "$python" scripts/check/provider.py lint "${provider_args[@]}"
        ;;
    esac
  done

  if [[ "$mode" == fix ]]; then
    cargo "+$channel" fmt --all
  fi
}
if [[ "$mode" == check ]]; then
  run_checks fix
fi
run_checks verify
[[ "$mode" != targets ]] || exit 0

"$python" scripts/asm/p256.py check
"$python" scripts/asm/p384.py check
"$python" scripts/asm/rsa_pair.py check
"$python" scripts/asm/provenance.py check
scripts/check/lint-independent-workspaces.sh
RUSTDOCFLAGS="${RUSTDOCFLAGS:+$RUSTDOCFLAGS }-D warnings" cargo doc --workspace --no-deps --all-features --locked
