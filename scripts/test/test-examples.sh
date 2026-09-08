#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
RUSTUP_TOOLCHAIN=$("$SCRIPT_DIR/../lib/toolchain.sh" --host)
export RUSTUP_TOOLCHAIN

examples=$(cargo metadata --locked --no-deps --format-version 1 | jq -cer '
  [.packages[] | select(.name == "rscrypto") | .targets[]
   | select((.kind | index("example")) and (.crate_types | index("bin")))]
  | sort_by(.name)
  | if length > 0 then . else error("no runnable examples found") end
')
total=$(jq 'length' <<<"$examples")
rows=$(jq -r '.[] | [.name, ((."required-features" // []) | join(","))] | @tsv' <<<"$examples")
index=0
while IFS=$'\t' read -r name features; do
  index=$((index + 1))
  printf '[%d/%d] %s\n' "$index" "$total" "$name"
  args=(run --locked --quiet --no-default-features --example "$name")
  [[ -z "$features" ]] || args+=(--features "$features")
  cargo "${args[@]}"
done <<<"$rows"
echo "All $total examples passed"
