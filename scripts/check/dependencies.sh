#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export RUSTUP_TOOLCHAIN
RUSTUP_TOOLCHAIN=$(scripts/lib/toolchain.sh)
# Cargo omits versionless path dev-dependencies from published packages. Allow only the local timing runner.
cargo metadata --locked --all-features --format-version 1 | jq '
  def normalized: gsub("\\\\"; "/");
  ((.workspace_root | normalized) + "/tools/ct-dudect/vendor/dudect-bencher") as $runner
  | .packages[] | select(.source == null) | . as $owner | .dependencies[]
  | select(.req == "*" and (.path != null or ((.source // "") | startswith("git+"))))
  | select($owner.name != "rscrypto" or .kind != "dev" or .name != "dudect-bencher" or ((.path // "") | normalized) != $runner)
  | error("versionless local dependency: " + $owner.name + " -> " + .name)
'
# deny.toml covers the supported target catalog; do not filter to this host.
# Metadata unifies dependency features with dev oracles. Check duplicate versions
# without those dev edges, then retain every other rule on the complete graph.
cargo deny --locked --workspace --all-features --exclude-dev check -D warnings bans
cargo deny --locked --workspace --all-features check -D warnings -A duplicate all
cargo audit
