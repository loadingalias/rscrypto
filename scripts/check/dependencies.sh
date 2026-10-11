#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export RUSTUP_TOOLCHAIN
RUSTUP_TOOLCHAIN=$(scripts/lib/toolchain.sh)
# deny.toml covers the supported target catalog; do not filter to this host.
# Metadata unifies dependency features with dev oracles. Check duplicate versions
# without those dev edges, then retain every other rule on the complete graph.
cargo deny --locked --workspace --all-features --exclude-dev check -D warnings bans
cargo deny --locked --workspace --all-features check -D warnings -A duplicate all
cargo audit
