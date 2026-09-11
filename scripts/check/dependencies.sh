#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export RUSTUP_TOOLCHAIN
RUSTUP_TOOLCHAIN=$(scripts/lib/toolchain.sh)
# deny.toml covers the supported target catalog; do not filter to this host.
cargo deny --locked --workspace --all-features check -D warnings all
cargo audit
