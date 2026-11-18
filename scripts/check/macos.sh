#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export RUSTUP_TOOLCHAIN
RUSTUP_TOOLCHAIN=$(scripts/lib/toolchain.sh)
# Full qualification includes doctests even if an interactive test session skipped them.
unset RSCRYPTO_SKIP_DOCTESTS
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
  echo 'macOS validation requires the local Apple Silicon Mac.' >&2
  exit 1
}
[[ "$(scripts/lib/toolchain.sh --print-host)" == aarch64-apple-darwin ]] || {
  echo 'macOS validation requires an aarch64-apple-darwin Rust toolchain.' >&2
  exit 1
}
scripts/check/check.sh native
just test --all --release
just test --all --release --portable
just test-evidence
just test-rsa-macos-asm
