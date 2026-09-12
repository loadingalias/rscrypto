#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
  echo 'macOS validation requires the local Apple Silicon Mac.' >&2
  exit 1
}
[[ "$(scripts/lib/toolchain.sh --print-host)" == aarch64-apple-darwin ]] || {
  echo 'macOS validation requires an aarch64-apple-darwin Rust toolchain.' >&2
  exit 1
}
just ci-check
just test --all --release
just test --all --release --portable
just test-rsa-macos-asm
