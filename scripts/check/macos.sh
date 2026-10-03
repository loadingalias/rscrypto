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
# A clean checkout whose tree passed `just ci-check`, for example in the pre-commit hook, reuses that pass.
if [[ -n "$(git status --porcelain)" ]] || ! scripts/lib/python.sh scripts/check/qualified.py check ci-check HEAD; then
  just ci-check
  [[ -n "$(git status --porcelain)" ]] || scripts/lib/python.sh scripts/check/qualified.py record ci-check HEAD
fi
just test --all --release
just test --all --release --portable
just test-evidence
just test-rsa-macos-asm
