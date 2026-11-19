#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$("$ROOT/scripts/lib/python.sh" --print)"
cd "$ROOT"
for test in scripts/ct/*_test.py; do "$PYTHON" "$test"; done
"$PYTHON" scripts/ct/internal.py --target "$(scripts/lib/toolchain.sh --print-host)" -- \
  scripts/lib/toolchain.sh --exec cargo test --locked --manifest-path tools/ct-dudect/Cargo.toml \
  -p rscrypto-ct-dudect -p dudect-bencher --lib --bins --tests
just --justfile "$ROOT/justfile" test-evidence
