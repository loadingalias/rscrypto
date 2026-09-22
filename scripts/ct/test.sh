#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$("$ROOT/scripts/lib/python.sh" --print)"
cd "$ROOT"
"$PYTHON" "$ROOT/scripts/ct/asm_heuristics_test.py"
"$PYTHON" "$ROOT/scripts/ct/dudect_report_test.py"
"$PYTHON" "$ROOT/scripts/ct/evidence_validation_test.py"
"$PYTHON" "$ROOT/scripts/ct/dudect_pipeline_test.py"
"$PYTHON" "$ROOT/scripts/ct/preparation_test.py"
"$PYTHON" "$ROOT/scripts/ct/internal_test.py"
"$PYTHON" "$ROOT/scripts/ct/smoke_test.py"
"$PYTHON" "$ROOT/scripts/ct/zeroization_test.py"
"$PYTHON" "$ROOT/scripts/ct/ci_test.py"
"$PYTHON" "$ROOT/scripts/ct/replay_test.py"
"$PYTHON" scripts/ct/internal.py --target "$(scripts/lib/toolchain.sh --print-host)" -- \
  scripts/lib/toolchain.sh --exec cargo test --locked --manifest-path tools/ct-dudect/Cargo.toml \
  -p rscrypto-ct-dudect -p dudect-bencher --lib --bins
just --justfile "$ROOT/justfile" test-evidence
