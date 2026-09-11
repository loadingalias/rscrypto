#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$("$ROOT/scripts/lib/python.sh" --print)"
"$PYTHON" "$ROOT/scripts/ct/asm_heuristics_test.py"
"$PYTHON" "$ROOT/scripts/ct/dudect_report_test.py"
"$PYTHON" "$ROOT/scripts/ct/evidence_validation_test.py"
"$PYTHON" "$ROOT/scripts/ct/dudect_pipeline_test.py"
"$PYTHON" "$ROOT/scripts/ct/preparation_test.py"
"$PYTHON" "$ROOT/scripts/ct/smoke_test.py"
"$PYTHON" "$ROOT/scripts/ct/zeroization_test.py"
"$PYTHON" "$ROOT/scripts/ct/ci_test.py"
just --justfile "$ROOT/justfile" test-harnesses
