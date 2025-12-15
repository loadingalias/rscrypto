#!/usr/bin/env bash
set -euo pipefail

CRATE="${1:-}"

echo "Running Criterion Benchmarks..."

# Suggest native CPU for maximum performance
if [[ -z "${RUSTFLAGS:-}" ]] || [[ ! "$RUSTFLAGS" =~ "target-cpu" ]]; then
  echo "Tip: For maximum performance, run with:"
  echo "  RUSTFLAGS='-C target-cpu=native' just bench"
  echo ""
fi

if [ -n "$CRATE" ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Running benchmarks for: $CRATE"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  cargo bench -p "$CRATE"
else
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Running all benchmarks"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  cargo bench --workspace
fi

echo ""
echo "Results: target/criterion/"
