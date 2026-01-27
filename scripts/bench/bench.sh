#!/usr/bin/env bash
set -euo pipefail

CRATE="${1:-}"
BENCH="${2:-}"

echo "Running Criterion Benchmarks..."

# Suggest native CPU for maximum performance
if [[ -z "${RUSTFLAGS:-}" ]] || [[ ! "$RUSTFLAGS" =~ "target-cpu" ]]; then
  echo "Tip: For maximum performance, run with:"
  echo "  RUSTFLAGS='-C target-cpu=native' just bench"
  echo ""
fi

if [ -n "$CRATE" ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if [ -n "$BENCH" ]; then
    echo "Running benchmarks for: $CRATE ($BENCH)"
  else
    echo "Running benchmarks for: $CRATE"
  fi
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if [ -n "$BENCH" ]; then
    cargo bench -p "$CRATE" --bench "$BENCH"
  else
    cargo bench -p "$CRATE"
  fi
else
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Running all benchmarks"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  cargo bench --workspace
fi

echo ""
echo "Results: target/criterion/"
