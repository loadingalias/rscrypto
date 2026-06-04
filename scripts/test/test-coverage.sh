#!/usr/bin/env bash
set -euo pipefail

# Coverage Reporting for rscrypto
#
# Generates Rust source coverage from two deterministic sources:
#   1. Unit/integration/property tests via cargo-nextest.
#   2. Committed fuzz corpora via corpus replay integration tests.
#
# Each cargo-llvm-cov invocation re-establishes its per-workspace
# instrumentation allowlist, so the fuzz crates (which live in their own
# isolated `[workspace]`) are instrumented when their replay tests run.
#
# Usage:
#   ./scripts/test/test-coverage.sh                # Total LCOV: nextest + fuzz replay
#   ./scripts/test/test-coverage.sh --nextest      # Test suite LCOV only
#   ./scripts/test/test-coverage.sh --fuzz         # Fuzz corpus replay LCOV only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
maybe_disable_sccache

COV_DIR="$REPO_ROOT/coverage"
mkdir -p "$COV_DIR"

RUN_NEXTEST=true
RUN_FUZZ=true
case "${1:-}" in
  --nextest)  RUN_FUZZ=false ;;
  --fuzz)     RUN_NEXTEST=false ;;
  -h|--help)
    echo "Usage: $0 [--nextest|--fuzz]"
    echo "  (no args)    Both nextest + fuzz coverage"
    echo "  --nextest    Test suite coverage only"
    echo "  --fuzz       Fuzz corpus coverage only"
    exit 0
    ;;
esac

require_tool() {
  command -v "$1" &>/dev/null || { echo "Error: $1 not found"; exit 1; }
}
require_tool cargo-llvm-cov
[ "$RUN_NEXTEST" = true ] && require_tool cargo-nextest

if [ "$RUN_NEXTEST" = true ] && [ "$RUN_FUZZ" = true ]; then
  LCOV="$COV_DIR/total.lcov"
elif [ "$RUN_NEXTEST" = true ]; then
  LCOV="$COV_DIR/nextest.lcov"
else
  LCOV="$COV_DIR/fuzz.lcov"
fi

rm -f "$COV_DIR"/*.lcov "$COV_DIR/SUMMARY.md"
cargo llvm-cov clean --workspace

if [ "$RUN_NEXTEST" = true ]; then
  echo "━━━ Test Suite Coverage Capture (cargo-nextest) ━━━"
  cargo llvm-cov nextest --no-report --workspace --all-features
fi

if [ "$RUN_FUZZ" = true ]; then
  echo "━━━ Fuzz Corpus Coverage Capture (deterministic replay) ━━━"
  export RSCRYPTO_FUZZ_REPLAY_MISSING=skip

  echo "Full fuzz workspace replay"
  cargo llvm-cov test --no-report \
    --manifest-path "$REPO_ROOT/fuzz/Cargo.toml" \
    --all-features --test corpus_replay -- --nocapture

  echo "Scoped fuzz package replay"
  while IFS= read -r manifest; do
    echo "  -> ${manifest#"$REPO_ROOT"/}"
    cargo llvm-cov test --no-report \
      --manifest-path "$manifest" \
      --all-features --test corpus_replay -- --nocapture
  done < <(find "$REPO_ROOT/fuzz-packages" -mindepth 2 -maxdepth 2 -name Cargo.toml | sort)
fi

echo "━━━ Generating LCOV report ━━━"
cargo llvm-cov report --lcov --output-path "$LCOV"

LF=$(awk -F: '/^LF:/{s+=$2} END{print s+0}' "$LCOV")
LH=$(awk -F: '/^LH:/{s+=$2} END{print s+0}' "$LCOV")
FNF=$(awk -F: '/^FNF:/{s+=$2} END{print s+0}' "$LCOV")
FNH=$(awk -F: '/^FNH:/{s+=$2} END{print s+0}' "$LCOV")
LINE_PCT=$(awk "BEGIN{ if ($LF>0) printf \"%.1f\", 100*$LH/$LF; else print \"0.0\" }")
FN_PCT=$(awk "BEGIN{ if ($FNF>0) printf \"%.1f\", 100*$FNH/$FNF; else print \"0.0\" }")

cat > "$COV_DIR/SUMMARY.md" <<EOF
# Coverage Summary

| Metric    | Hit | Total | Coverage |
|-----------|-----|-------|----------|
| Lines     | ${LH} | ${LF} | ${LINE_PCT}% |
| Functions | ${FNH} | ${FNF} | ${FN_PCT}% |

Generated: $(date -u +"%Y-%m-%d %H:%M UTC")

## Sources
$([ "$RUN_NEXTEST" = true ] && echo "- nextest (workspace, --all-features)")
$([ "$RUN_FUZZ" = true ] && echo "- fuzz corpus replay (full + scoped)")
EOF

echo ""
echo "  Lines:     ${LH}/${LF} (${LINE_PCT}%)"
echo "  Functions: ${FNH}/${FNF} (${FN_PCT}%)"
echo "  Report:    $LCOV"
echo "  Summary:   $COV_DIR/SUMMARY.md"
