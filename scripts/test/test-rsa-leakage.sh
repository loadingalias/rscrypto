#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running RSA first-order timing leakage gate..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

maybe_disable_sccache

export RSCRYPTO_RSA_LEAKAGE_SAMPLES="${RSCRYPTO_RSA_LEAKAGE_SAMPLES:-2000}"
export RSCRYPTO_RSA_LEAKAGE_T_THRESHOLD="${RSCRYPTO_RSA_LEAKAGE_T_THRESHOLD:-8.0}"
export RSCRYPTO_RSA_LEAKAGE_WARMUP="${RSCRYPTO_RSA_LEAKAGE_WARMUP:-64}"

echo "Samples per case: $RSCRYPTO_RSA_LEAKAGE_SAMPLES"
echo "Welch t threshold: $RSCRYPTO_RSA_LEAKAGE_T_THRESHOLD"
echo "Warmup iterations: $RSCRYPTO_RSA_LEAKAGE_WARMUP"
echo ""

cargo test --release --test rsa_leakage --features rsa,diag,getrandom \
  rsa_private_operations_do_not_show_first_order_timing_leakage \
  -- --ignored --nocapture

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "RSA leakage gate passed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
