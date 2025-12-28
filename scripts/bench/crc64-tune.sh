#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CRC64 Tuning (Fast, Binary-based)
#
# The previous Criterion-based tuning discovery bench (`--bench crc64`) was
# intentionally removed to keep CI and local benchmarking focused on `comp`.
#
# This script is now a thin wrapper around the `checksum` crate's `crc64-tune`
# binary, which prints recommended `RSCRYPTO_CRC64_*` settings.
#
# Usage:
#   scripts/bench/crc64-tune.sh          # full (may take longer)
#   scripts/bench/crc64-tune.sh --quick  # faster, noisier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

export RUSTC_WRAPPER=${RUSTC_WRAPPER:-}

echo "Running crc64-tune (binary)..."
echo "Tip: For maximum fidelity, use native CPU flags:"
echo "  RUSTFLAGS='-C target-cpu=native' scripts/bench/crc64-tune.sh --quick"
echo ""

RUSTC_WRAPPER= cargo run -p checksum --release --bin crc64-tune -- "$@"

