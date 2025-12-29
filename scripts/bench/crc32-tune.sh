#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CRC32 Tuning (Fast, Binary-based)
#
# Usage:
#   scripts/bench/crc32-tune.sh          # full (may take longer)
#   scripts/bench/crc32-tune.sh --quick  # faster, noisier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

export RUSTC_WRAPPER=${RUSTC_WRAPPER:-}

echo "Running crc32-tune (binary)..."
echo "Tip: For maximum fidelity, use native CPU flags:"
echo "  RUSTFLAGS='-C target-cpu=native' scripts/bench/crc32-tune.sh --quick"
echo ""

RUSTC_WRAPPER= cargo run -p checksum --release --bin crc32-tune -- "$@"

