#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"

CRITERION_ARGS=(--noplot)
FILTER=()
if [[ "$MODE" == "--quick" ]]; then
  # Fast, noisy-but-useful for crossover discovery.
  CRITERION_ARGS+=(--warm-up-time 0.1 --measurement-time 0.2 --sample-size 20)
  # Tuning is primarily about crossovers and stream count for full buffers.
  # Restrict to oneshot benches to keep the matrix fast.
  FILTER=("oneshot/")
fi

echo "CRC64 tuning bench matrix"
echo "  Output: target/criterion/"
echo ""

# Suggest native CPU for maximum performance
if [[ -z "${RUSTFLAGS:-}" ]] || [[ ! "$RUSTFLAGS" =~ "target-cpu" ]]; then
  echo "Tip: For maximum performance, run with:"
  echo "  RUSTFLAGS='-C target-cpu=native' scripts/bench/crc64-tune.sh${MODE:+ $MODE}"
  echo ""
fi

run() {
  local label="$1"
  shift
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "$label"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  env "$@" cargo bench -p checksum --bench crc64 -- "${CRITERION_ARGS[@]}" "${FILTER[@]}"
}

run "auto (no overrides)" \
  RSCRYPTO_CRC64_FORCE= \
  RSCRYPTO_CRC64_STREAMS=

run "force=portable" \
  RSCRYPTO_CRC64_FORCE=portable \
  RSCRYPTO_CRC64_STREAMS=

for streams in 1 2 4 7; do
  run "force=pclmul streams=${streams}" \
    RSCRYPTO_CRC64_FORCE=pclmul \
    RSCRYPTO_CRC64_STREAMS="$streams"
done

for streams in 1 2 4 7; do
  run "force=vpclmul streams=${streams}" \
    RSCRYPTO_CRC64_FORCE=vpclmul \
    RSCRYPTO_CRC64_STREAMS="$streams"
done

echo ""
echo "Next:"
echo "  python3 scripts/bench/criterion-summary.py --group-prefix 'crc64/' --only oneshot"
