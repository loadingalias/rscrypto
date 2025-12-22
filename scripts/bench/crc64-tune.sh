#!/usr/bin/env bash
# CRC64 Tuning Discovery Script
#
# Runs a COMPLETE benchmark matrix to discover optimal settings for ALL:
# - Kernels (portable, pclmul, vpclmul, pmull, pmull-eor3, sve2-pmull)
# - Stream counts (supported values only; avoids "requested != effective" skew)
#   - x86_64: 1 / 2 / 4 / 7-way folding
#   - aarch64: 1 / 2 / 3-way folding
# - Buffer sizes (64B to 1MB)
#
# This tests EVERY combination to find the true optimal configuration.
#
# Usage:
#   scripts/bench/crc64-tune.sh           # Full benchmark (slow, accurate)
#   scripts/bench/crc64-tune.sh --quick   # Quick discovery (fast, noisy)
#   cargo run -p checksum --release --bin crc64-tune -- --quick   # Fast tuner (recommended exports)
#
# Environment:
#   RUSTFLAGS='-C target-cpu=native' scripts/bench/crc64-tune.sh
#
# Output:
#   tune-results/summary.tsv    - All results
#   tune-results/analysis.json  - Optimal settings
set -euo pipefail

MODE="${1:-}"

CRITERION_ARGS=(--noplot)
FILTER=()

CRC64_TUNE_FAILED=0

maybe_disable_sccache() {
  # Some environments export RUSTC_WRAPPER=sccache but do not allow sccache to
  # execute compilers or create its state. Detect that case early and fall back
  # to direct rustc to avoid silently producing stale/partial results.
  if [[ "${RUSTC_WRAPPER:-}" == "sccache" ]]; then
    if ! sccache rustc -vV >/dev/null 2>&1; then
      echo "⚠️  WARNING: sccache is configured but not usable; disabling RUSTC_WRAPPER for this run."
      export RUSTC_WRAPPER=
    fi
  fi
}

clear_old_crc64_results() {
  # Avoid mixing results from previous runs. Criterion output is append-only
  # across different group names; we clear only the crc64 subtree.
  rm -rf target/criterion/crc64 2>/dev/null || true
}

case "$MODE" in
  --quick)
    # Fast, noisy-but-useful for crossover discovery
    CRITERION_ARGS+=(--warm-up-time 0.1 --measurement-time 0.3 --sample-size 25)
    FILTER=("oneshot/")
    ;;
  "")
    # Full benchmark (default) - more accurate
    CRITERION_ARGS+=(--warm-up-time 1 --measurement-time 3 --sample-size 50)
    FILTER=("oneshot/")
    ;;
  *)
    echo "Usage: $0 [--quick]"
    exit 1
    ;;
esac

# Make benchmark runs reproducible and avoid stale results.
maybe_disable_sccache
clear_old_crc64_results

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Platform Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

detect_platform() {
  local arch
  arch=$(uname -m)
  case "$arch" in
    x86_64|amd64) echo "x86_64" ;;
    aarch64|arm64) echo "aarch64" ;;
    s390x) echo "s390x" ;;
    ppc64le|ppc64) echo "ppc64" ;;
    riscv64) echo "riscv64" ;;
    *) echo "unknown" ;;
  esac
}

detect_cpu_features() {
  local platform="$1"
  local features=""

  case "$platform" in
    x86_64)
      # Linux
      if [[ -f /proc/cpuinfo ]]; then
        grep -q "pclmulqdq" /proc/cpuinfo 2>/dev/null && features="$features pclmul"
        grep -q "avx512f" /proc/cpuinfo 2>/dev/null && features="$features vpclmul vpternlogd"
        grep -q "avx512vl" /proc/cpuinfo 2>/dev/null && features="$features avx512vl"
      fi
      # macOS
      if command -v sysctl &>/dev/null; then
        sysctl -n machdep.cpu.features 2>/dev/null | grep -qi "PCLMULQDQ" && features="$features pclmul"
        sysctl -n machdep.cpu.leaf7_features 2>/dev/null | grep -qi "AVX512" && features="$features vpclmul"
      fi
      ;;
    aarch64)
      # Linux
      if [[ -f /proc/cpuinfo ]]; then
        grep -q "aes" /proc/cpuinfo 2>/dev/null && features="$features pmull"
        grep -q "sha3" /proc/cpuinfo 2>/dev/null && features="$features pmull-eor3"
        grep -q "sve2" /proc/cpuinfo 2>/dev/null && features="$features sve2-pmull"
      fi
      # macOS (Apple Silicon)
      if command -v sysctl &>/dev/null; then
        sysctl -n hw.optional.arm.FEAT_AES 2>/dev/null | grep -q "1" && features="$features pmull"
        sysctl -n hw.optional.arm.FEAT_SHA3 2>/dev/null | grep -q "1" && features="$features pmull-eor3"
      fi
      ;;
    s390x)
      features="$features vx"
      ;;
    ppc64)
      features="$features altivec"
      grep -q "vsx" /proc/cpuinfo 2>/dev/null && features="$features vsx"
      ;;
  esac

  echo "$features"
}

detect_cpu_name() {
  local name="unknown"

  if [[ -f /proc/cpuinfo ]]; then
    name=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "")
    if [[ -z "$name" ]]; then
      # ARM: implementer + part
      local impl part
      impl=$(grep -m1 "CPU implementer" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "")
      part=$(grep -m1 "CPU part" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "")
      [[ -n "$impl" && -n "$part" ]] && name="ARM impl=$impl part=$part"
    fi
  fi

  # macOS
  if [[ -z "$name" || "$name" == "unknown" ]]; then
    name=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")
  fi

  # Fallback
  [[ -z "$name" ]] && name="$(uname -m) CPU"
  echo "$name"
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Benchmark Runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

run() {
  local label="$1"
  shift
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  $label"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if ! env "$@" cargo bench -p checksum --bench crc64 -- "${CRITERION_ARGS[@]}" "${FILTER[@]}" 2>&1; then
    CRC64_TUNE_FAILED=1
    echo "⚠️  WARNING: benchmark run failed: $label" >&2
  fi
}

run_if_supported() {
  local label="$1"
  local required_feature="$2"
  shift 2

  # If we can't detect features (e.g., sandboxed macOS without sysctl),
  # run anyway and let the library clamp unsafe force modes to safe fallbacks.
  if [[ -z "${DETECTED_FEATURES// }" ]]; then
    run "$label" "$@"
  elif [[ " $DETECTED_FEATURES " == *" $required_feature "* ]]; then
    run "$label" "$@"
  else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $label [SKIPPED - $required_feature not detected]"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  fi
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PLATFORM=$(detect_platform)
DETECTED_FEATURES=$(detect_cpu_features "$PLATFORM")
CPU_NAME=$(detect_cpu_name)

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║              CRC64 COMPLETE TUNING DISCOVERY MATRIX                      ║"
echo "╠══════════════════════════════════════════════════════════════════════════╣"
echo "║ Platform:  $PLATFORM"
echo "║ CPU:       $CPU_NAME"
echo "║ Features: ${DETECTED_FEATURES:-none detected}"
echo "║ Mode:      ${MODE:-full}"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check for native CPU optimization
if [[ -z "${RUSTFLAGS:-}" ]] || [[ ! "$RUSTFLAGS" =~ "target-cpu" ]]; then
  echo "⚠️  WARNING: For accurate results, run with:"
  echo "   RUSTFLAGS='-C target-cpu=native' $0 ${MODE:-}"
  echo ""
fi

# Create results directory
mkdir -p tune-results

# Save platform info
cat > tune-results/platform.json << EOF
{
  "platform": "$PLATFORM",
  "cpu_name": "$CPU_NAME",
  "features": "$DETECTED_FEATURES",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "rustflags": "${RUSTFLAGS:-}",
  "mode": "${MODE:-full}"
}
EOF

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Baseline: Auto Selection (what the user gets by default)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "══════════════════════════════════════════════════════════════════════════"
echo "BASELINE: Auto Selection (default behavior)"
echo "══════════════════════════════════════════════════════════════════════════"

run "auto (default)" \
  RSCRYPTO_CRC64_FORCE= \
  RSCRYPTO_CRC64_STREAMS=

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Portable Baseline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "══════════════════════════════════════════════════════════════════════════"
echo "PORTABLE: Table-based slice-by-16 (no SIMD)"
echo "══════════════════════════════════════════════════════════════════════════"

run "portable/slice16" \
  RSCRYPTO_CRC64_FORCE=portable \
  RSCRYPTO_CRC64_STREAMS=1

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# x86_64 Kernels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [[ "$PLATFORM" == "x86_64" ]]; then

  echo ""
  echo "══════════════════════════════════════════════════════════════════════════"
  echo "x86_64: PCLMULQDQ (128-bit carryless multiply)"
  echo "  Available on: Intel Westmere+, AMD Bulldozer+"
  echo "══════════════════════════════════════════════════════════════════════════"

  # Only test stream counts the implementation can actually execute.
  # (If you pass 3/5/6/8 today, the selector clamps to 2/4/7 and the results
  # are mislabeled and harder to analyze.)
  for streams in 1 2 4 7; do
    run_if_supported "pclmul ${streams}-way" "pclmul" \
      RSCRYPTO_CRC64_FORCE=pclmul \
      RSCRYPTO_CRC64_STREAMS="$streams"
  done

  echo ""
  echo "══════════════════════════════════════════════════════════════════════════"
  echo "x86_64: VPCLMULQDQ (AVX-512 carryless multiply)"
  echo "  Available on: Intel Ice Lake+, AMD Zen4+"
  echo "  Uses VPTERNLOGD for 3-way XOR when available"
  echo "══════════════════════════════════════════════════════════════════════════"

  for streams in 1 2 4 7; do
    run_if_supported "vpclmul ${streams}-way" "vpclmul" \
      RSCRYPTO_CRC64_FORCE=vpclmul \
      RSCRYPTO_CRC64_STREAMS="$streams"
  done

fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# aarch64 Kernels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [[ "$PLATFORM" == "aarch64" ]]; then

  echo ""
  echo "══════════════════════════════════════════════════════════════════════════"
  echo "aarch64: PMULL (NEON polynomial multiply)"
  echo "  Available on: ARMv8 with Crypto extensions"
  echo "  CPUs: Apple A7+, Cortex-A53+, Graviton, Neoverse"
  echo "══════════════════════════════════════════════════════════════════════════"

  # Only test stream counts the implementation can actually execute.
  for streams in 1 2 3; do
    run_if_supported "pmull ${streams}-way" "pmull" \
      RSCRYPTO_CRC64_FORCE=pmull \
      RSCRYPTO_CRC64_STREAMS="$streams"
  done

  echo ""
  echo "══════════════════════════════════════════════════════════════════════════"
  echo "aarch64: PMULL+EOR3 (SHA3 3-way XOR optimization)"
  echo "  Available on: ARMv8.2-SHA3 (Apple M1+, Graviton3+, Neoverse V1+)"
  echo "  Uses EOR3 instruction for single-cycle 3-way XOR in folding"
  echo "══════════════════════════════════════════════════════════════════════════"

  for streams in 1 2 3; do
    run_if_supported "pmull-eor3 ${streams}-way" "pmull-eor3" \
      RSCRYPTO_CRC64_FORCE=pmull-eor3 \
      RSCRYPTO_CRC64_STREAMS="$streams"
  done

  echo ""
  echo "══════════════════════════════════════════════════════════════════════════"
  echo "aarch64: SVE2-PMULL (Scalable Vector Extension 2)"
  echo "  Available on: Neoverse V1/N2/V2, Graviton3+, Cortex-X2+"
  echo "  Variable vector width (128-2048 bits)"
  echo "══════════════════════════════════════════════════════════════════════════"

  for streams in 1 2 3; do
    run_if_supported "sve2-pmull ${streams}-way" "sve2-pmull" \
      RSCRYPTO_CRC64_FORCE=sve2-pmull \
      RSCRYPTO_CRC64_STREAMS="$streams"
  done

fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Other Architectures (if we ever add them)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [[ "$PLATFORM" == "s390x" ]]; then
  echo ""
  echo "══════════════════════════════════════════════════════════════════════════"
  echo "s390x: Vector Facility"
  echo "══════════════════════════════════════════════════════════════════════════"
  # TODO: Add s390x kernels when implemented
  echo "  (s390x CRC64 kernels not yet implemented)"
fi

if [[ "$PLATFORM" == "ppc64" ]]; then
  echo ""
  echo "══════════════════════════════════════════════════════════════════════════"
  echo "POWER: AltiVec/VSX vector extensions"
  echo "══════════════════════════════════════════════════════════════════════════"
  # TODO: Add POWER kernels when implemented
  echo "  (POWER CRC64 kernels not yet implemented)"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generate Summary and Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                          Benchmark Complete                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Generate summary from criterion results
echo "Generating summary..."
if python3 scripts/bench/criterion-summary.py \
  --group-prefix 'crc64/' \
  --only oneshot \
  > tune-results/summary.tsv 2>/dev/null; then

  RESULT_COUNT=$(wc -l < tune-results/summary.tsv)
  echo "Summary saved: tune-results/summary.tsv ($RESULT_COUNT results)"
else
  echo "Warning: Summary generation failed"
  CRC64_TUNE_FAILED=1
fi

# Run analysis
echo ""
echo "Running analysis..."
if python3 scripts/bench/crc64-analyze.py \
  --summary tune-results/summary.tsv \
  --platform-file tune-results/platform.json \
  --output tune-results/analysis.json; then
  echo ""
  echo "Analysis saved: tune-results/analysis.json"
else
  echo "Warning: Analysis failed"
  CRC64_TUNE_FAILED=1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Files created:"
echo "  tune-results/platform.json  - CPU/platform info"
echo "  tune-results/summary.tsv    - All benchmark results"
echo "  tune-results/analysis.json  - Optimal settings analysis"
echo ""
echo "To view analysis:"
echo "  python3 scripts/bench/crc64-analyze.py --summary tune-results/summary.tsv"
echo ""
echo "To compare specific sizes:"
echo "  cat tune-results/summary.tsv | grep '65536' | sort -t$'\\t' -k5 -rn | head"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$CRC64_TUNE_FAILED" -ne 0 ]]; then
  if [[ "${RSCRYPTO_CRC64_TUNE_IGNORE_FAILURES:-}" == "1" ]]; then
    echo "⚠️  WARNING: one or more benchmark steps failed; ignoring due to RSCRYPTO_CRC64_TUNE_IGNORE_FAILURES=1" >&2
    exit 0
  fi
  echo "ERROR: one or more benchmark steps failed; results may be incomplete." >&2
  echo "Set RSCRYPTO_CRC64_TUNE_IGNORE_FAILURES=1 to keep partial outputs." >&2
  exit 1
fi
