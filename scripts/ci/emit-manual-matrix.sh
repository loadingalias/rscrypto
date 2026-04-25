#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  bench) ;;
  *)
    echo "usage: scripts/ci/emit-manual-matrix.sh <bench>" >&2
    exit 2
    ;;
esac

append_unique() {
  local value="${1:-}"
  local array_name="${2:-}"
  local -a current=()
  local item
  [[ -z "$value" || -z "$array_name" ]] && return 0
  eval "current=(\"\${${array_name}[@]:+\${${array_name}[@]}}\")"
  for item in "${current[@]:+${current[@]}}"; do
    if [[ "$item" == "$value" ]]; then
      return 0
    fi
  done
  eval "$array_name+=(\"\$value\")"
}

normalize_platform() {
  local raw="${1:-}"
  raw="$(echo "$raw" | tr '[:upper:]' '[:lower:]' | xargs)"
  case "$raw" in
    amd-zen4|zen4) echo "amd-zen4" ;;
    intel-spr|spr) echo "intel-spr" ;;
    intel-icl|icl) echo "intel-icl" ;;
    amd-zen5|zen5) echo "amd-zen5" ;;
    graviton3|g3) echo "graviton3" ;;
    graviton4|g4) echo "graviton4" ;;
    ibm-s390x|s390x) echo "ibm-s390x" ;;
    ibm-power10|power10|p10) echo "ibm-power10" ;;
    rise-riscv|riscv|riscv64|th1520|em-rv1) echo "rise-riscv" ;;
    all) echo "all" ;;
    *) echo "" ;;
  esac
}

# Bench matrix rows. Shape matches the inputs to _rust-job.yaml:
#   runner, timeout_minutes, tools_mode, toolchain_components
# Plus bench-specific plumbing (platform, os, display_name, artifact_suffix).
# Caching is always disabled for bench — callers set enable_*_cache=false in bench.yaml.
append_row_for_platform() {
  local platform="${1:-}"
  local runner_uarch="runs-on=${GH_RUN_ID_VAL}/runner="
  case "$platform" in
    amd-zen4)
      ROWS+=("{\"platform\":\"amd-zen4\",\"os\":\"linux\",\"display_name\":\"AMD Zen4\",\"artifact_suffix\":\"amd-zen4\",\"timeout_minutes\":${RUNSON_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}amd-zen4\",\"tools_mode\":\"minimal\",\"toolchain_components\":\"${COMPONENTS_STD}\"}")
      ;;
    intel-spr)
      ROWS+=("{\"platform\":\"intel-spr\",\"os\":\"linux\",\"display_name\":\"Intel Sapphire Rapids\",\"artifact_suffix\":\"intel-spr\",\"timeout_minutes\":${RUNSON_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}intel-spr\",\"tools_mode\":\"minimal\",\"toolchain_components\":\"${COMPONENTS_STD}\"}")
      ;;
    intel-icl)
      ROWS+=("{\"platform\":\"intel-icl\",\"os\":\"linux\",\"display_name\":\"Intel Ice Lake\",\"artifact_suffix\":\"intel-icl\",\"timeout_minutes\":${RUNSON_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}intel-icl\",\"tools_mode\":\"minimal\",\"toolchain_components\":\"${COMPONENTS_STD}\"}")
      ;;
    amd-zen5)
      ROWS+=("{\"platform\":\"amd-zen5\",\"os\":\"linux\",\"display_name\":\"AMD Zen5\",\"artifact_suffix\":\"amd-zen5\",\"timeout_minutes\":${RUNSON_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}amd-zen5\",\"tools_mode\":\"minimal\",\"toolchain_components\":\"${COMPONENTS_STD}\"}")
      ;;
    graviton3)
      ROWS+=("{\"platform\":\"graviton3\",\"os\":\"linux\",\"display_name\":\"AWS Graviton3\",\"artifact_suffix\":\"graviton3\",\"timeout_minutes\":${RUNSON_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}graviton3\",\"tools_mode\":\"minimal\",\"toolchain_components\":\"${COMPONENTS_STD}\"}")
      ;;
    graviton4)
      ROWS+=("{\"platform\":\"graviton4\",\"os\":\"linux\",\"display_name\":\"AWS Graviton4\",\"artifact_suffix\":\"graviton4\",\"timeout_minutes\":${RUNSON_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}graviton4\",\"tools_mode\":\"minimal\",\"toolchain_components\":\"${COMPONENTS_STD}\"}")
      ;;
    ibm-s390x)
      ROWS+=("{\"platform\":\"ibm-s390x\",\"os\":\"linux\",\"display_name\":\"IBM Z s390x\",\"artifact_suffix\":\"ibm-s390x\",\"timeout_minutes\":${IBM_TIMEOUT_MINUTES},\"runner\":\"ubuntu-24.04-s390x\",\"tools_mode\":\"ibm\",\"toolchain_components\":\"clippy, rustfmt\"}")
      ;;
    ibm-power10)
      ROWS+=("{\"platform\":\"ibm-power10\",\"os\":\"linux\",\"display_name\":\"IBM POWER10 ppc64le\",\"artifact_suffix\":\"ibm-power10\",\"timeout_minutes\":${IBM_TIMEOUT_MINUTES},\"runner\":\"ubuntu-24.04-ppc64le-p10\",\"tools_mode\":\"ibm\",\"toolchain_components\":\"clippy, rustfmt\"}")
      ;;
    rise-riscv)
      ROWS+=("{\"platform\":\"rise-riscv\",\"os\":\"linux\",\"display_name\":\"RISE RISC-V riscv64\",\"artifact_suffix\":\"rise-riscv\",\"timeout_minutes\":${RISCV_TIMEOUT_MINUTES},\"runner\":\"ubuntu-24.04-riscv\",\"tools_mode\":\"ibm\",\"toolchain_components\":\"clippy, rustfmt\"}")
      ;;
    *)
      echo "error: unsupported bench platform '$platform'" >&2
      exit 2
      ;;
  esac
}

GH_RUN_ID_VAL="${GH_RUN_ID:-${GITHUB_RUN_ID:-}}"
if [[ -z "$GH_RUN_ID_VAL" ]]; then
  echo "error: GH_RUN_ID or GITHUB_RUN_ID must be set" >&2
  exit 2
fi

ROWS=()
COMPONENTS_STD="clippy, rustfmt, rust-src"
RUNSON_TIMEOUT_MINUTES=180
IBM_TIMEOUT_MINUTES=240
RISCV_TIMEOUT_MINUTES=240
ALL_PLATFORMS=(
  "amd-zen4"
  "intel-spr"
  "intel-icl"
  "amd-zen5"
  "graviton3"
  "graviton4"
  "ibm-s390x"
  "ibm-power10"
  "rise-riscv"
)

PLATFORMS_INPUT="${BENCH_PLATFORMS:-}"
PLATFORMS_INPUT="$(echo "$PLATFORMS_INPUT" | xargs)"

if [[ -z "$PLATFORMS_INPUT" ]]; then
  echo "error: BENCH_PLATFORMS is required (example: amd-zen4,intel-spr or all)" >&2
  exit 2
fi

SELECTED_PLATFORMS=()
IFS=',' read -r -a platform_tokens <<< "$PLATFORMS_INPUT"
for token in "${platform_tokens[@]:+${platform_tokens[@]}}"; do
  normalized="$(normalize_platform "$token")"
  if [[ -z "$normalized" ]]; then
    echo "error: unknown bench platform '$token'" >&2
    echo "supported: ${ALL_PLATFORMS[*]}, aliases: zen4 spr icl zen5 g3 g4 s390x power10 riscv, or all" >&2
    exit 2
  fi

  if [[ "$normalized" == "all" ]]; then
    for platform in "${ALL_PLATFORMS[@]}"; do
      append_unique "$platform" SELECTED_PLATFORMS
    done
    continue
  fi

  append_unique "$normalized" SELECTED_PLATFORMS
done

for platform in "${SELECTED_PLATFORMS[@]:+${SELECTED_PLATFORMS[@]}}"; do
  append_row_for_platform "$platform"
done

HAS_TARGETS="false"
MATRIX_JSON="[]"

if [[ "${#ROWS[@]}" -gt 0 ]]; then
  HAS_TARGETS="true"
  MATRIX_JSON="$(IFS=,; echo "[${ROWS[*]}]")"
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "has_targets=$HAS_TARGETS" >> "$GITHUB_OUTPUT"
  echo "matrix=$MATRIX_JSON" >> "$GITHUB_OUTPUT"
else
  echo "$MATRIX_JSON"
fi
