#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  bench|ct) ;;
  *)
    echo "usage: scripts/ci/emit-manual-matrix.sh <bench|ct>" >&2
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

platform_is_supported() {
  local needle="${1:-}"
  local item
  for item in "${ALL_PLATFORMS[@]:+${ALL_PLATFORMS[@]}}"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
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

append_ct_row_for_platform() {
  local platform="${1:-}"
  local runner_uarch="runs-on=${GH_RUN_ID_VAL}/runner="
  case "$platform" in
    amd-zen4)
      ROWS+=("{\"platform\":\"amd-zen4\",\"target\":\"x86_64-unknown-linux-gnu\",\"os\":\"linux\",\"display_name\":\"AMD Zen4\",\"artifact_suffix\":\"amd-zen4\",\"timeout_minutes\":${CT_RUNSON_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}amd-zen4\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":true}")
      ;;
    intel-spr)
      ROWS+=("{\"platform\":\"intel-spr\",\"target\":\"x86_64-unknown-linux-gnu\",\"os\":\"linux\",\"display_name\":\"Intel Sapphire Rapids\",\"artifact_suffix\":\"intel-spr\",\"timeout_minutes\":${CT_RUNSON_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}intel-spr\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":true}")
      ;;
    intel-icl)
      ROWS+=("{\"platform\":\"intel-icl\",\"target\":\"x86_64-unknown-linux-gnu\",\"os\":\"linux\",\"display_name\":\"Intel Ice Lake\",\"artifact_suffix\":\"intel-icl\",\"timeout_minutes\":${CT_RUNSON_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}intel-icl\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":true}")
      ;;
    amd-zen5)
      ROWS+=("{\"platform\":\"amd-zen5\",\"target\":\"x86_64-unknown-linux-gnu\",\"os\":\"linux\",\"display_name\":\"AMD Zen5\",\"artifact_suffix\":\"amd-zen5\",\"timeout_minutes\":${CT_RUNSON_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}amd-zen5\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":true}")
      ;;
    graviton3)
      ROWS+=("{\"platform\":\"graviton3\",\"target\":\"aarch64-unknown-linux-gnu\",\"os\":\"linux\",\"display_name\":\"AWS Graviton3\",\"artifact_suffix\":\"graviton3\",\"timeout_minutes\":${CT_RUNSON_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}graviton3\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":true}")
      ;;
    graviton4)
      ROWS+=("{\"platform\":\"graviton4\",\"target\":\"aarch64-unknown-linux-gnu\",\"os\":\"linux\",\"display_name\":\"AWS Graviton4\",\"artifact_suffix\":\"graviton4\",\"timeout_minutes\":${CT_RUNSON_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}graviton4\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":true}")
      ;;
    ibm-s390x)
      ROWS+=("{\"platform\":\"ibm-s390x\",\"target\":\"s390x-unknown-linux-gnu\",\"os\":\"linux\",\"display_name\":\"IBM Z s390x\",\"artifact_suffix\":\"ibm-s390x\",\"timeout_minutes\":${CT_IBM_TIMEOUT_MINUTES},\"runner\":\"ubuntu-24.04-s390x\",\"tools_mode\":\"ibm\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":false}")
      ;;
    ibm-power10)
      ROWS+=("{\"platform\":\"ibm-power10\",\"target\":\"powerpc64le-unknown-linux-gnu\",\"os\":\"linux\",\"display_name\":\"IBM POWER10 ppc64le\",\"artifact_suffix\":\"ibm-power10\",\"timeout_minutes\":${CT_IBM_TIMEOUT_MINUTES},\"runner\":\"ubuntu-24.04-ppc64le-p10\",\"tools_mode\":\"ibm\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":false}")
      ;;
    rise-riscv)
      ROWS+=("{\"platform\":\"rise-riscv\",\"target\":\"riscv64gc-unknown-linux-gnu\",\"os\":\"linux\",\"display_name\":\"RISE RISC-V riscv64\",\"artifact_suffix\":\"rise-riscv\",\"timeout_minutes\":${CT_RISCV_TIMEOUT_MINUTES},\"runner\":\"ubuntu-24.04-riscv\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":false}")
      ;;
    *)
      echo "error: unsupported CT platform '$platform'" >&2
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
COMPONENTS_CT="clippy, rustfmt, rust-src, llvm-tools-preview"
RUNSON_TIMEOUT_MINUTES=180
IBM_TIMEOUT_MINUTES=240
RISCV_TIMEOUT_MINUTES=240
BENCH_PLATFORMS_ALL=(
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
CT_PLATFORMS_ALL=(
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
CT_RUNSON_TIMEOUT_MINUTES=360
CT_IBM_TIMEOUT_MINUTES=420
CT_RISCV_TIMEOUT_MINUTES=480

if [[ "$MODE" == "ct" ]]; then
  PLATFORMS_INPUT="${CT_PLATFORMS:-}"
  ALL_PLATFORMS=("${CT_PLATFORMS_ALL[@]}")
else
  PLATFORMS_INPUT="${BENCH_PLATFORMS:-}"
  ALL_PLATFORMS=("${BENCH_PLATFORMS_ALL[@]}")
fi
PLATFORMS_INPUT="$(echo "$PLATFORMS_INPUT" | xargs)"

if [[ -z "$PLATFORMS_INPUT" ]]; then
  echo "error: ${MODE^^}_PLATFORMS is required (example: amd-zen4,intel-spr or all)" >&2
  exit 2
fi

SELECTED_PLATFORMS=()
IFS=',' read -r -a platform_tokens <<< "$PLATFORMS_INPUT"
for token in "${platform_tokens[@]:+${platform_tokens[@]}}"; do
  normalized="$(normalize_platform "$token")"
  if [[ -z "$normalized" ]]; then
    echo "error: unknown $MODE platform '$token'" >&2
    echo "supported: ${ALL_PLATFORMS[*]}, aliases: zen4 spr icl zen5 g3 g4 s390x power10 riscv, or all" >&2
    exit 2
  fi

  if [[ "$normalized" == "all" ]]; then
    for platform in "${ALL_PLATFORMS[@]}"; do
      append_unique "$platform" SELECTED_PLATFORMS
    done
    continue
  fi

  if ! platform_is_supported "$normalized"; then
    echo "error: unsupported $MODE platform '$token' (normalized: $normalized)" >&2
    echo "supported: ${ALL_PLATFORMS[*]}, aliases: zen4 spr icl zen5 g3 g4 s390x power10 riscv, or all" >&2
    exit 2
  fi

  append_unique "$normalized" SELECTED_PLATFORMS
done

for platform in "${SELECTED_PLATFORMS[@]:+${SELECTED_PLATFORMS[@]}}"; do
  if [[ "$MODE" == "ct" ]]; then
    append_ct_row_for_platform "$platform"
  else
    append_row_for_platform "$platform"
  fi
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
