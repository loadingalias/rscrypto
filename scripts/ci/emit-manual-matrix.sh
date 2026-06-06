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
    windows-x64|win-x64|windows|win) echo "windows-x64" ;;
    windows-arm64|win-arm64|woa) echo "windows-arm64" ;;
    macos-arm64|mac-arm64|darwin-arm64|apple-silicon|m1|m2|m3|m4) echo "macos-arm64" ;;
    macos-x64|mac-x64|darwin-x64|macos-intel|mac-intel) echo "macos-x64" ;;
    linux-musl-x64|musl-x64|x86_64-musl) echo "linux-musl-x64" ;;
    linux-musl-arm64|musl-arm64|aarch64-musl) echo "linux-musl-arm64" ;;
    thumbv6m|thumb|cortex-m0) echo "thumbv6m-none" ;;
    riscv32imac|riscv32-none) echo "riscv32imac-none" ;;
    aarch64-none|arm64-none) echo "aarch64-none" ;;
    x86_64-none|x64-none) echo "x86_64-none" ;;
    wasm32-unknown|wasm-unknown) echo "wasm32-unknown" ;;
    wasm32-wasip1|wasip1|wasi) echo "wasm32-wasip1" ;;
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
      ROWS+=("{\"platform\":\"ibm-s390x\",\"target\":\"s390x-unknown-linux-gnu\",\"os\":\"linux\",\"display_name\":\"IBM Z s390x\",\"artifact_suffix\":\"ibm-s390x\",\"timeout_minutes\":${CT_IBM_TIMEOUT_MINUTES},\"runner\":\"ubuntu-24.04-s390x\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":false}")
      ;;
    ibm-power10)
      ROWS+=("{\"platform\":\"ibm-power10\",\"target\":\"powerpc64le-unknown-linux-gnu\",\"os\":\"linux\",\"display_name\":\"IBM POWER10 ppc64le\",\"artifact_suffix\":\"ibm-power10\",\"timeout_minutes\":${CT_IBM_TIMEOUT_MINUTES},\"runner\":\"ubuntu-24.04-ppc64le-p10\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":false}")
      ;;
    rise-riscv)
      ROWS+=("{\"platform\":\"rise-riscv\",\"target\":\"riscv64gc-unknown-linux-gnu\",\"os\":\"linux\",\"display_name\":\"RISE RISC-V riscv64\",\"artifact_suffix\":\"rise-riscv\",\"timeout_minutes\":${CT_RISCV_TIMEOUT_MINUTES},\"runner\":\"ubuntu-24.04-riscv\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":false}")
      ;;
    windows-x64)
      ROWS+=("{\"platform\":\"windows-x64\",\"target\":\"x86_64-pc-windows-msvc\",\"os\":\"windows\",\"display_name\":\"Windows x64 MSVC\",\"artifact_suffix\":\"windows-x64\",\"timeout_minutes\":${CT_WINDOWS_TIMEOUT_MINUTES},\"runner\":\"windows-latest\",\"tools_mode\":\"minimal\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":false}")
      ;;
    windows-arm64)
      ROWS+=("{\"platform\":\"windows-arm64\",\"target\":\"aarch64-pc-windows-msvc\",\"os\":\"windows\",\"display_name\":\"Windows ARM64 MSVC\",\"artifact_suffix\":\"windows-arm64\",\"timeout_minutes\":${CT_WINDOWS_TIMEOUT_MINUTES},\"runner\":\"windows-11-arm\",\"tools_mode\":\"minimal\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":false}")
      ;;
    macos-arm64)
      ROWS+=("{\"platform\":\"macos-arm64\",\"target\":\"aarch64-apple-darwin\",\"os\":\"macos\",\"display_name\":\"macOS Apple Silicon\",\"artifact_suffix\":\"macos-arm64\",\"timeout_minutes\":${CT_MACOS_TIMEOUT_MINUTES},\"runner\":\"macos-15-xlarge\",\"tools_mode\":\"minimal\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":false}")
      ;;
    macos-x64)
      ROWS+=("{\"platform\":\"macos-x64\",\"target\":\"x86_64-apple-darwin\",\"os\":\"macos\",\"display_name\":\"macOS Intel\",\"artifact_suffix\":\"macos-x64\",\"timeout_minutes\":${CT_MACOS_TIMEOUT_MINUTES},\"runner\":\"macos-15-large\",\"tools_mode\":\"minimal\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":false}")
      ;;
    linux-musl-x64)
      ROWS+=("{\"platform\":\"linux-musl-x64\",\"target\":\"x86_64-unknown-linux-musl\",\"os\":\"linux\",\"display_name\":\"Linux x64 MUSL\",\"artifact_suffix\":\"linux-musl-x64\",\"timeout_minutes\":${CT_RUNSON_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}linux-x64-ci\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":true}")
      ;;
    linux-musl-arm64)
      ROWS+=("{\"platform\":\"linux-musl-arm64\",\"target\":\"aarch64-unknown-linux-musl\",\"os\":\"linux\",\"display_name\":\"Linux ARM64 MUSL\",\"artifact_suffix\":\"linux-musl-arm64\",\"timeout_minutes\":${CT_RUNSON_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}linux-arm64-ci\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":true}")
      ;;
    thumbv6m-none)
      ROWS+=("{\"platform\":\"thumbv6m-none\",\"target\":\"thumbv6m-none-eabi\",\"os\":\"none\",\"display_name\":\"thumbv6m no_std\",\"artifact_suffix\":\"thumbv6m-none\",\"timeout_minutes\":${CT_CROSS_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}linux-x64-ci\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":true}")
      ;;
    riscv32imac-none)
      ROWS+=("{\"platform\":\"riscv32imac-none\",\"target\":\"riscv32imac-unknown-none-elf\",\"os\":\"none\",\"display_name\":\"RISC-V 32 no_std\",\"artifact_suffix\":\"riscv32imac-none\",\"timeout_minutes\":${CT_CROSS_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}linux-x64-ci\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":true}")
      ;;
    aarch64-none)
      ROWS+=("{\"platform\":\"aarch64-none\",\"target\":\"aarch64-unknown-none\",\"os\":\"none\",\"display_name\":\"AArch64 no_std\",\"artifact_suffix\":\"aarch64-none\",\"timeout_minutes\":${CT_CROSS_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}linux-arm64-ci\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":true}")
      ;;
    x86_64-none)
      ROWS+=("{\"platform\":\"x86_64-none\",\"target\":\"x86_64-unknown-none\",\"os\":\"none\",\"display_name\":\"x86_64 no_std\",\"artifact_suffix\":\"x86_64-none\",\"timeout_minutes\":${CT_CROSS_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}linux-x64-ci\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":true}")
      ;;
    wasm32-unknown)
      ROWS+=("{\"platform\":\"wasm32-unknown\",\"target\":\"wasm32-unknown-unknown\",\"os\":\"wasm\",\"display_name\":\"WASM unknown\",\"artifact_suffix\":\"wasm32-unknown\",\"timeout_minutes\":${CT_CROSS_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}linux-x64-ci\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":true}")
      ;;
    wasm32-wasip1)
      ROWS+=("{\"platform\":\"wasm32-wasip1\",\"target\":\"wasm32-wasip1\",\"os\":\"wasm\",\"display_name\":\"WASM WASI Preview 1\",\"artifact_suffix\":\"wasm32-wasip1\",\"timeout_minutes\":${CT_CROSS_TIMEOUT_MINUTES},\"runner\":\"${runner_uarch}linux-x64-ci\",\"tools_mode\":\"ct-linux\",\"toolchain_components\":\"${COMPONENTS_CT}\",\"enable_magic_cache\":true}")
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
  "windows-x64"
  "windows-arm64"
  "macos-arm64"
  "macos-x64"
  "linux-musl-x64"
  "linux-musl-arm64"
)
CT_RUNSON_TIMEOUT_MINUTES=360
CT_IBM_TIMEOUT_MINUTES=420
CT_RISCV_TIMEOUT_MINUTES=480
CT_WINDOWS_TIMEOUT_MINUTES=360
CT_MACOS_TIMEOUT_MINUTES=360
CT_CROSS_TIMEOUT_MINUTES=360

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
    echo "supported: ${ALL_PLATFORMS[*]}, aliases: zen4 spr icl zen5 g3 g4 s390x power10 riscv windows windows-arm64, or all" >&2
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
