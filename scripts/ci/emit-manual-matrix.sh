#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  bench|tune) ;;
  *)
    echo "usage: scripts/ci/emit-manual-matrix.sh <bench|tune>" >&2
    exit 2
    ;;
esac

to_bool() {
  local raw="${1:-}"
  raw="$(echo "$raw" | tr '[:upper:]' '[:lower:]' | xargs)"
  case "$raw" in
    1|true|yes|on|y) echo "true" ;;
    *) echo "false" ;;
  esac
}

emit_row_if_enabled() {
  local enabled="${1:-false}"
  local row="${2:-}"
  if [[ "$(to_bool "$enabled")" == "true" ]]; then
    ROWS+=("$row")
  fi
}

GH_RUN_ID_VAL="${GH_RUN_ID:-${GITHUB_RUN_ID:-}}"
if [[ -z "$GH_RUN_ID_VAL" ]]; then
  echo "error: GH_RUN_ID or GITHUB_RUN_ID must be set" >&2
  exit 2
fi

ROWS=()
COMPONENTS_STD="clippy, rustfmt, rust-src"
RUNSON_TIMEOUT_MINUTES=120
if [[ "$MODE" == "tune" ]]; then
  RUNSON_TIMEOUT_MINUTES=55
fi

emit_row_if_enabled "${RUN_AMD_ZEN4:-false}" \
  "{\"platform\":\"amd-zen4\",\"display_name\":\"AMD Zen4\",\"artifact_suffix\":\"amd-zen4\",\"timeout_minutes\":${RUNSON_TIMEOUT_MINUTES},\"setup_kind\":\"runson\",\"runner\":\"runs-on=${GH_RUN_ID_VAL}/runner=amd-zen4\",\"cache_key\":\"\",\"tools_mode\":\"standard\",\"toolchain_components\":\"${COMPONENTS_STD}\",\"runson_mode\":\"runson-bench\"}"

emit_row_if_enabled "${RUN_INTEL_SPR:-false}" \
  "{\"platform\":\"intel-spr\",\"display_name\":\"Intel Sapphire Rapids\",\"artifact_suffix\":\"intel-spr\",\"timeout_minutes\":${RUNSON_TIMEOUT_MINUTES},\"setup_kind\":\"runson\",\"runner\":\"runs-on=${GH_RUN_ID_VAL}/runner=intel-spr\",\"cache_key\":\"\",\"tools_mode\":\"standard\",\"toolchain_components\":\"${COMPONENTS_STD}\",\"runson_mode\":\"runson-bench\"}"

emit_row_if_enabled "${RUN_INTEL_ICL:-false}" \
  "{\"platform\":\"intel-icl\",\"display_name\":\"Intel Ice Lake\",\"artifact_suffix\":\"intel-icl\",\"timeout_minutes\":${RUNSON_TIMEOUT_MINUTES},\"setup_kind\":\"runson\",\"runner\":\"runs-on=${GH_RUN_ID_VAL}/runner=intel-icl\",\"cache_key\":\"\",\"tools_mode\":\"standard\",\"toolchain_components\":\"${COMPONENTS_STD}\",\"runson_mode\":\"runson-bench\"}"

emit_row_if_enabled "${RUN_AMD_ZEN5:-false}" \
  "{\"platform\":\"amd-zen5\",\"display_name\":\"AMD Zen5\",\"artifact_suffix\":\"amd-zen5\",\"timeout_minutes\":${RUNSON_TIMEOUT_MINUTES},\"setup_kind\":\"runson\",\"runner\":\"runs-on=${GH_RUN_ID_VAL}/runner=amd-zen5\",\"cache_key\":\"\",\"tools_mode\":\"standard\",\"toolchain_components\":\"${COMPONENTS_STD}\",\"runson_mode\":\"runson-bench\"}"

emit_row_if_enabled "${RUN_GRAVITON3:-false}" \
  "{\"platform\":\"graviton3\",\"display_name\":\"AWS Graviton3\",\"artifact_suffix\":\"graviton3\",\"timeout_minutes\":${RUNSON_TIMEOUT_MINUTES},\"setup_kind\":\"runson\",\"runner\":\"runs-on=${GH_RUN_ID_VAL}/runner=graviton3\",\"cache_key\":\"\",\"tools_mode\":\"standard\",\"toolchain_components\":\"${COMPONENTS_STD}\",\"runson_mode\":\"runson-bench\"}"

emit_row_if_enabled "${RUN_GRAVITON4:-false}" \
  "{\"platform\":\"graviton4\",\"display_name\":\"AWS Graviton4\",\"artifact_suffix\":\"graviton4\",\"timeout_minutes\":${RUNSON_TIMEOUT_MINUTES},\"setup_kind\":\"runson\",\"runner\":\"runs-on=${GH_RUN_ID_VAL}/runner=graviton4\",\"cache_key\":\"\",\"tools_mode\":\"standard\",\"toolchain_components\":\"${COMPONENTS_STD}\",\"runson_mode\":\"runson-bench\"}"

emit_row_if_enabled "${RUN_IBM_S390X:-false}" \
  "{\"platform\":\"ibm-s390x\",\"display_name\":\"IBM Z s390x\",\"artifact_suffix\":\"ibm-s390x\",\"timeout_minutes\":120,\"setup_kind\":\"default\",\"runner\":\"ubuntu-24.04-s390x\",\"cache_key\":\"manual-ibm-${MODE}-s390x\",\"tools_mode\":\"none\",\"toolchain_components\":\"\",\"runson_mode\":\"runson-bench\"}"

emit_row_if_enabled "${RUN_IBM_POWER10:-false}" \
  "{\"platform\":\"ibm-power10\",\"display_name\":\"IBM POWER10 ppc64le\",\"artifact_suffix\":\"ibm-power10\",\"timeout_minutes\":120,\"setup_kind\":\"default\",\"runner\":\"ubuntu-24.04-ppc64le-p10\",\"cache_key\":\"manual-ibm-${MODE}-power10\",\"tools_mode\":\"none\",\"toolchain_components\":\"\",\"runson_mode\":\"runson-bench\"}"

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
