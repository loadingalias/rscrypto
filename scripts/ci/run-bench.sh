#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${BENCH_OUTPUT_DIR:-benchmark-results}"
mkdir -p "$OUT_DIR"

to_bool() {
  local raw="${1:-}"
  raw="$(echo "$raw" | tr '[:upper:]' '[:lower:]' | xargs)"
  case "$raw" in
    1|true|yes|on|y) echo "true" ;;
    0|false|no|off|n|"") echo "false" ;;
    *)
      echo "warning: unrecognized boolean value '$1'; treating as false" >&2
      echo "false"
      ;;
  esac
}

array_contains() {
  local needle="${1:-}"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

append_unique() {
  local value="${1:-}"
  local array_name="${2:-}"
  local -a current=()
  local item
  [[ -z "$value" ]] && return 0
  [[ -z "$array_name" ]] && return 0
  eval "current=(\"\${${array_name}[@]:+\${${array_name}[@]}}\")"
  for item in "${current[@]:+${current[@]}}"; do
    if [[ "$item" == "$value" ]]; then
      return 0
    fi
  done
  eval "$array_name+=(\"\$value\")"
}

normalize_csv_lower() {
  local raw="${1:-}"
  local -a parts=()
  local -a normalized=()
  local token
  IFS=',' read -r -a parts <<< "$raw"
  for token in "${parts[@]:+${parts[@]}}"; do
    token="$(echo "$token" | xargs)"
    [[ -z "$token" ]] && continue
    token="$(echo "$token" | tr '[:upper:]' '[:lower:]')"
    append_unique "$token" normalized
  done

  if [[ "${#normalized[@]}" -eq 0 ]]; then
    echo ""
  else
    (IFS=','; echo "${normalized[*]}")
  fi
}

normalize_csv_raw() {
  local raw="${1:-}"
  local -a parts=()
  local -a normalized=()
  local token
  IFS=',' read -r -a parts <<< "$raw"
  for token in "${parts[@]:+${parts[@]}}"; do
    token="$(echo "$token" | xargs)"
    [[ -z "$token" ]] && continue
    append_unique "$token" normalized
  done

  if [[ "${#normalized[@]}" -eq 0 ]]; then
    echo ""
  else
    (IFS=','; echo "${normalized[*]}")
  fi
}

normalize_selector() {
  echo "${1:-}" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]'
}

ms_to_seconds() {
  local ms="${1:-}"
  awk -v ms="$ms" 'BEGIN { printf "%.3f", (ms + 0) / 1000.0 }'
}

DEFAULT_CHECKSUM_ALGOS=(
  "crc16-ccitt"
  "crc16-ibm"
  "crc24-openpgp"
  "crc32-ieee"
  "crc32c"
  "crc64-xz"
  "crc64-nvme"
)

DEFAULT_HASH_ALGOS=(
  "sha224"
  "sha256"
  "sha384"
  "sha512"
  "sha512-224"
  "sha512-256"
  "sha3-224"
  "sha3-256"
  "sha3-384"
  "sha3-512"
  "shake128"
  "shake256"
  "blake2b-512"
  "blake2s-256"
  "blake3"
  "xxh3"
  "rapidhash"
  "siphash"
  "keccakf1600"
  "ascon-hash256"
  "ascon-xof128"
)

ALL_KNOWN_ALGOS=("${DEFAULT_CHECKSUM_ALGOS[@]}" "${DEFAULT_HASH_ALGOS[@]}")

checksum_filter_token() {
  local algo="${1:-}"
  case "$algo" in
    crc16-ccitt) echo "crc16/ccitt" ;;
    crc16-ibm) echo "crc16/ibm" ;;
    crc24-openpgp) echo "crc24/openpgp" ;;
    crc32-ieee) echo "crc32/ieee" ;;
    crc32c) echo "crc32c" ;;
    crc64-xz) echo "crc64/xz" ;;
    crc64-nvme) echo "crc64/nvme" ;;
    *) echo "$algo" ;;
  esac
}

hash_filter_token() {
  local algo="${1:-}"
  case "$algo" in
    sha3-224) echo "sha3_224" ;;
    sha3-256) echo "sha3_256" ;;
    sha3-384) echo "sha3_384" ;;
    sha3-512) echo "sha3_512" ;;
    sha512-224) echo "sha512_224" ;;
    sha512-256) echo "sha512_256" ;;
    blake2b-512) echo "blake2b512" ;;
    blake2s-256) echo "blake2s256" ;;
    ascon-hash256) echo "ascon_hash256" ;;
    ascon-xof128) echo "ascon_xof128" ;;
    keccakf1600) echo "keccak" ;;
    xxh3) echo "xxh3" ;;
    blake3) echo "blake3" ;;
    *) echo "$algo" ;;
  esac
}

default_benches_for_crate() {
  local crate="${1:-}"
  case "$crate" in
    checksum) echo "comp,kernels" ;;
    hashes) echo "comp,kernels,blake3" ;;
    *) echo "" ;;
  esac
}

supports_hash_kernels() {
  local algo="${1:-}"
  case "$algo" in
    sha256|sha512|sha3-256|sha3-512|shake256|xxh3|rapidhash|siphash|blake3) return 0 ;;
    *) return 1 ;;
  esac
}

build_algo_plan_rows() {
  local algo="${1:-}"
  local token
  token="$(hash_filter_token "$algo")"

  if array_contains "$algo" "${DEFAULT_CHECKSUM_ALGOS[@]}"; then
    token="$(checksum_filter_token "$algo")"
    PLAN_ROWS+=("checksum|comp|$token")
    PLAN_ROWS+=("checksum|kernels|$token")
    return 0
  fi

  if [[ "$algo" == "blake3" ]]; then
    # Dedicated bench target already includes oneshot/streaming/keyed/xof/derive variants.
    PLAN_ROWS+=("hashes|blake3|blake3")
    return 0
  fi

  if supports_hash_kernels "$algo"; then
    PLAN_ROWS+=("hashes|comp|$token")
    PLAN_ROWS+=("hashes|kernels|$token")
  else
    PLAN_ROWS+=("hashes|comp|$token")
  fi
}

dedupe_plan_rows() {
  local -a unique=()
  local row
  for row in "${PLAN_ROWS[@]:+${PLAN_ROWS[@]}}"; do
    append_unique "$row" unique
  done
  PLAN_ROWS=("${unique[@]:+${unique[@]}}")
}

CRATES_INPUT="$(normalize_csv_lower "${BENCH_CRATES:-}")"
BENCHES_INPUT="$(normalize_csv_lower "${BENCH_BENCHES:-}")"
ONLY_INPUT="$(normalize_csv_lower "${BENCH_ONLY:-}")"
FILTER_INPUT="$(normalize_csv_raw "${BENCH_FILTER:-}")"
QUICK_INPUT="$(to_bool "${BENCH_QUICK:-false}")"
CLEAN_INPUT="$(to_bool "${BENCH_CLEAN:-true}")"
WARMUP_MS_INPUT="${BENCH_WARMUP_MS:-}"
MEASURE_MS_INPUT="${BENCH_MEASURE_MS:-}"
SAMPLE_SIZE_INPUT="${BENCH_SAMPLE_SIZE:-}"
PROFILE_TIME_SECS_INPUT="${BENCH_PROFILE_TIME_SECS:-}"

if [[ -n "$WARMUP_MS_INPUT" && ! "$WARMUP_MS_INPUT" =~ ^[0-9]+$ ]]; then
  echo "error: BENCH_WARMUP_MS must be an integer >= 0 (got '$WARMUP_MS_INPUT')" >&2
  exit 2
fi

if [[ -n "$MEASURE_MS_INPUT" && ! "$MEASURE_MS_INPUT" =~ ^[0-9]+$ ]]; then
  echo "error: BENCH_MEASURE_MS must be an integer >= 0 (got '$MEASURE_MS_INPUT')" >&2
  exit 2
fi

if [[ -n "$SAMPLE_SIZE_INPUT" && (! "$SAMPLE_SIZE_INPUT" =~ ^[0-9]+$ || "$SAMPLE_SIZE_INPUT" -lt 10) ]]; then
  echo "error: BENCH_SAMPLE_SIZE must be an integer >= 10 (got '$SAMPLE_SIZE_INPUT')" >&2
  exit 2
fi

if [[ -n "$PROFILE_TIME_SECS_INPUT" && ! "$PROFILE_TIME_SECS_INPUT" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "error: BENCH_PROFILE_TIME_SECS must be numeric (got '$PROFILE_TIME_SECS_INPUT')" >&2
  exit 2
fi

CRITERION_ARGS=()
if [[ "$QUICK_INPUT" == "true" ]]; then
  CRITERION_ARGS+=(--quick --noplot)
  if [[ -n "$WARMUP_MS_INPUT" || -n "$MEASURE_MS_INPUT" || -n "$SAMPLE_SIZE_INPUT" || -n "$PROFILE_TIME_SECS_INPUT" ]]; then
    echo "note: BENCH_QUICK=true ignores BENCH_WARMUP_MS/BENCH_MEASURE_MS/BENCH_SAMPLE_SIZE/BENCH_PROFILE_TIME_SECS"
  fi
else
  if [[ -n "$WARMUP_MS_INPUT" ]]; then
    CRITERION_ARGS+=(--warm-up-time "$(ms_to_seconds "$WARMUP_MS_INPUT")")
  fi
  if [[ -n "$MEASURE_MS_INPUT" ]]; then
    CRITERION_ARGS+=(--measurement-time "$(ms_to_seconds "$MEASURE_MS_INPUT")")
  fi
  if [[ -n "$SAMPLE_SIZE_INPUT" ]]; then
    CRITERION_ARGS+=(--sample-size "$SAMPLE_SIZE_INPUT")
  fi
  if [[ -n "$PROFILE_TIME_SECS_INPUT" ]]; then
    CRITERION_ARGS+=(--profile-time "$PROFILE_TIME_SECS_INPUT")
  fi
fi

if [[ "$CLEAN_INPUT" == "true" ]]; then
  rm -rf target/criterion
fi

LOG_PATH="$OUT_DIR/output.txt"
: > "$LOG_PATH"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running benchmark pipeline"
echo "Quick mode: $QUICK_INPUT"
if [[ -n "$ONLY_INPUT" ]]; then
  echo "Selectors: $ONLY_INPUT"
fi
if [[ -n "$CRATES_INPUT" ]]; then
  echo "Crates override: $CRATES_INPUT"
fi
if [[ -n "$BENCHES_INPUT" ]]; then
  echo "Benches override: $BENCHES_INPUT"
fi
if [[ -n "$FILTER_INPUT" ]]; then
  echo "Filter override: $FILTER_INPUT"
fi
echo "Criterion args: ${CRITERION_ARGS[*]-<none>}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PLAN_ROWS=()
RAW_FILTERS=()
SELECTED_ALGOS=()

if [[ -n "$ONLY_INPUT" ]]; then
  IFS=',' read -r -a only_values <<< "$ONLY_INPUT"
  for selector in "${only_values[@]:+${only_values[@]}}"; do
    key="$(normalize_selector "$selector")"
    case "$key" in
      all)
        for algo in "${ALL_KNOWN_ALGOS[@]}"; do append_unique "$algo" SELECTED_ALGOS; done
        ;;
      checksum|checksums)
        for algo in "${DEFAULT_CHECKSUM_ALGOS[@]}"; do append_unique "$algo" SELECTED_ALGOS; done
        ;;
      hashes|hash)
        for algo in "${DEFAULT_HASH_ALGOS[@]}"; do append_unique "$algo" SELECTED_ALGOS; done
        ;;
      blake3)
        append_unique "blake3" SELECTED_ALGOS
        ;;
      crc64|crc64nvme|crc64xz)
        append_unique "crc64-xz" SELECTED_ALGOS
        append_unique "crc64-nvme" SELECTED_ALGOS
        ;;
      crc32)
        append_unique "crc32-ieee" SELECTED_ALGOS
        append_unique "crc32c" SELECTED_ALGOS
        ;;
      crc16)
        append_unique "crc16-ccitt" SELECTED_ALGOS
        append_unique "crc16-ibm" SELECTED_ALGOS
        ;;
      *)
        matched="false"
        for algo in "${ALL_KNOWN_ALGOS[@]}"; do
          if [[ "$(normalize_selector "$algo")" == "$key" ]]; then
            append_unique "$algo" SELECTED_ALGOS
            matched="true"
            break
          fi
        done
        if [[ "$matched" == "false" ]]; then
          append_unique "$selector" RAW_FILTERS
        fi
        ;;
    esac
  done

  for algo in "${SELECTED_ALGOS[@]:+${SELECTED_ALGOS[@]}}"; do
    build_algo_plan_rows "$algo"
  done
fi

if [[ -n "$FILTER_INPUT" ]]; then
  IFS=',' read -r -a raw_filter_values <<< "$FILTER_INPUT"
  for f in "${raw_filter_values[@]:+${raw_filter_values[@]}}"; do
    append_unique "$f" RAW_FILTERS
  done
fi

if [[ "${#RAW_FILTERS[@]}" -gt 0 ]]; then
  raw_crates=()
  raw_benches=()

  if [[ -n "$CRATES_INPUT" ]]; then
    IFS=',' read -r -a raw_crates <<< "$CRATES_INPUT"
  elif [[ "${#PLAN_ROWS[@]}" -gt 0 ]]; then
    for row in "${PLAN_ROWS[@]}"; do
      IFS='|' read -r crate _ _ <<< "$row"
      append_unique "$crate" raw_crates
    done
  else
    raw_crates=("checksum" "hashes")
  fi

  if [[ -n "$BENCHES_INPUT" ]]; then
    IFS=',' read -r -a raw_benches <<< "$BENCHES_INPUT"
  fi

  for filter in "${RAW_FILTERS[@]}"; do
    for crate in "${raw_crates[@]:+${raw_crates[@]}}"; do
      benches_csv=""
      if [[ "${#raw_benches[@]}" -gt 0 ]]; then
        benches_csv="$(IFS=','; echo "${raw_benches[*]}")"
      else
        benches_csv="$(default_benches_for_crate "$crate")"
      fi

      if [[ -z "$benches_csv" ]]; then
        echo "warning: no default bench set for crate '$crate'; skipping raw filter '$filter'" | tee -a "$LOG_PATH"
        continue
      fi

      IFS=',' read -r -a benches_values <<< "$benches_csv"
      for bench in "${benches_values[@]:+${benches_values[@]}}"; do
        PLAN_ROWS+=("$crate|$bench|$filter")
      done
    done
  done
fi

dedupe_plan_rows

if [[ -n "$CRATES_INPUT" && "${#PLAN_ROWS[@]}" -gt 0 ]]; then
  IFS=',' read -r -a crate_filters <<< "$CRATES_INPUT"
  filtered=()
  for row in "${PLAN_ROWS[@]}"; do
    IFS='|' read -r crate bench filter <<< "$row"
    if array_contains "$crate" "${crate_filters[@]}"; then
      filtered+=("$crate|$bench|$filter")
    fi
  done
  PLAN_ROWS=("${filtered[@]:+${filtered[@]}}")
fi

if [[ -n "$BENCHES_INPUT" && "${#PLAN_ROWS[@]}" -gt 0 ]]; then
  IFS=',' read -r -a bench_filters <<< "$BENCHES_INPUT"
  filtered=()
  for row in "${PLAN_ROWS[@]}"; do
    IFS='|' read -r crate bench filter <<< "$row"
    if array_contains "$bench" "${bench_filters[@]}"; then
      filtered+=("$crate|$bench|$filter")
    fi
  done
  PLAN_ROWS=("${filtered[@]:+${filtered[@]}}")
fi

run_bench_cmd() {
  local crate="$1"
  local bench="$2"
  local filter="${3:-}"
  local -a cmd=(cargo bench --profile bench -p "$crate" --bench "$bench")
  if [[ -n "$filter" || "${#CRITERION_ARGS[@]}" -gt 0 ]]; then
    cmd+=(--)
    if [[ -n "$filter" ]]; then
      cmd+=("$filter")
    fi
    if [[ "${#CRITERION_ARGS[@]}" -gt 0 ]]; then
      cmd+=("${CRITERION_ARGS[@]}")
    fi
  fi

  echo "" | tee -a "$LOG_PATH"
  echo "Running: ${cmd[*]}" | tee -a "$LOG_PATH"
  "${cmd[@]}" 2>&1 | tee -a "$LOG_PATH"
}

if [[ "${#PLAN_ROWS[@]}" -gt 0 ]]; then
  echo "Execution plan (${#PLAN_ROWS[@]} row(s)):" | tee -a "$LOG_PATH"
  for row in "${PLAN_ROWS[@]}"; do
    IFS='|' read -r crate bench filter <<< "$row"
    echo "  - crate=$crate bench=$bench filter=$filter" | tee -a "$LOG_PATH"
  done

  for row in "${PLAN_ROWS[@]}"; do
    IFS='|' read -r crate bench filter <<< "$row"
    run_bench_cmd "$crate" "$bench" "$filter"
  done

  exit 0
fi

CRATE_FLAGS=()
BENCH_FLAGS=()
if [[ -n "$CRATES_INPUT" ]]; then
  IFS=',' read -r -a crates_values <<< "$CRATES_INPUT"
  for crate in "${crates_values[@]:+${crates_values[@]}}"; do
    CRATE_FLAGS+=(-p "$crate")
  done
else
  CRATE_FLAGS+=(--workspace)
fi

if [[ -n "$BENCHES_INPUT" ]]; then
  IFS=',' read -r -a benches_values <<< "$BENCHES_INPUT"
  for bench in "${benches_values[@]:+${benches_values[@]}}"; do
    BENCH_FLAGS+=(--bench "$bench")
  done
fi

GENERIC_FILTER=""
if [[ -n "$FILTER_INPUT" ]]; then
  IFS=',' read -r -a filters_values <<< "$FILTER_INPUT"
  GENERIC_FILTER="${filters_values[0]}"
fi

echo "No selector plan generated; running generic cargo bench invocation." | tee -a "$LOG_PATH"
if [[ -n "$GENERIC_FILTER" ]]; then
  echo "Using first filter token for generic run: $GENERIC_FILTER" | tee -a "$LOG_PATH"
fi

cmd=(cargo bench --profile bench "${CRATE_FLAGS[@]}" "${BENCH_FLAGS[@]}")
if [[ -n "$GENERIC_FILTER" || "${#CRITERION_ARGS[@]}" -gt 0 ]]; then
  cmd+=(--)
  if [[ -n "$GENERIC_FILTER" ]]; then
    cmd+=("$GENERIC_FILTER")
  fi
  if [[ "${#CRITERION_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${CRITERION_ARGS[@]}")
  fi
fi

echo "Running: ${cmd[*]}" | tee -a "$LOG_PATH"
"${cmd[@]}" 2>&1 | tee -a "$LOG_PATH"
