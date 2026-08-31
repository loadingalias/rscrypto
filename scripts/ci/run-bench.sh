#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${BENCH_OUTPUT_DIR:-target/benchmark_results}"

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

csv_has_token() {
  local csv="${1:-}"
  local needle="${2:-}"
  local -a parts=()
  local token
  [[ -z "$csv" || -z "$needle" ]] && return 1
  IFS=',' read -r -a parts <<< "$csv"
  for token in "${parts[@]:+${parts[@]}}"; do
    if [[ "$token" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

normalize_selector() {
  echo "${1:-}" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]'
}

ms_to_seconds() {
  local ms="${1:-}"
  awk -v ms="$ms" 'BEGIN { printf "%.3f", (ms + 0) / 1000.0 }'
}

BENCHMARK_CATALOG="scripts/bench/benchmark_catalog.py"
PYTHON="$(scripts/lib/python.sh --print)"

catalog() {
  "$PYTHON" "$BENCHMARK_CATALOG" "$@"
}

default_benches_for_crate() {
  catalog default-benches "${1:-}"
}

bench_features_for_target() {
  catalog features "${1:-}"
}

bench_binary_for_target() {
  catalog binary "${1:-}"
}

bench_features_for_invocation() {
  catalog features "${1:-}"
}

append_algo_plan_rows() {
  local algorithms_csv="${1:-}"
  local raw_filter="${2:-}"
  local row
  while IFS= read -r row; do
    [[ -n "$row" ]] && PLAN_ROWS+=("$row")
  done < <(catalog plan-algorithms "$algorithms_csv" "$raw_filter")
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

expand_bench_shorthand() {
  catalog expand-benches "$1"
}
BENCHES_INPUT="$(expand_bench_shorthand "$BENCHES_INPUT")"

ONLY_INPUT="$(normalize_csv_lower "${BENCH_ONLY:-}")"
FILTER_INPUT="$(normalize_csv_raw "${BENCH_FILTER:-}")"
QUICK_INPUT="$(to_bool "${BENCH_QUICK:-false}")"
CLEAN_INPUT="$(to_bool "${BENCH_CLEAN:-true}")"
WARMUP_MS_INPUT="${BENCH_WARMUP_MS:-}"
MEASURE_MS_INPUT="${BENCH_MEASURE_MS:-}"
SAMPLE_SIZE_INPUT="${BENCH_SAMPLE_SIZE:-}"
PROFILE_TIME_SECS_INPUT="${BENCH_PROFILE_TIME_SECS:-}"
ATTACH_CRITERION_INPUT="$(to_bool "${BENCH_ATTACH_CRITERION:-false}")"
DRY_RUN_PLAN_INPUT="$(to_bool "${BENCH_DRY_RUN_PLAN:-false}")"
ALLOW_FULL_HASHES_COMP_INPUT="$(to_bool "${BENCH_ALLOW_FULL_HASHES_COMP:-false}")"

ENFORCE_BLAKE3_GAP_GATE_INPUT="$(to_bool "${BENCH_ENFORCE_BLAKE3_GAP_GATE:-false}")"
PLATFORM_INPUT="$(echo "${BENCH_PLATFORM:-}" | tr '[:upper:]' '[:lower:]' | xargs)"

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

if [[ -n "$PROFILE_TIME_SECS_INPUT" ]]; then
  if [[ ! "$PROFILE_TIME_SECS_INPUT" =~ ^[0-9]+(\.[0-9]+)?$ ]] \
    || ! awk -v seconds="$PROFILE_TIME_SECS_INPUT" 'BEGIN { exit !(seconds >= 1) }'; then
    echo "error: BENCH_PROFILE_TIME_SECS must be numeric and at least 1 (got '$PROFILE_TIME_SECS_INPUT')" >&2
    exit 2
  fi
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
  rm -rf target/criterion || true
fi

if [[ "$ENFORCE_BLAKE3_GAP_GATE_INPUT" == "true" && "$QUICK_INPUT" == "true" ]]; then
  echo "error: BENCH_ENFORCE_BLAKE3_GAP_GATE=true requires BENCH_QUICK=false" >&2
  exit 2
fi

targets_hashes="false"
targets_comp="false"
if [[ -z "$CRATES_INPUT" ]] || csv_has_token "$CRATES_INPUT" "hashes"; then
  targets_hashes="true"
fi
if [[ -z "$BENCHES_INPUT" ]]; then
  targets_comp="true"
else
  for bench in sha2 sha3 kmac_cshake ascon xxh3 rapidhash; do
    if csv_has_token "$BENCHES_INPUT" "$bench"; then
      targets_comp="true"
      break
    fi
  done
fi
if [[ "$QUICK_INPUT" != "true" \
  && "$ALLOW_FULL_HASHES_COMP_INPUT" != "true" \
  && "$targets_hashes" == "true" \
  && "$targets_comp" == "true" \
  && -z "$ONLY_INPUT" \
  && -z "$FILTER_INPUT" ]]; then
  echo "error: refusing unscoped hashes/comp run (expensive and often timeout-prone on CI lanes)." >&2
  echo "hint: set BENCH_ONLY and/or BENCH_FILTER to scope the run, use --quick, or explicitly allow full coverage." >&2
  echo "hint: if you intentionally want full hashes/comp coverage, set BENCH_ALLOW_FULL_HASHES_COMP=true." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
LOG_PATH="$OUT_DIR/output.txt"
: > "$LOG_PATH"

# Structured results (set by bench.sh for local runs; unset in direct CI calls)
RESULTS_DIR="${BENCH_RESULTS_DIR:-}"
RESULTS_PATH=""
if [[ -n "$RESULTS_DIR" ]]; then
  mkdir -p "$RESULTS_DIR"
  RESULTS_PATH="$RESULTS_DIR/results.txt"
  {
    echo "date=${BENCH_RUN_DATE}"
    echo "time=${BENCH_RUN_TIME}"
    echo "mode=${BENCH_RUN_MODE}"
    echo "platform=${BENCH_RUN_OS}-${BENCH_RUN_ARCH}"
    echo "commit=${BENCH_RUN_COMMIT}"
    echo ""
  } > "$RESULTS_PATH"
fi

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
echo "Attach raw Criterion: $ATTACH_CRITERION_INPUT"
echo "Dry-run plan: $DRY_RUN_PLAN_INPUT"
echo "Allow full hashes/comp: $ALLOW_FULL_HASHES_COMP_INPUT"
echo "Enforce BLAKE3 gap gate: $ENFORCE_BLAKE3_GAP_GATE_INPUT"
if [[ -n "$PLATFORM_INPUT" ]]; then
  echo "Bench platform: $PLATFORM_INPUT"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

maybe_attach_criterion() {
  if [[ "$ATTACH_CRITERION_INPUT" != "true" ]]; then
    return 0
  fi
  if [[ ! -d target/criterion ]]; then
    echo "note: BENCH_ATTACH_CRITERION=true but target/criterion does not exist" | tee -a "$LOG_PATH"
    return 0
  fi
  local archive_path="$OUT_DIR/criterion.tgz"
  tar -C target -czf "$archive_path" criterion
  echo "Packed raw Criterion artifact: $archive_path" | tee -a "$LOG_PATH"
}

run_blake3_enforced_gates() {
  local failed=0

  if [[ "$ENFORCE_BLAKE3_GAP_GATE_INPUT" == "true" ]]; then
    if ! bash scripts/bench/blake3-gap-gate.sh | tee -a "$LOG_PATH"; then
      failed=1
    fi
  fi

  return "$failed"
}

PLAN_ROWS=()
RAW_FILTERS=()
SELECTED_ALGOS=()
HAS_ALL_SELECTOR="false"

if [[ -n "$ONLY_INPUT" ]]; then
  IFS=',' read -r -a only_values <<< "$ONLY_INPUT"
  for selector in "${only_values[@]:+${only_values[@]}}"; do
    key="$(normalize_selector "$selector")"
    if resolved_algorithms="$(catalog resolve-selector "$selector")"; then
      [[ "$key" == "all" ]] && HAS_ALL_SELECTOR="true"
      IFS=',' read -r -a resolved_values <<< "$resolved_algorithms"
      for algo in "${resolved_values[@]:+${resolved_values[@]}}"; do
        append_unique "$algo" SELECTED_ALGOS
      done
    else
      append_unique "$selector" RAW_FILTERS
    fi
  done

  # When an explicit raw Criterion filter is provided, treat it as the
  # authoritative benchmark matcher and avoid adding a second broad plan row
  # from BENCH_ONLY (e.g. "blake3"), which would run extra surfaces.
  if [[ -z "$FILTER_INPUT" ]]; then
    selected_algorithms_csv="$(IFS=','; echo "${SELECTED_ALGOS[*]}")"
    append_algo_plan_rows "$selected_algorithms_csv"
  fi
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

  if [[ -z "$CRATES_INPUT" && -z "$BENCHES_INPUT" && "${#SELECTED_ALGOS[@]}" -gt 0 ]]; then
    selected_algorithms_csv="$(IFS=','; echo "${SELECTED_ALGOS[*]}")"
    for filter in "${RAW_FILTERS[@]}"; do
      append_algo_plan_rows "$selected_algorithms_csv" "$filter"
    done
    dedupe_plan_rows
  else

    if [[ -n "$CRATES_INPUT" ]]; then
      IFS=',' read -r -a raw_crates <<< "$CRATES_INPUT"
    elif [[ -n "$BENCHES_INPUT" ]]; then
      raw_crates=("workspace")
    elif [[ "${#PLAN_ROWS[@]}" -gt 0 ]]; then
      for row in "${PLAN_ROWS[@]}"; do
        IFS='|' read -r crate _ _ <<< "$row"
        append_unique "$crate" raw_crates
      done
    elif [[ "${#SELECTED_ALGOS[@]}" -gt 0 ]]; then
      selected_algorithms_csv="$(IFS=','; echo "${SELECTED_ALGOS[*]}")"
      IFS=',' read -r -a raw_crates <<< "$(catalog crates-for-algorithms "$selected_algorithms_csv")"
      if [[ "${#raw_crates[@]}" -eq 0 ]]; then
        raw_crates=("checksum" "hashes" "auth" "aead")
      fi
    else
      raw_crates=("checksum" "hashes" "auth" "aead")
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
fi

dedupe_plan_rows

if [[ "$HAS_ALL_SELECTOR" == "true" && -z "$CRATES_INPUT" && -z "$BENCHES_INPUT" ]]; then
  IFS=',' read -r -a required_benches <<< "$(catalog required-benches)"
  for required_bench in "${required_benches[@]}"; do
    found_bench="false"
    for row in "${PLAN_ROWS[@]:+${PLAN_ROWS[@]}}"; do
      IFS='|' read -r _ bench _ <<< "$row"
      if [[ "$bench" == "$required_bench" ]]; then
        found_bench="true"
        break
      fi
    done
    if [[ "$found_bench" == "false" ]]; then
      echo "error: BENCH_ONLY=all did not schedule required bench target '$required_bench'" | tee -a "$LOG_PATH"
      maybe_attach_criterion
      exit 2
    fi
  done
fi

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

if [[ "${#PLAN_ROWS[@]}" -eq 0 && ( -n "$ONLY_INPUT" || -n "$FILTER_INPUT" ) ]]; then
  echo "error: selector inputs produced an empty execution plan; refusing generic fallback." | tee -a "$LOG_PATH"
  echo "hint: check BENCH_ONLY/BENCH_FILTER spelling, or clear selectors if you intend a broad run." | tee -a "$LOG_PATH"
  maybe_attach_criterion
  exit 2
fi

run_bench_cmd() {
  local crate="$1"
  local bench="$2"
  local filter="${3:-}"
  local bench_features
  local cargo_bench
  local -a cmd

  bench_features="$(bench_features_for_target "$bench")"
  cargo_bench="$(bench_binary_for_target "$bench")"
  cmd=(cargo bench --locked --profile bench --features "$bench_features" --bench "$cargo_bench")
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
  if [[ -n "$RESULTS_PATH" ]]; then
    {
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo "bench=$cargo_bench"
      [[ -n "$filter" ]] && echo "filter=$filter"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    } >> "$RESULTS_PATH"
    "${cmd[@]}" 2>&1 | tee -a "$LOG_PATH" "$RESULTS_PATH"
    echo "" >> "$RESULTS_PATH"
  else
    "${cmd[@]}" 2>&1 | tee -a "$LOG_PATH"
  fi
}

if [[ "${#PLAN_ROWS[@]}" -gt 0 ]]; then
  echo "Execution plan (${#PLAN_ROWS[@]} row(s)):" | tee -a "$LOG_PATH"
  for row in "${PLAN_ROWS[@]}"; do
    IFS='|' read -r crate bench filter <<< "$row"
    echo "  - crate=$crate bench=$bench filter=$filter" | tee -a "$LOG_PATH"
  done

  if [[ "$DRY_RUN_PLAN_INPUT" == "true" ]]; then
    echo "Dry-run requested; benchmark commands were not executed." | tee -a "$LOG_PATH"
    maybe_attach_criterion
    exit 0
  fi

  for row in "${PLAN_ROWS[@]}"; do
    IFS='|' read -r crate bench filter <<< "$row"
    run_bench_cmd "$crate" "$bench" "$filter"
  done

  if [[ -n "${RESULTS_PATH:-}" && -f "$RESULTS_PATH" ]]; then
    echo ""
    echo "Results: $RESULTS_PATH"
  fi

  if ! run_blake3_enforced_gates; then
    maybe_attach_criterion
    exit 1
  fi
  maybe_attach_criterion
  exit 0
fi

BENCH_FLAGS=()
# Single-crate layout: no -p flags needed (workspace has only rscrypto).

if [[ -n "$BENCHES_INPUT" ]]; then
  IFS=',' read -r -a benches_values <<< "$BENCHES_INPUT"
  for bench in "${benches_values[@]:+${benches_values[@]}}"; do
    BENCH_FLAGS+=(--bench "$(bench_binary_for_target "$bench")")
  done
else
  IFS=',' read -r -a criterion_binaries <<< "$(catalog criterion-binaries)"
  for bench in "${criterion_binaries[@]}"; do
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
GENERIC_FEATURES="$(bench_features_for_invocation "$BENCHES_INPUT")"
echo "Using features: $GENERIC_FEATURES" | tee -a "$LOG_PATH"

cmd=(cargo bench --locked --profile bench --features "$GENERIC_FEATURES")
if [[ "${#BENCH_FLAGS[@]}" -gt 0 ]]; then
  cmd+=("${BENCH_FLAGS[@]}")
fi
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
if [[ -n "$RESULTS_PATH" ]]; then
  GENERIC_BENCH_LABEL="${BENCHES_INPUT:-all}"
  {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "bench=$GENERIC_BENCH_LABEL"
    [[ -n "$GENERIC_FILTER" ]] && echo "filter=$GENERIC_FILTER"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  } >> "$RESULTS_PATH"
  "${cmd[@]}" 2>&1 | tee -a "$LOG_PATH" "$RESULTS_PATH"
  echo "" >> "$RESULTS_PATH"
else
  "${cmd[@]}" 2>&1 | tee -a "$LOG_PATH"
fi

if [[ -n "${RESULTS_PATH:-}" && -f "$RESULTS_PATH" ]]; then
  echo ""
  echo "Results: $RESULTS_PATH"
fi

if ! run_blake3_enforced_gates; then
  maybe_attach_criterion
  exit 1
fi
maybe_attach_criterion
