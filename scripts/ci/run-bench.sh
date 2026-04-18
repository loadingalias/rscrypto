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
  "sha512-256"
  "sha3-224"
  "sha3-256"
  "sha3-384"
  "sha3-512"
  "shake128"
  "shake256"
  "blake2"
  "blake3"
  "xxh3"
  "rapidhash"
  "ascon-hash256"
  "ascon-xof128"
)

DEFAULT_AUTH_ALGOS=(
  "hmac-sha256"
  "hmac-sha384"
  "hmac-sha512"
  "hkdf-sha256"
  "hkdf-sha384"
  "pbkdf2-sha256"
  "pbkdf2-sha512"
  "ed25519"
  "x25519"
)

DEFAULT_AEAD_ALGOS=(
  "xchacha20-poly1305"
  "chacha20-poly1305"
  "aes-256-gcm-siv"
  "aes-256-gcm"
  "aegis-256"
)

ALL_KNOWN_ALGOS=("${DEFAULT_CHECKSUM_ALGOS[@]}" "${DEFAULT_HASH_ALGOS[@]}" "${DEFAULT_AUTH_ALGOS[@]}" "${DEFAULT_AEAD_ALGOS[@]}")

checksum_filter_token() {
  local algo="${1:-}"
  case "$algo" in
    crc16-ccitt) echo "crc16-ccitt" ;;
    crc16-ibm) echo "crc16-ibm" ;;
    crc24-openpgp) echo "crc24-openpgp" ;;
    crc32-ieee) echo "^crc32/" ;;
    crc32c) echo "crc32c" ;;
    crc64-xz) echo "crc64-xz" ;;
    crc64-nvme) echo "crc64-nvme" ;;
    *) echo "$algo" ;;
  esac
}

hash_filter_token() {
  local algo="${1:-}"
  case "$algo" in
    sha3-224) echo "sha3-224" ;;
    sha3-256) echo "sha3-256" ;;
    sha3-384) echo "sha3-384" ;;
    sha3-512) echo "sha3-512" ;;
    sha512) echo "^sha512/" ;;
    sha512-256) echo "sha512-256" ;;

    ascon-hash256) echo "ascon-hash256" ;;
    ascon-xof128) echo "ascon-xof128" ;;
    xxh3) echo "xxh3" ;;
    blake2) echo "^blake2/(rscrypto|rustcrypto|keyed|streaming)/" ;;
    blake3) echo "blake3" ;;
    *) echo "$algo" ;;
  esac
}

auth_filter_token() {
  local algo="${1:-}"
  case "$algo" in
    hmac-sha256) echo "^hmac-sha256" ;;
    hmac-sha384) echo "hmac-sha384" ;;
    hmac-sha512) echo "hmac-sha512" ;;
    hkdf-sha256) echo "^hkdf-sha256" ;;
    hkdf-sha384) echo "hkdf-sha384" ;;
    pbkdf2-sha256) echo "^pbkdf2-sha256/" ;;
    pbkdf2-sha512) echo "^pbkdf2-sha512/" ;;
    ed25519) echo "ed25519" ;;
    x25519) echo "x25519" ;;
    *) echo "$algo" ;;
  esac
}

aead_filter_token() {
  local algo="${1:-}"
  case "$algo" in
    xchacha20-poly1305) echo "xchacha20-poly1305" ;;
    chacha20-poly1305) echo "^chacha20-poly1305/" ;;
    aes-256-gcm-siv) echo "aes-256-gcm-siv" ;;
    aes-256-gcm) echo "^aes-256-gcm/" ;;
    aegis-256) echo "aegis-256" ;;
    *) echo "$algo" ;;
  esac
}

default_benches_for_crate() {
  local crate="${1:-}"
  case "$crate" in
    checksum) echo "crc" ;;
    hashes) echo "sha2,sha3,ascon,xxh3,rapidhash,blake2,blake3" ;;
    auth) echo "auth" ;;
    aead) echo "aead" ;;
    *) echo "" ;;
  esac
}

merge_csvs() {
  local -a parts=()
  local -a merged=()
  local csv
  local token

  for csv in "$@"; do
    [[ -z "$csv" ]] && continue
    IFS=',' read -r -a parts <<< "$csv"
    for token in "${parts[@]:+${parts[@]}}"; do
      token="$(echo "$token" | xargs)"
      [[ -z "$token" ]] && continue
      append_unique "$token" merged
    done
  done

  if [[ "${#merged[@]}" -eq 0 ]]; then
    echo ""
  else
    (IFS=','; echo "${merged[*]}")
  fi
}

bench_features_for_target() {
  local bench="${1:-}"
  case "$bench" in
    crc) echo "parallel,checksums" ;;
    sha2) echo "parallel,sha2" ;;
    sha3) echo "parallel,sha3" ;;
    ascon) echo "parallel,ascon-hash" ;;
    xxh3) echo "parallel,xxh3" ;;
    rapidhash) echo "parallel,rapidhash" ;;
    blake2) echo "parallel,blake2b,blake2s" ;;
    blake3) echo "parallel,blake3" ;;
    auth) echo "parallel,hmac,hkdf,pbkdf2,ed25519,x25519" ;;
    aead) echo "parallel,aes-gcm,aes-gcm-siv,chacha20poly1305,xchacha20poly1305,aegis256" ;;
    *) echo "parallel" ;;
  esac
}

bench_features_for_invocation() {
  local benches_csv="${1:-}"
  local features=""
  local -a benches=()
  local bench

  if [[ -z "$benches_csv" ]]; then
    echo "parallel,full"
    return 0
  fi

  IFS=',' read -r -a benches <<< "$benches_csv"
  for bench in "${benches[@]:+${benches[@]}}"; do
    features="$(merge_csvs "$features" "$(bench_features_for_target "$bench")")"
  done

  if [[ -z "$features" ]]; then
    echo "parallel"
  else
    echo "$features"
  fi
}

bench_target_for_hash_algo() {
  local algo="${1:-}"
  case "$algo" in
    sha224|sha256|sha384|sha512|sha512-256) echo "sha2" ;;
    sha3-224|sha3-256|sha3-384|sha3-512|shake128|shake256) echo "sha3" ;;
    ascon-hash256|ascon-xof128) echo "ascon" ;;
    xxh3) echo "xxh3" ;;
    rapidhash) echo "rapidhash" ;;
    blake2) echo "blake2" ;;
    blake3) echo "blake3" ;;
    *) return 1 ;;
  esac
}

append_algo_plan_row() {
  local algo="${1:-}"
  local raw_filter="${2:-}"
  local bench=""
  local crate=""
  local token=""

  if array_contains "$algo" "${DEFAULT_CHECKSUM_ALGOS[@]}"; then
    crate="checksum"
    bench="crc"
    token="${raw_filter:-$(checksum_filter_token "$algo")}"
    PLAN_ROWS+=("$crate|$bench|$token")
    return 0
  fi

  if array_contains "$algo" "${DEFAULT_HASH_ALGOS[@]}"; then
    if ! bench="$(bench_target_for_hash_algo "$algo")"; then
      return 0
    fi
    crate="hashes"
    token="${raw_filter:-$(hash_filter_token "$algo")}"
    PLAN_ROWS+=("$crate|$bench|$token")
    return 0
  fi

  if array_contains "$algo" "${DEFAULT_AUTH_ALGOS[@]}"; then
    crate="auth"
    bench="auth"
    token="${raw_filter:-$(auth_filter_token "$algo")}"
    PLAN_ROWS+=("$crate|$bench|$token")
    return 0
  fi

  if array_contains "$algo" "${DEFAULT_AEAD_ALGOS[@]}"; then
    crate="aead"
    bench="aead"
    token="${raw_filter:-$(aead_filter_token "$algo")}"
    PLAN_ROWS+=("$crate|$bench|$token")
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

# Expand bench shorthand and legacy aliases onto real bench targets.
expand_bench_shorthand() {
  local raw="$1"
  [[ -n "$raw" ]] || return 0
  local -a expanded=()
  local token
  IFS=',' read -r -a tokens <<< "$raw"
  for token in "${tokens[@]}"; do
    case "$token" in
      comp) expanded+=("crc" "sha2" "sha3" "ascon" "auth" "aead" "xxh3" "rapidhash" "blake3") ;;
      kernels) expanded+=("blake3") ;;
      checksum_comp|checksum_kernels) expanded+=("crc") ;;
      hashes_comp) expanded+=("sha2" "sha3" "ascon" "xxh3" "rapidhash" "blake3") ;;
      auth_comp) expanded+=("auth") ;;
      aead_comp) expanded+=("aead") ;;
      hashes_kernels) expanded+=("blake3") ;;
      *)       expanded+=("$token") ;;
    esac
  done
  (IFS=','; echo "${expanded[*]}")
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
ALLOW_FULL_HASHES_COMP_INPUT="$(to_bool "${BENCH_ALLOW_FULL_HASHES_COMP:-false}")"

ENFORCE_BLAKE3_GAP_GATE_INPUT="$(to_bool "${BENCH_ENFORCE_BLAKE3_GAP_GATE:-false}")"
ENFORCE_BLAKE3_KERNEL_GATE_INPUT="$(to_bool "${BENCH_ENFORCE_BLAKE3_KERNEL_GATE:-false}")"
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
  rm -rf target/criterion || true
fi

if [[ "$ENFORCE_BLAKE3_GAP_GATE_INPUT" == "true" && "$QUICK_INPUT" == "true" ]]; then
  echo "error: BENCH_ENFORCE_BLAKE3_GAP_GATE=true requires BENCH_QUICK=false" >&2
  exit 2
fi

if [[ "$ENFORCE_BLAKE3_KERNEL_GATE_INPUT" == "true" && "$QUICK_INPUT" == "true" ]]; then
  echo "error: BENCH_ENFORCE_BLAKE3_KERNEL_GATE=true requires BENCH_QUICK=false" >&2
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
  for bench in sha2 sha3 ascon xxh3 rapidhash; do
    if csv_has_token "$BENCHES_INPUT" "$bench"; then
      targets_comp="true"
      break
    fi
  done
fi
if [[ "$ALLOW_FULL_HASHES_COMP_INPUT" != "true" \
  && "$targets_hashes" == "true" \
  && "$targets_comp" == "true" \
  && -z "$ONLY_INPUT" \
  && -z "$FILTER_INPUT" ]]; then
  echo "error: refusing unscoped hashes/comp run (expensive and often timeout-prone on CI lanes)." >&2
  echo "hint: set BENCH_ONLY and/or BENCH_FILTER to scope the run, or explicitly allow full coverage." >&2
  echo "hint: if you intentionally want full hashes/comp coverage, set BENCH_ALLOW_FULL_HASHES_COMP=true." >&2
  exit 2
fi

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
echo "Allow full hashes/comp: $ALLOW_FULL_HASHES_COMP_INPUT"
echo "Enforce BLAKE3 gap gate: $ENFORCE_BLAKE3_GAP_GATE_INPUT"
echo "Enforce BLAKE3 kernel gate: $ENFORCE_BLAKE3_KERNEL_GATE_INPUT"
if [[ -n "$PLATFORM_INPUT" ]]; then
  echo "Bench platform: $PLATFORM_INPUT"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

platform_is_supported_kernel_gate_lane() {
  case "$1" in
    intel-icl|intel-spr|amd-zen4|amd-zen5|graviton3|graviton4|ibm-s390x|ibm-power10) return 0 ;;
    *) return 1 ;;
  esac
}

blake3_kernel_gate_thresholds() {
  local platform="$1"
  case "$platform" in
    intel-spr) echo "256=8.0,1024=7.0,4096=6.0,16384=6.0,65536=6.0" ;;
    intel-icl) echo "256=9.0,1024=8.0,4096=7.0,16384=7.0,65536=7.0" ;;
    amd-zen5) echo "256=9.0,1024=8.0,4096=7.0,16384=7.0,65536=7.0" ;;
    amd-zen4) echo "256=9.0,1024=8.0,4096=7.0,16384=7.0,65536=7.0" ;;
    graviton3) echo "256=7.0,1024=7.0,4096=12.0,16384=12.0,65536=8.0" ;;
    graviton4) echo "256=7.0,1024=7.0,4096=12.0,16384=12.0,65536=8.0" ;;
    ibm-s390x) echo "256=12.0,1024=10.0,4096=10.0,16384=10.0,65536=10.0" ;;
    ibm-power10) echo "256=12.0,1024=10.0,4096=10.0,16384=10.0,65536=10.0" ;;
    *) echo "256=10.0,1024=9.0,4096=8.0,16384=8.0,65536=8.0" ;;
  esac
}

blake3_kernel_gate_prefix() {
  local platform="$1"
  case "$platform" in
    intel-icl|intel-spr|amd-zen4|amd-zen5) echo "rscrypto/x86_64/" ;;
    graviton3|graviton4) echo "rscrypto/aarch64/" ;;
    ibm-s390x) echo "rscrypto/s390x/" ;;
    ibm-power10) echo "rscrypto/power" ;;
    rise-riscv) echo "rscrypto/riscv64/" ;;
    *) echo "rscrypto/" ;;
  esac
}

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

  if [[ "$ENFORCE_BLAKE3_KERNEL_GATE_INPUT" == "true" ]]; then
    if platform_is_supported_kernel_gate_lane "$PLATFORM_INPUT"; then
      local local_thresholds
      local ours_prefix
      local -a cmd
      local_thresholds="$(blake3_kernel_gate_thresholds "$PLATFORM_INPUT")"
      ours_prefix="$(blake3_kernel_gate_prefix "$PLATFORM_INPUT")"
      echo "" | tee -a "$LOG_PATH"
      echo "Running kernel diagnostics for BLAKE3 gate..." | tee -a "$LOG_PATH"
      cmd=(cargo bench --profile bench --features "$(bench_features_for_target blake3)" --bench blake3 -- kernel-ab)
      if [[ "${#CRITERION_ARGS[@]}" -gt 0 ]]; then
        cmd+=("${CRITERION_ARGS[@]}")
      fi
      echo "Running: RSCRYPTO_BLAKE3_BENCH_DIAGNOSTICS=1 ${cmd[*]}" | tee -a "$LOG_PATH"
      if ! RSCRYPTO_BLAKE3_BENCH_DIAGNOSTICS=1 "${cmd[@]}" 2>&1 | tee -a "$LOG_PATH"; then
        failed=1
      fi
      if ! bash scripts/bench/blake3-gap-gate.sh \
        --group blake3/kernel-ab \
        --ours-prefix "$ours_prefix" \
        --rival official \
        --max-gap-case "$local_thresholds" \
        --label "blake3/kernel-ab ($PLATFORM_INPUT)" | tee -a "$LOG_PATH"; then
        failed=1
      fi
    else
      echo "Skipping BLAKE3 kernel gate on unsupported lane '$PLATFORM_INPUT'." | tee -a "$LOG_PATH"
    fi
  fi

  return "$failed"
}

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
      auth)
        for algo in "${DEFAULT_AUTH_ALGOS[@]}"; do append_unique "$algo" SELECTED_ALGOS; done
        ;;
      aead)
        for algo in "${DEFAULT_AEAD_ALGOS[@]}"; do append_unique "$algo" SELECTED_ALGOS; done
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

  # When an explicit raw Criterion filter is provided, treat it as the
  # authoritative benchmark matcher and avoid adding a second broad plan row
  # from BENCH_ONLY (e.g. "blake3"), which would run extra surfaces.
  if [[ -z "$FILTER_INPUT" ]]; then
    for algo in "${SELECTED_ALGOS[@]:+${SELECTED_ALGOS[@]}}"; do
      append_algo_plan_row "$algo"
    done
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
    for filter in "${RAW_FILTERS[@]}"; do
      for algo in "${SELECTED_ALGOS[@]:+${SELECTED_ALGOS[@]}}"; do
        append_algo_plan_row "$algo" "$filter"
      done
    done
    dedupe_plan_rows
  else

    if [[ -n "$CRATES_INPUT" ]]; then
      IFS=',' read -r -a raw_crates <<< "$CRATES_INPUT"
    elif [[ "${#PLAN_ROWS[@]}" -gt 0 ]]; then
      for row in "${PLAN_ROWS[@]}"; do
        IFS='|' read -r crate _ _ <<< "$row"
        append_unique "$crate" raw_crates
      done
    elif [[ "${#SELECTED_ALGOS[@]}" -gt 0 ]]; then
      for algo in "${SELECTED_ALGOS[@]:+${SELECTED_ALGOS[@]}}"; do
        if array_contains "$algo" "${DEFAULT_CHECKSUM_ALGOS[@]}"; then
          append_unique "checksum" raw_crates
        elif array_contains "$algo" "${DEFAULT_HASH_ALGOS[@]}"; then
          append_unique "hashes" raw_crates
        elif array_contains "$algo" "${DEFAULT_AUTH_ALGOS[@]}"; then
          append_unique "auth" raw_crates
        elif array_contains "$algo" "${DEFAULT_AEAD_ALGOS[@]}"; then
          append_unique "aead" raw_crates
        fi
      done
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
  local -a cmd

  bench_features="$(bench_features_for_target "$bench")"
  cmd=(cargo bench --profile bench --features "$bench_features" --bench "$bench")
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
      echo "bench=$bench"
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

cmd=(cargo bench --profile bench --features "$GENERIC_FEATURES" "${BENCH_FLAGS[@]}")
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
