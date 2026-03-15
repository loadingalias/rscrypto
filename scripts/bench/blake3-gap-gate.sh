#!/usr/bin/env bash
# BLAKE3 oneshot performance gap gate.
#
# Reads Criterion benchmark output and checks that our implementation stays
# within ratcheted throughput thresholds compared to the official baseline.
set -euo pipefail

# --- defaults -----------------------------------------------------------------

ROOT="target/criterion"
GROUP="blake3/oneshot"
OURS="rscrypto"
OURS_PREFIX=""
RIVAL="official"
MAX_GAP_CASE=""
LABEL=""

# Ratcheted thresholds: close to current baseline so we lock wins while
# preserving enough headroom for normal runner variance.
declare -A DEFAULT_THRESHOLDS=(
  [256]=4.8
  [1024]=6.8
  [4096]=7.8
  [16384]=13.0
  [65536]=4.5
)

# --- arg parsing --------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)         ROOT="$2";         shift 2 ;;
    --group)        GROUP="$2";        shift 2 ;;
    --ours)         OURS="$2";         shift 2 ;;
    --ours-prefix)  OURS_PREFIX="$2";  shift 2 ;;
    --rival)        RIVAL="$2";        shift 2 ;;
    --max-gap-case) MAX_GAP_CASE="$2"; shift 2 ;;
    --label)        LABEL="$2";        shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "$ROOT" ]]; then
  echo "error: missing criterion root: $ROOT" >&2
  exit 2
fi

# --- build thresholds ---------------------------------------------------------

declare -A THRESHOLDS
for k in "${!DEFAULT_THRESHOLDS[@]}"; do
  THRESHOLDS[$k]="${DEFAULT_THRESHOLDS[$k]}"
done

if [[ -n "$MAX_GAP_CASE" ]]; then
  IFS=',' read -ra TOKENS <<< "$MAX_GAP_CASE"
  for token in "${TOKENS[@]}"; do
    token="${token// /}"
    [[ -n "$token" ]] || continue
    case_key="${token%%=*}"
    pct_val="${token#*=}"
    if [[ -z "$case_key" || -z "$pct_val" ]]; then
      echo "error: invalid threshold token '$token' (expected SIZE=PCT)" >&2
      exit 2
    fi
    THRESHOLDS[$case_key]="$pct_val"
  done
fi

# --- extract throughput from Criterion JSON -----------------------------------

# For a given group_id, function_id, and value_str, find the benchmark.json +
# estimates.json pair and compute units_per_sec.
#
# Criterion layout: <root>/<group>/<function>/<value>/{new,base}/benchmark.json
# We prefer "new" over "base".

get_throughput() {
  local func="$1" value="$2"
  local search_dir="$ROOT/$GROUP/$func/$value"

  if [[ ! -d "$search_dir" ]]; then
    echo ""
    return
  fi

  local bench_json="" estimates_json=""

  # prefer new/ over base/
  if [[ -f "$search_dir/new/benchmark.json" && -f "$search_dir/new/estimates.json" ]]; then
    bench_json="$search_dir/new/benchmark.json"
    estimates_json="$search_dir/new/estimates.json"
  elif [[ -f "$search_dir/base/benchmark.json" && -f "$search_dir/base/estimates.json" ]]; then
    bench_json="$search_dir/base/benchmark.json"
    estimates_json="$search_dir/base/estimates.json"
  elif [[ -f "$search_dir/benchmark.json" && -f "$search_dir/estimates.json" ]]; then
    bench_json="$search_dir/benchmark.json"
    estimates_json="$search_dir/estimates.json"
  else
    echo ""
    return
  fi

  # Extract throughput units and mean_ns, compute units_per_sec
  jq -r --argjson est "$(cat "$estimates_json")" '
    .throughput as $tp |
    ($tp.Bytes // $tp.Elements // null) as $units |
    ($est.mean.point_estimate // null) as $mean_ns |
    if $units == null or $mean_ns == null or $mean_ns <= 0 then
      ""
    else
      (($units * 1000000000) / $mean_ns | tostring)
    end
  ' "$bench_json"
}

# When --ours-prefix is set, find the best matching function_id
find_best_ours() {
  local value="$1"
  local group_dir="$ROOT/$GROUP"
  local best_func="" best_rate=0

  if [[ -n "$OURS_PREFIX" ]]; then
    for func_dir in "$group_dir"/"$OURS_PREFIX"*/; do
      [[ -d "$func_dir" ]] || continue
      local func
      func="$(basename "$func_dir")"
      local rate
      rate="$(get_throughput "$func" "$value")"
      if [[ -n "$rate" ]]; then
        if awk "BEGIN { exit !($rate > $best_rate) }"; then
          best_rate="$rate"
          best_func="$func"
        fi
      fi
    done
  fi

  if [[ -z "$best_func" ]]; then
    local rate
    rate="$(get_throughput "$OURS" "$value")"
    if [[ -n "$rate" ]]; then
      echo "$OURS $rate"
      return
    fi
    echo ""
    return
  fi

  echo "$best_func $best_rate"
}

# --- run gate -----------------------------------------------------------------

GATE_LABEL="${LABEL:-$GROUP}"
FAILURES=()

echo "BLAKE3 gap gate ($GATE_LABEL):"

# Sort thresholds by numeric key
SORTED_CASES=($(for k in "${!THRESHOLDS[@]}"; do echo "$k"; done | sort -n))

for case_size in "${SORTED_CASES[@]}"; do
  max_gap="${THRESHOLDS[$case_size]}"

  ours_result="$(find_best_ours "$case_size")"
  if [[ -z "$ours_result" ]]; then
    FAILURES+=("missing data for size=$case_size (need ${OURS_PREFIX:-$OURS} and $RIVAL)")
    continue
  fi
  ours_func="${ours_result%% *}"
  ours_rate="${ours_result##* }"

  rival_rate="$(get_throughput "$RIVAL" "$case_size")"
  if [[ -z "$rival_rate" ]]; then
    FAILURES+=("missing data for size=$case_size (need ${OURS_PREFIX:-$OURS} and $RIVAL)")
    continue
  fi

  # Compute gap percentage: (rival/ours - 1) * 100
  need_pct="$(awk "BEGIN { printf \"%.2f\", ($rival_rate / $ours_rate - 1.0) * 100.0 }")"

  echo "  size=$case_size: ours=$ours_func need=+${need_pct}% (limit +$(printf '%.2f' "$max_gap")%)"

  if awk "BEGIN { exit !($need_pct > $max_gap) }"; then
    FAILURES+=("size=$case_size gap too large: need +${need_pct}% vs allowed +$(printf '%.2f' "$max_gap")%")
  fi
done

if [[ ${#FAILURES[@]} -gt 0 ]]; then
  echo "" >&2
  echo "Gate failed:" >&2
  for f in "${FAILURES[@]}"; do
    echo "  - $f" >&2
  done
  exit 1
fi

echo "Gate passed."
