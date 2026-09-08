#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# shellcheck source=../lib/fuzz-packages.sh
source "$SCRIPT_DIR/../lib/fuzz-packages.sh"

RUSTUP_TOOLCHAIN=$("$SCRIPT_DIR/../lib/toolchain.sh" --nightly)
export RUSTUP_TOOLCHAIN
export CARGO_RAIL_CACHE=off

# Configuration (can be overridden via environment)
DURATION_SECS=${RSCRYPTO_FUZZ_DURATION_SECS:-60}
TIMEOUT=${RSCRYPTO_FUZZ_TIMEOUT_SECS:-30}
RSS_LIMIT=${RSCRYPTO_FUZZ_RSS_LIMIT_MB:-2048}
MAX_LEN=${RSCRYPTO_FUZZ_MAX_LEN:-65536}
JOBS=${RSCRYPTO_FUZZ_JOBS:-1}

default_target_concurrency() {
  local processor_count=""

  if command -v getconf >/dev/null 2>&1; then
    processor_count=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)
  fi
  if [[ ! "$processor_count" =~ ^[1-9][0-9]*$ ]] && command -v sysctl >/dev/null 2>&1; then
    processor_count=$(sysctl -n hw.logicalcpu 2>/dev/null || true)
  fi
  if [[ ! "$processor_count" =~ ^[1-9][0-9]*$ ]]; then
    processor_count=${NUMBER_OF_PROCESSORS:-}
  fi

  if [[ "$processor_count" =~ ^[1-9][0-9]*$ ]] && ((processor_count >= 4)); then
    printf '2'
  else
    printf '1'
  fi
}

TARGET_CONCURRENCY=${RSCRYPTO_FUZZ_TARGET_CONCURRENCY:-$(default_target_concurrency)}

show_help() {
  echo "Fuzz testing for rscrypto"
  echo ""
  echo "Usage:"
  echo "  $0                        Build scoped packages, run full harness"
  echo "  $0 --all                  Build and run full + scoped packages"
  echo "  $0 --full                 Run full harness targets (${DURATION_SECS}s each)"
  echo "  $0 --scoped               Run scoped targets (${DURATION_SECS}s each)"
  echo "  $0 --targets A,B          Filter the chosen package scope (default: full) by exact target names"
  echo "  $0 <target>               Run specific target"
  echo "  $0 --build [--full|--scoped|--all]  Build selected fuzz packages"
  echo "  $0 --list                 List available targets by package"
  echo "  $0 --clean                Clean build artifacts; preserve corpora"
  echo ""
  echo "Environment variables:"
  echo "  RSCRYPTO_FUZZ_DURATION_SECS  Duration per target (default: 60)"
  echo "  RSCRYPTO_FUZZ_TIMEOUT_SECS   Timeout per test case (default: 30)"
  echo "  RSCRYPTO_FUZZ_RSS_LIMIT_MB   Memory limit in MB (default: 2048)"
  echo "  RSCRYPTO_FUZZ_MAX_LEN        Max input length (default: 65536)"
  echo "  RSCRYPTO_FUZZ_JOBS           LibFuzzer workers per target (default: 1)"
  echo "  RSCRYPTO_FUZZ_TARGET_CONCURRENCY  Independent targets to run concurrently (default: 1 below 4 CPUs, otherwise 2)"
  echo "  RSCRYPTO_FUZZ_TARGET_DIR     Shared cargo target dir (default: fuzz/target)"
}

check_requirements() {
  if ! cargo --version &>/dev/null; then
    echo "Error: Rust toolchain not found"
    exit 1
  fi
  if ! cargo fuzz --version &>/dev/null; then
    echo "Error: cargo-fuzz not found"
    echo "Install with: cargo install cargo-fuzz"
    exit 1
  fi
  if [ ! -d "$FUZZ_ROOT" ]; then
    echo "No fuzz/ directory found at repo root" >&2
    exit 1
  fi

  discover_fuzz_packages
  if [ ${#FUZZ_ALL_PACKAGES[@]} -eq 0 ]; then
    echo "No cargo-fuzz packages found" >&2
    exit 1
  fi
}

list_targets() {
  local package_dir targets

  echo "Available fuzz targets:"
  echo ""
  for package_dir in "${FUZZ_ALL_PACKAGES[@]}"; do
    echo "$(fuzz_package_label "$package_dir"):"
    targets=$(fuzz_list_targets "$package_dir") || return 2
    while IFS= read -r target; do
      [ -z "$target" ] && continue
      echo "  $target"
    done <<<"$targets"
    echo ""
  done
}

build_packages() {
  local scope=$1
  local package_dir

  fuzz_select_packages "$scope"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Building fuzz targets ($scope)..."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  for package_dir in "${SELECTED_FUZZ_PACKAGES[@]}"; do
    echo "  Building: $(fuzz_package_label "$package_dir")"
    fuzz_in_package "$package_dir" build --target "$(fuzz_host_target)"
  done

  echo "Selected fuzz packages built successfully"
}

clean_artifacts() {
  echo "Cleaning fuzz artifacts..."

  local package_dir
  for package_dir in "${FUZZ_ALL_PACKAGES[@]}"; do
    fuzz_in_package "$package_dir" clean 2>/dev/null || true
    rm -rf "$package_dir/artifacts" "$package_dir/coverage"
  done
  rm -rf "$FUZZ_SHARED_TARGET_DIR"

  echo "Fuzz artifacts cleaned"
}

# Map a fuzz target name to its dictionary file, if any.
#
# libFuzzer accepts `-dict=<file>` to seed mutation with high-value tokens.
# Targets opt in by name; absence of a mapping means no dictionary, which is
# the correct behaviour for purely numeric or property-only targets.
fuzz_dictionary_for_target() {
  local target="$1"
  local dict=""

  case "$target" in
    aead_aegis256|aead_aes256gcm|aead_aes256gcmsiv|aead_aes_siv_cmac256|aead_ascon128|aead_chacha20poly1305|aead_xchacha20poly1305)
      dict="$REPO_ROOT/fuzz/dictionaries/aead_boundary.dict"
      ;;
    auth_phc)
      dict="$REPO_ROOT/fuzz/dictionaries/phc.dict"
      ;;
    hex_parse)
      dict="$REPO_ROOT/fuzz/dictionaries/hex_parse.dict"
      ;;
  esac

  if [ -n "$dict" ] && [ -f "$dict" ]; then
    printf '%s' "$dict"
  fi
}

run_target_in_package() {
  local package_dir="$1"
  local target="$2"
  local duration="$3"

  local label
  label="$(fuzz_package_label "$package_dir")"

  echo "  Running: ${label}/${target} (${duration}s)"

  mkdir -p "$package_dir/artifacts/$target"

  local fuzz_args=(
    "-max_total_time=$duration"
    "-timeout=$TIMEOUT"
    "-rss_limit_mb=$RSS_LIMIT"
    "-max_len=$MAX_LEN"
    "-artifact_prefix=$package_dir/artifacts/$target/"
  )

  local dict
  dict="$(fuzz_dictionary_for_target "$target")"
  if [ -n "$dict" ]; then
    fuzz_args+=( "-dict=$dict" )
  fi

  local fuzz_exit=0
  fuzz_in_package "$package_dir" run "$target" \
      --jobs="$JOBS" \
      --target "$(fuzz_host_target)" \
      -- "${fuzz_args[@]}" 2>&1 | { grep -v "^INFO:" || true; } || fuzz_exit=$?

  if [ "$fuzz_exit" -eq 0 ]; then
    echo "    Completed: ${label}/${target}"
    return 0
  fi

  echo "    Failed or found crash: ${label}/${target}"
  return 1
}

RUN_PACKAGE_DIRS=()
RUN_TARGETS=()

run_batch() {
  local label="$1"
  local duration="$2"
  local failed=0
  local crashed=""

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Fuzz Testing ($label)"
  echo "Duration: ${duration}s per target"
  echo "Concurrent targets: $TARGET_CONCURRENCY"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  local total=${#RUN_TARGETS[@]}
  if [ "$total" -eq 0 ]; then
    echo "No fuzz targets selected" >&2
    return 1
  fi

  local run_log_dir
  run_log_dir=$(mktemp -d)
  local batch_start=0
  local batch_end
  local index
  while [ "$batch_start" -lt "$total" ]; do
    batch_end=$((batch_start + TARGET_CONCURRENCY))
    if [ "$batch_end" -gt "$total" ]; then
      batch_end=$total
    fi

    local pids=()
    local logs=()
    for ((index = batch_start; index < batch_end; index++)); do
      local log_path="$run_log_dir/$index.log"
      run_target_in_package "${RUN_PACKAGE_DIRS[$index]}" "${RUN_TARGETS[$index]}" "$duration" >"$log_path" 2>&1 &
      pids+=("$!")
      logs+=("$log_path")
    done

    local batch_index
    local fuzz_status
    for batch_index in "${!pids[@]}"; do
      index=$((batch_start + batch_index))
      fuzz_status=0
      if wait "${pids[$batch_index]}"; then
        fuzz_status=0
      else
        fuzz_status=$?
      fi
      cat "${logs[$batch_index]}"
      rm -f "${logs[$batch_index]}"
      if [ "$fuzz_status" -ne 0 ]; then
        failed=$((failed + 1))
        crashed="${crashed}  $(fuzz_package_label "${RUN_PACKAGE_DIRS[$index]}")/${RUN_TARGETS[$index]}\n"
      fi
    done

    batch_start=$batch_end
  done
  rm -rf "$run_log_dir"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Summary: $total targets, $failed failed"
  if [ $failed -gt 0 ]; then
    echo -e "Crashed:\n$crashed"
    return 1
  else
    echo "All fuzz targets passed"
  fi
}

run_scope() {
  local scope="$1"
  local duration="$2"
  local filter="${3:-}"
  local package_dir target targets seen=, matched=,
  local -a requested=()
  if [[ -n "$filter" ]]; then
    [[ "$filter" != *, ]] || { echo "Invalid empty fuzz target" >&2; return 2; }
    IFS=',' read -r -a requested <<<"$filter"
    for target in "${requested[@]}"; do
      [[ "$target" =~ ^[a-z0-9_]+$ && "$seen" != *",$target,"* ]] || {
        echo "Invalid or duplicate fuzz target: ${target:-<empty>}" >&2
        return 2
      }
      seen+="$target,"
    done
  fi

  fuzz_select_packages "$scope"
  RUN_PACKAGE_DIRS=()
  RUN_TARGETS=()
  for package_dir in "${SELECTED_FUZZ_PACKAGES[@]}"; do
    targets=$(fuzz_list_targets "$package_dir") || return 2
    while IFS= read -r target; do
      [[ -n "$target" ]] || continue
      [[ -z "$filter" || "$seen" == *",$target,"* ]] || continue
      RUN_PACKAGE_DIRS+=("$package_dir")
      RUN_TARGETS+=("$target")
      matched+="$target,"
    done <<<"$targets"
  done
  for target in "${requested[@]:+${requested[@]}}"; do
    [[ "$matched" == *",$target,"* ]] || {
      echo "Unknown fuzz target in $scope scope: $target" >&2
      return 1
    }
  done
  run_batch "$scope" "$duration"
}

ACTION="default"
PACKAGE_SCOPE="full"
TARGET=""
TARGETS_CSV=""
TARGET_DURATION="$DURATION_SECS"
scope_selected=false
duration_selected=false

select_action() {
  [[ "$ACTION" == default || "$ACTION" == "$1" ]] || {
    echo "Conflicting fuzz actions: $ACTION and $1" >&2
    exit 2
  }
  ACTION=$1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) show_help; exit 0 ;;
    --list|--build|--clean) select_action "${1#--}" ;;
    --all|--full|--scoped)
      [[ "$scope_selected" == false || "$PACKAGE_SCOPE" == "${1#--}" ]] || {
        echo "Conflicting fuzz package scopes" >&2
        exit 2
      }
      PACKAGE_SCOPE=${1#--}
      scope_selected=true
      ;;
    --targets)
      select_action selected
      shift
      [[ $# -gt 0 && -n "$1" && "$1" != -* && -z "$TARGETS_CSV" ]] || {
        echo "--targets requires one comma-separated target set" >&2
        exit 2
      }
      TARGETS_CSV=$1
      ;;
    -*) echo "Unknown fuzz option: $1" >&2; exit 2 ;;
    *)
      if [[ -z "$TARGET" ]]; then
        TARGET=$1
      elif [[ "$duration_selected" == false ]]; then
        TARGET_DURATION=$1
        duration_selected=true
      else
        echo "Unexpected fuzz argument: $1" >&2
        exit 2
      fi
      ;;
  esac
  shift
done

if [[ -n "$TARGET" && "$ACTION" != default ]]; then
  echo "Positional targets cannot be combined with --$ACTION" >&2
  exit 2
fi
if [[ "$ACTION" == default && "$scope_selected" == true ]]; then
  ACTION=run
fi
[[ "$TARGET_DURATION" =~ ^[1-9][0-9]*$ ]] || {
  echo "Fuzz duration must be a positive integer" >&2
  exit 2
}

case "$TARGET_CONCURRENCY" in
  ''|*[!0-9]*|0)
    echo "RSCRYPTO_FUZZ_TARGET_CONCURRENCY must be a positive integer" >&2
    exit 2
    ;;
esac

check_requirements

case "$ACTION" in
  list)
    list_targets
    ;;
  build)
    build_packages "$PACKAGE_SCOPE"
    ;;
  clean)
    clean_artifacts
    ;;
  selected)
    run_scope "$PACKAGE_SCOPE" "$TARGET_DURATION" "$TARGETS_CSV"
    ;;
  run)
    run_scope "$PACKAGE_SCOPE" "$TARGET_DURATION" "$TARGET"
    ;;
  default)
    if [[ -z "$TARGET" ]]; then
      build_packages scoped
    fi
    run_scope "$PACKAGE_SCOPE" "$TARGET_DURATION" "$TARGET"
    ;;
esac
