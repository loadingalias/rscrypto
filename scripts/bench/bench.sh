#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/bench/bench.sh [selector ...] [key=value ...]

Selectors:
  aead
  auth
  rsa
  blake2
  blake3
  crc32c
  checksum
  hashes
  mlkem

Key=Value Overrides:
  crates=<csv> | crate=<csv>
  benches=<csv> | bench=<csv>
  only=<csv>
  filter=<csv>
  quick=true|false
  warmup_ms=<int>
  measure_ms=<int>
  sample_size=<int>
  profile_time_secs=<num>
  output_dir=<path>
  clean=true|false
USAGE
}

append_csv() {
  local current="${1:-}"
  local token="${2:-}"
  token="$(echo "$token" | xargs)"
  if [[ -z "$token" ]]; then
    echo "$current"
    return 0
  fi
  if [[ -z "$current" ]]; then
    echo "$token"
  else
    echo "$current,$token"
  fi
}

apply_kv() {
  local key="${1:-}"
  local value="${2:-}"
  key="$(echo "$key" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"

  case "$key" in
    crate | crates)
      CRATES="$(append_csv "$CRATES" "$value")"
      ;;
    bench | benches)
      BENCHES="$(append_csv "$BENCHES" "$value")"
      ;;
    only)
      ONLY="$(append_csv "$ONLY" "$value")"
      ;;
    filter)
      FILTER="$(append_csv "$FILTER" "$value")"
      ;;
    quick)
      QUICK="$value"
      ;;
    warmup_ms)
      WARMUP_MS="$value"
      ;;
    measure_ms)
      MEASURE_MS="$value"
      ;;
    sample_size)
      SAMPLE_SIZE="$value"
      ;;
    profile_time_secs)
      PROFILE_TIME_SECS="$value"
      ;;
    output_dir)
      OUTPUT_DIR="$value"
      ;;
    clean)
      CLEAN="$value"
      ;;
    *)
      echo "error: unknown key '$key' in '$key=$value'" >&2
      usage >&2
      exit 2
      ;;
  esac
}

CRATES="${BENCH_CRATES:-}"
BENCHES="${BENCH_BENCHES:-}"
ONLY="${BENCH_ONLY:-}"
FILTER="${BENCH_FILTER:-}"
QUICK="${BENCH_QUICK:-false}"
WARMUP_MS="${BENCH_WARMUP_MS:-}"
MEASURE_MS="${BENCH_MEASURE_MS:-}"
SAMPLE_SIZE="${BENCH_SAMPLE_SIZE:-}"
PROFILE_TIME_SECS="${BENCH_PROFILE_TIME_SECS:-}"
OUTPUT_DIR="${BENCH_OUTPUT_DIR:-}"
CLEAN="${BENCH_CLEAN:-true}"

while [[ $# -gt 0 ]]; do
  token="$1"
  case "$token" in
    -h | --help)
      usage
      exit 0
      ;;
    --quick)
      QUICK="true"
      ;;
    --no-quick)
      QUICK="false"
      ;;
    --clean)
      CLEAN="true"
      ;;
    --no-clean)
      CLEAN="false"
      ;;
    --*=*)
      apply_kv "${token%%=*}" "${token#*=}"
      ;;
    *=*)
      apply_kv "${token%%=*}" "${token#*=}"
      ;;
    --*)
      echo "error: unknown option '$token'" >&2
      usage >&2
      exit 2
      ;;
    *)
      ONLY="$(append_csv "$ONLY" "$token")"
      ;;
  esac
  shift
done

# ── Structured results output ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

detect_bench_os() {
  case "$(uname -s)" in
    Linux) echo "linux" ;;
    Darwin) echo "macos" ;;
    Windows | MINGW* | MSYS* | CYGWIN*) echo "windows" ;;
    *) uname -s | tr '[:upper:]' '[:lower:]' ;;
  esac
}

detect_bench_arch() {
  case "$(uname -m)" in
    arm64) echo "aarch64" ;;
    x86_64 | amd64) echo "x86-64" ;;
    *) uname -m ;;
  esac
}

RUN_OS="$(detect_bench_os)"
RUN_ARCH="$(detect_bench_arch)"
RUN_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ -n "${RSCRYPTO_BENCH_MODE:-}" ]]; then
  RUN_MODE="$RSCRYPTO_BENCH_MODE"
elif [[ -n "${DEV_MACHINE_TARGET:-}" ]]; then
  RUN_MODE="remote"
else
  RUN_MODE="local"
fi

if [[ "$RUN_MODE" == "local" && "$RUN_OS" == "macos" \
  && -z "${RUSTFLAGS+x}" && -z "${CARGO_ENCODED_RUSTFLAGS+x}" ]]; then
  export RUSTFLAGS="-C target-cpu=native"
fi

if [[ "$RUN_MODE" == "local" ]]; then
  RUN_DATE="$(date +"%Y-%m-%d")"
  RUN_TIME="$(date +"%H_%M_%S")"
else
  RUN_DATE="$(date -u +"%Y-%m-%d")"
  RUN_TIME="$(date -u +"%H_%M_%S")"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rscrypto-bench.XXXXXX")"
  trap 'rm -rf "$OUTPUT_DIR"' EXIT
fi

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "error: sha256sum or shasum is required for benchmark provenance" >&2
    return 1
  fi
}

write_source_manifest() {
  local destination="$1" source_file digest
  : > "$destination"
  while IFS= read -r -d '' source_file; do
    if [[ -f "$REPO_ROOT/$source_file" ]]; then
      digest="$(sha256_file "$REPO_ROOT/$source_file")"
      printf '%s  %q\n' "$digest" "$source_file" >> "$destination"
    else
      printf 'MISSING  %q\n' "$source_file" >> "$destination"
    fi
  done < <(
    git -C "$REPO_ROOT" ls-files --cached --others --exclude-standard -z -- \
      Cargo.toml Cargo.lock build.rs rust-toolchain.toml .cargo \
      .config/benchmark-matrix.json src benches scripts/bench \
      | sort -z
  )
}

REMOTE_ARTIFACT_DIR=""
REMOTE_RUN_ID=""
if [[ -n "${DEV_MACHINE_TARGET:-}" ]]; then
  REMOTE_RUN_ID="${RSCRYPTO_BENCH_RUN_ID:-${DEV_MACHINE_TARGET}-${RUN_DATE//-/}T${RUN_TIME//_/}Z-$$}"
  if [[ ! "$REMOTE_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || "$REMOTE_RUN_ID" == *..* ]]; then
    echo "error: invalid RSCRYPTO_BENCH_RUN_ID '$REMOTE_RUN_ID'" >&2
    exit 2
  fi
  REMOTE_ARTIFACT_DIR="$REPO_ROOT/benchmark_results/criterion/$REMOTE_RUN_ID"
  [[ ! -e "$REMOTE_ARTIFACT_DIR" ]] || {
    echo "error: remote benchmark run already exists: $REMOTE_RUN_ID" >&2
    exit 2
  }
  mkdir -p "$REMOTE_ARTIFACT_DIR"
  RESULTS_DIR="$REMOTE_ARTIFACT_DIR"
  write_source_manifest "$REMOTE_ARTIFACT_DIR/source-files.sha256"
  SOURCE_ID="$(sha256_file "$REMOTE_ARTIFACT_DIR/source-files.sha256")"
  {
    echo "run_id=$REMOTE_RUN_ID"
    echo "target=$DEV_MACHINE_TARGET"
    echo "instance_type=${DEV_MACHINE_INSTANCE_TYPE:-unknown}"
    echo "date=$RUN_DATE"
    echo "time=$RUN_TIME"
    echo "mode=$RUN_MODE"
    echo "platform=$RUN_OS-$RUN_ARCH"
    echo "commit=$RUN_COMMIT"
    echo "source_identity=sha256:$SOURCE_ID"
  } > "$REMOTE_ARTIFACT_DIR/remote-run.txt"
  uname -a > "$REMOTE_ARTIFACT_DIR/uname.txt"
  if command -v lscpu >/dev/null 2>&1; then
    lscpu > "$REMOTE_ARTIFACT_DIR/lscpu.txt"
  fi
  rustc -Vv > "$REMOTE_ARTIFACT_DIR/rustc.txt"
  cargo -V > "$REMOTE_ARTIFACT_DIR/cargo.txt"
  git -C "$REPO_ROOT" status --short > "$REMOTE_ARTIFACT_DIR/git-status.txt"
else
  RESULTS_DIR="$REPO_ROOT/benchmark_results/$RUN_DATE/$RUN_OS/$RUN_ARCH"
fi

bench_status=0
BENCH_CRATES="$CRATES" \
  BENCH_BENCHES="$BENCHES" \
  BENCH_ONLY="$ONLY" \
  BENCH_FILTER="$FILTER" \
  BENCH_QUICK="$QUICK" \
  BENCH_WARMUP_MS="$WARMUP_MS" \
  BENCH_MEASURE_MS="$MEASURE_MS" \
  BENCH_SAMPLE_SIZE="$SAMPLE_SIZE" \
  BENCH_PROFILE_TIME_SECS="$PROFILE_TIME_SECS" \
  BENCH_OUTPUT_DIR="$OUTPUT_DIR" \
  BENCH_CLEAN="$CLEAN" \
  BENCH_RESULTS_DIR="$RESULTS_DIR" \
  BENCH_RUN_DATE="$RUN_DATE" \
  BENCH_RUN_TIME="$RUN_TIME" \
  BENCH_RUN_OS="$RUN_OS" \
  BENCH_RUN_ARCH="$RUN_ARCH" \
  BENCH_RUN_COMMIT="$RUN_COMMIT" \
  BENCH_RUN_MODE="$RUN_MODE" \
  scripts/bench/run.sh || bench_status=$?

if [[ -n "$REMOTE_ARTIFACT_DIR" ]]; then
  TRANSFER_DIR="$REPO_ROOT/benchmark_results/.transfers"
  mkdir -p "$TRANSFER_DIR"
  tar -cf "$TRANSFER_DIR/$REMOTE_RUN_ID.tar" \
    -C "$REPO_ROOT/benchmark_results" "criterion/$REMOTE_RUN_ID"
  TRANSFER_DIGEST="$(sha256_file "$TRANSFER_DIR/$REMOTE_RUN_ID.tar")"
  printf '%s  %s.tar\n' "$TRANSFER_DIGEST" "$REMOTE_RUN_ID" \
    > "$TRANSFER_DIR/$REMOTE_RUN_ID.tar.sha256"
  echo "Remote run ID: $REMOTE_RUN_ID"
fi

exit "$bench_status"
