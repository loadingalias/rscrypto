#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

die() {
  echo "rust job error: $*" >&2
  exit 2
}

require_nonempty() {
  local name=$1
  local value=$2
  [[ -n "$value" ]] || die "$name is required"
}

require_one_of() {
  local name=$1
  local value=$2
  shift 2

  local allowed
  for allowed in "$@"; do
    if [[ "$value" == "$allowed" ]]; then
      return 0
    fi
  done

  die "invalid $name: $value"
}

require_bool() {
  require_one_of "$1" "$2" true false
}

require_positive_integer() {
  local name=$1
  local value=$2
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer"
}

require_commit_sha() {
  local value=$1
  [[ "$value" =~ ^([[:xdigit:]]{40}|[[:xdigit:]]{64})$ ]] \
    || die "base_sha must be a full commit ID"
}

host_diagnostics() {
  local cpuinfo_lines=$1
  uname -a
  lscpu
  sed -n "1,${cpuinfo_lines}p" /proc/cpuinfo
}

run_quality() {
  just ci-check
}

run_cargo_graph() {
  mkdir -p target/cargo-rail
  cargo rail config validate --strict
  cargo rail config migrate --check
  cargo rail unify --check --explain --format json \
    --output target/cargo-rail/unify-result.json
}

run_feature_contracts() {
  just check-feature-matrix
  just test-feature-matrix
}

run_native() {
  local target=${RSCRYPTO_CI_TARGET:-}
  local test_mode=${RSCRYPTO_CI_TEST_MODE:-}
  require_one_of target "$target" \
    x86_64-unknown-linux-gnu \
    aarch64-unknown-linux-gnu \
    x86_64-pc-windows-msvc \
    aarch64-pc-windows-msvc
  require_one_of test_mode "$test_mode" local commit weekly
  export RSCRYPTO_TEST_MODE="$test_mode"

  case "$target" in
    x86_64-pc-windows-msvc)
      echo "Windows x86_64 lane: compile smoke plus BLAKE3 ASM runtime vectors"
      cargo clippy --workspace --lib --all-features -- -D warnings
      cargo test --workspace --all-features --no-run
      cargo test --workspace --features blake3 \
        --test blake3_official_vectors \
        --test blake3_differential
      ;;
    aarch64-pc-windows-msvc)
      echo "Windows AArch64 lane: compile-only smoke"
      cargo clippy --workspace --lib --all-features -- -D warnings
      cargo test --workspace --all-features --no-run
      ;;
    x86_64-unknown-linux-gnu | aarch64-unknown-linux-gnu)
      bash scripts/ci/native-check.sh --all-targets
      bash scripts/test/test.sh --all
      ;;
  esac
}

run_native_ibm() {
  local test_mode=${RSCRYPTO_CI_TEST_MODE:-}
  require_one_of test_mode "$test_mode" local commit weekly
  export RSCRYPTO_TEST_MODE="$test_mode"

  host_diagnostics 50
  bash scripts/ci/native-check.sh
  bash scripts/test/test.sh --all
}

run_native_riscv() {
  host_diagnostics 50
  export RSCRYPTO_TEST_MODE=weekly
  export RSCRYPTO_CI_RESOURCE_PROFILE=constrained
  export RSCRYPTO_SKIP_DOCTESTS=1
  bash scripts/ci/native-check.sh
  bash scripts/test/test.sh --all
}

run_cross_targets() {
  bash scripts/ci/cross-targets.sh deep
}

run_supply_chain() {
  local mode=${RSCRYPTO_CI_SUPPLY_CHAIN_MODE:-}
  require_one_of supply_chain_mode "$mode" light full

  if [[ "$mode" == "full" ]]; then
    cargo deny check all
    # RustCrypto `rsa` is used only as a dev/test/bench oracle. Production RSA
    # verification is implemented in `src/auth/rsa.rs`; keep this scoped to the
    # known Marvin advisory until the oracle dependency is removed or fixed.
    cargo audit --ignore RUSTSEC-2023-0071
  else
    cargo deny check advisories
  fi
}

run_dependabot_smoke() {
  local base_sha=${RSCRYPTO_CI_BASE_SHA:-}
  require_commit_sha "$base_sha"
  export GITHUB_BASE_SHA="$base_sha"
  bash scripts/ci/dependabot-smoke.sh
}

run_miri() {
  local tree_borrows=${RSCRYPTO_CI_MIRI_TREE_BORROWS:-false}
  require_bool miri_tree_borrows "$tree_borrows"
  if [[ "$tree_borrows" == "true" ]]; then
    export MIRIFLAGS=-Zmiri-tree-borrows
  fi
  just test-miri
}

run_fuzz() {
  export RSCRYPTO_FUZZ_DURATION_SECS=60
  just test-fuzz --all

  rm -rf -- fuzz-output
  mkdir -p fuzz-output
  mapfile -t corpus_dirs < <(
    {
      [[ -d fuzz/corpus ]] && printf '%s\n' fuzz/corpus
      find fuzz-packages -mindepth 2 -maxdepth 2 -type d -name corpus
    } | sort
  )
  if [[ "${#corpus_dirs[@]}" -eq 0 ]]; then
    tar -czf fuzz-output/corpus.tar.gz --files-from /dev/null
  else
    tar -czf fuzz-output/corpus.tar.gz "${corpus_dirs[@]}"
  fi
}

run_fuzz_asan() {
  just test-fuzz-asan --all
}

run_mlkem_aarch64() {
  local platform=${RSCRYPTO_CI_PLATFORM:-}
  local display_name
  require_one_of platform "$platform" graviton3 graviton4
  case "$platform" in
    graviton3) display_name="AWS Graviton3" ;;
    graviton4) display_name="AWS Graviton4" ;;
  esac

  echo "ML-KEM gate host: $display_name ($platform)"
  host_diagnostics 80
  export MLKEM_AARCH64_GATE_PLATFORM="$platform"
  bash scripts/ci/mlkem-aarch64-gate.sh
}

run_benchmark() {
  local platform=${RSCRYPTO_CI_PLATFORM:-}
  local targets=${RSCRYPTO_CI_BENCH_TARGETS:-all}
  local filter=${RSCRYPTO_CI_BENCH_FILTER:-}
  local quick=${RSCRYPTO_CI_BENCH_QUICK:-false}
  local run_date
  local run_time
  local run_commit
  require_one_of platform "$platform" \
    amd-zen4 intel-spr intel-icl amd-zen5 \
    graviton3 graviton4 ibm-s390x ibm-power10 rise-riscv
  require_bool bench_quick "$quick"

  echo "Bench host: $platform (linux)"
  host_diagnostics 50

  run_date="$(date -u +"%Y-%m-%d")"
  run_time="$(date -u +"%H_%M_%S")"
  run_commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

  export BENCH_OUTPUT_DIR=benchmark-results
  export BENCH_RESULTS_DIR=benchmark-results
  export BENCH_RUN_DATE="$run_date"
  export BENCH_RUN_TIME="$run_time"
  export BENCH_RUN_COMMIT="$run_commit"
  export BENCH_RUN_OS=linux
  export BENCH_RUN_ARCH="$platform"
  export BENCH_RUN_MODE=ci
  export BENCH_ONLY="$targets"
  export BENCH_FILTER="$filter"
  export BENCH_QUICK="$quick"
  export BENCH_ALLOW_FULL_HASHES_COMP=true
  export BENCH_PLATFORM="$platform"
  scripts/ci/run-bench.sh
}

ct_target_for_platform() {
  case "$1" in
    amd-zen4 | intel-spr | intel-icl | amd-zen5)
      echo x86_64-unknown-linux-gnu
      ;;
    graviton3 | graviton4)
      echo aarch64-unknown-linux-gnu
      ;;
    ibm-s390x)
      echo s390x-unknown-linux-gnu
      ;;
    ibm-power10)
      echo powerpc64le-unknown-linux-gnu
      ;;
    rise-riscv)
      echo riscv64gc-unknown-linux-gnu
      ;;
    *)
      die "invalid CT platform: $1"
      ;;
  esac
}

run_constant_time() {
  local platform=${RSCRYPTO_CI_PLATFORM:-}
  local runner=${RSCRYPTO_CI_RUNNER:-}
  local dudect_timeout=${RSCRYPTO_CI_DUDECT_TIMEOUT:-1800}
  local binsec_timeout=${RSCRYPTO_CI_BINSEC_TIMEOUT:-900}
  local raw_dudect_filter=${RSCRYPTO_CI_DUDECT_FILTER:-}
  local raw_dudect_gate=${RSCRYPTO_CI_DUDECT_GATE:-required}
  local raw_artifacts=${RSCRYPTO_CI_UPLOAD_RAW_ARTIFACTS:-false}
  local target=${RSCRYPTO_CI_TARGET:-}
  local expected_target
  expected_target="$(ct_target_for_platform "$platform")"
  [[ "$target" == "$expected_target" ]] || die "CT target does not match platform"
  require_positive_integer dudect_timeout "$dudect_timeout"
  require_positive_integer binsec_timeout "$binsec_timeout"
  require_bool upload_raw_artifacts "$raw_artifacts"

  local evidence_dir=target/ct-evidence-package
  mkdir -p "$evidence_dir"
  {
    echo "CT platform: $platform"
    echo "CT target: $target"
    echo "CT runner: $runner"
    uname -a || true
    rustc -vV
    cargo -V
    lscpu || true
    sed -n '1,80p' /proc/cpuinfo || true
  } 2>&1 | tee "$evidence_dir/host-$platform.log"

  local dudect_filter
  local dudect_gate
  dudect_filter="$(printf '%s' "$raw_dudect_filter" | tr -d '[:space:]')"
  dudect_gate="$(printf '%s' "$raw_dudect_gate" | tr -d '[:space:]')"
  if [[ "$raw_dudect_filter" != "$dudect_filter" ]]; then
    echo "Normalized whitespace in DudeCT filter input."
  fi
  require_one_of dudect_gate "$dudect_gate" required diagnostic all

  local -a args=(
    --target "$target"
    --dudect-timeout "$dudect_timeout"
    --binsec-timeout "$binsec_timeout"
    --dudect-gate "$dudect_gate"
  )
  if [[ -n "$dudect_filter" ]]; then
    args+=(--dudect-filter "$dudect_filter")
  fi

  local -a package_args=(
    --target "$target"
    --suffix "$platform"
    --out-dir "$evidence_dir"
  )
  local status=0
  {
    printf 'Running:'
    printf ' %q' scripts/ct/full.py "${args[@]}"
    printf '\n'
    scripts/ct/full.py "${args[@]}"
  } 2>&1 | tee "$evidence_dir/ct-full-$platform.log" || status=$?

  if [[ "$raw_artifacts" == "true" ]]; then
    package_args+=(--raw)
  fi
  scripts/ct/python.sh scripts/ct/package_evidence.py "${package_args[@]}"
  return "$status"
}

run_rsa_miri() {
  mkdir -p ci-evidence
  {
    uname -a
    lscpu
    just test-miri --rsa
  } 2>&1 | tee ci-evidence/rsa-miri-linux-x64.log
}

run_rsa_leakage() {
  local target=${RSCRYPTO_CI_TARGET:-}
  require_one_of target "$target" linux-x64 linux-arm64
  mkdir -p ci-evidence
  {
    uname -a
    lscpu
    RSCRYPTO_RSA_LEAKAGE_SAMPLES="${RSCRYPTO_RSA_LEAKAGE_SAMPLES:-4000}" \
    RSCRYPTO_RSA_LEAKAGE_T_THRESHOLD="${RSCRYPTO_RSA_LEAKAGE_T_THRESHOLD:-8.0}" \
      just test-rsa-leakage
  } 2>&1 | tee "ci-evidence/rsa-leakage-$target.log"
}

main() {
  if [[ $# -ne 0 ]]; then
    die "usage: scripts/ci/run-rust-job.sh"
  fi

  local operation=${RSCRYPTO_CI_OPERATION:-}
  require_nonempty operation "$operation"
  case "$operation" in
    quality) run_quality ;;
    cargo-graph) run_cargo_graph ;;
    feature-contracts) run_feature_contracts ;;
    native) run_native ;;
    native-ibm) run_native_ibm ;;
    native-riscv) run_native_riscv ;;
    cross-targets) run_cross_targets ;;
    supply-chain) run_supply_chain ;;
    dependabot-smoke) run_dependabot_smoke ;;
    miri) run_miri ;;
    fuzz) run_fuzz ;;
    fuzz-asan) run_fuzz_asan ;;
    mlkem-aarch64) run_mlkem_aarch64 ;;
    benchmark) run_benchmark ;;
    constant-time) run_constant_time ;;
    rsa-miri) run_rsa_miri ;;
    rsa-leakage) run_rsa_leakage ;;
    *) die "unsupported operation: $operation" ;;
  esac
}

main "$@"
