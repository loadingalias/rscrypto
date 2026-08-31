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

assert_single_libtest() {
  local test_name=$1
  shift

  local listing count
  if ! listing=$("$@" --list); then
    die "unable to list the test harness containing $test_name"
  fi
  count=$(printf '%s\n' "$listing" | awk -v expected="$test_name: test" '$0 == expected { count++ } END { print count + 0 }')
  [[ "$count" -eq 1 ]] || die "expected exactly one libtest named $test_name; found $count"
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

run_examples() {
  just test-examples
}

run_msrv() {
  cargo check --locked --workspace --lib --no-default-features
  cargo check --locked --workspace --lib --all-features
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
      cargo clippy --locked --workspace --lib --all-features
      cargo test --locked --workspace --all-features --no-run
      cargo test --locked --workspace --features blake3 \
        --test blake3_official_vectors \
        --test blake3_differential
      ;;
    aarch64-pc-windows-msvc)
      echo "Windows AArch64 lane: compile-only smoke"
      cargo clippy --locked --workspace --lib --all-features
      cargo test --locked --workspace --all-features --no-run
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

run_platform_amx() {
  local runner=${RSCRYPTO_CI_RUNNER:-}
  [[ "$runner" == *"intel-spr"* ]] || die "AMX permission evidence requires the intel-spr runner"
  [[ "$(uname -s)" == Linux ]] || die "AMX permission evidence requires Linux"
  [[ "$(uname -m)" == x86_64 ]] || die "AMX permission evidence requires x86-64"
  [[ "$(rustc -vV | sed -n 's/^host: //p')" == x86_64-unknown-linux-gnu ]] \
    || die "AMX permission evidence requires the x86_64-unknown-linux-gnu Rust host"

  host_diagnostics 80
  local flags
  flags=$(sed -n 's/^flags[[:space:]]*: //p' /proc/cpuinfo | head -n 1)
  [[ " $flags " == *" amx_tile "* ]] || die "intel-spr runner does not expose AMX-TILE"

  # This proof lane never consumes symbols; omitting them keeps both test
  # codegen artifact sets proportional to the work.
  export CARGO_PROFILE_TEST_DEBUG=0

  RSCRYPTO_REQUIRE_AMX=1 \
    assert_single_libtest \
      linux_x86_64_amx_permission_and_cache_are_process_scoped \
      cargo test --locked --test platform_amx_permission --
  RSCRYPTO_REQUIRE_AMX=1 \
    cargo test --locked --test platform_amx_permission \
      linux_x86_64_amx_permission_and_cache_are_process_scoped -- --exact --nocapture

  # NIGHTLY: Rust target-feature names for AMX remain unstable. This lane
  # deliberately forces them so the no_std permission gate is executable.
  local amx_rustflags="-C target-feature=+amx-tile,+amx-bf16,+amx-int8"
  RUSTFLAGS="$amx_rustflags" \
    assert_single_libtest \
      platform::detect::tests::no_std_linux_x86_64_masks_compile_time_amx_without_a_permission_probe \
      cargo test --locked --no-default-features --lib --
  RUSTFLAGS="$amx_rustflags" \
    cargo test --locked --no-default-features --lib \
      platform::detect::tests::no_std_linux_x86_64_masks_compile_time_amx_without_a_permission_probe \
      -- --exact --nocapture
}

run_cross_targets() {
  bash scripts/ci/cross-targets.sh deep
}

run_supply_chain() {
  local mode=${RSCRYPTO_CI_SUPPLY_CHAIN_MODE:-}
  require_one_of supply_chain_mode "$mode" light full

  if [[ "$mode" == "full" ]]; then
    cargo deny --locked check all
    # RustCrypto `rsa` is used only as a dev/test/bench oracle. Production RSA
    # verification is implemented in `src/auth/rsa.rs`; keep this scoped to the
    # known Marvin advisory until the oracle dependency is removed or fixed.
    cargo audit --ignore RUSTSEC-2023-0071
  else
    cargo deny --locked check advisories
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
  export RSCRYPTO_FUZZ_JOBS=1
  export RSCRYPTO_FUZZ_TARGET_CONCURRENCY=2
  local fuzz_status=0
  just test-fuzz --all || fuzz_status=$?

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

  return "$fuzz_status"
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

  export BENCH_OUTPUT_DIR=target/benchmark_results
  export BENCH_RESULTS_DIR=target/benchmark_results
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
  scripts/lib/python.sh scripts/ct/package_evidence.py "${package_args[@]}"
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

run_rsa_linux_x86_64_asm() {
  mkdir -p ci-evidence
  {
    uname -a
    [[ "$(uname -s)" == Linux ]] || die "RSA x86-64 assembly evidence requires Linux"
    [[ "$(uname -m)" == x86_64 ]] || die "RSA x86-64 assembly evidence requires x86-64"
    [[ "$(rustc -vV | sed -n 's/^host: //p')" == x86_64-unknown-linux-gnu ]] \
      || die "RSA x86-64 assembly evidence requires the x86_64-unknown-linux-gnu Rust host"

    local flags
    flags=$(sed -n 's/^flags[[:space:]]*: //p' /proc/cpuinfo | head -n 1)
    [[ " $flags " == *" bmi2 "* ]] || die "RSA x86-64 assembly evidence requires BMI2"
    [[ " $flags " == *" adx "* ]] || die "RSA x86-64 assembly evidence requires ADX"
    lscpu

    assert_single_libtest \
      auth::rsa::tests::x86_64_linux_rsa_montgomery_asm_matches_portable_across_supported_widths \
      cargo test --locked --features rsa,diag,getrandom --lib --
    cargo test --locked --features rsa,diag,getrandom --lib \
      auth::rsa::tests::x86_64_linux_rsa_montgomery_asm_matches_portable_across_supported_widths \
      -- --exact --nocapture
    assert_single_libtest \
      auth::rsa::tests::x86_64_linux_rsa_montgomery_asm_matches_portable_across_supported_widths \
      cargo test --locked --release --features rsa,diag,getrandom --lib --
    cargo test --locked --release --features rsa,diag,getrandom --lib \
      auth::rsa::tests::x86_64_linux_rsa_montgomery_asm_matches_portable_across_supported_widths \
      -- --exact --nocapture

    local build_output binary
    build_output=$(cargo test --locked --release --features rsa,diag \
      --test rsa_public_key --no-run --message-format=json)
    binary=$(printf '%s\n' "$build_output" \
      | sed -n 's/.*"executable":"\([^"]*rsa_public_key-[^"]*\)".*/\1/p' \
      | tail -n 1)
    [[ -n "$binary" && -x "$binary" ]] \
      || die "unable to resolve the optimized rsa_public_key test binary"
    printf 'Optimized RSA test binary: %s\n' "$binary"
    assert_single_libtest public_operation_montgomery_candidates_match_current_path "$binary"
    "$binary" public_operation_montgomery_candidates_match_current_path --exact --nocapture

    local binary_description binary_symbols
    binary_description=$(file "$binary") || die "unable to inspect the optimized rsa_public_key test binary"
    [[ "$binary_description" == *"ELF 64-bit LSB pie executable, x86-64"* ]] \
      || die "optimized rsa_public_key test binary is not x86-64 ELF"
    binary_symbols=$(nm "$binary") || die "unable to read the optimized rsa_public_key symbol table"
    [[ "$binary_symbols" == *"rscrypto_rsa_bn_mulx4x_mont_x86_64_elf"* ]] \
      || die "optimized rsa_public_key test binary lacks the x86-64 Montgomery multiply"
    [[ "$binary_symbols" == *"rscrypto_rsa_bn_sqr8x_mont_x86_64_elf"* ]] \
      || die "optimized rsa_public_key test binary lacks the x86-64 Montgomery square"
  } 2>&1 | tee ci-evidence/rsa-linux-x86_64-asm.log
}

main() {
  if [[ $# -ne 0 ]]; then
    die "usage: scripts/ci/run-rust-job.sh"
  fi

  local operation=${RSCRYPTO_CI_OPERATION:-}
  require_nonempty operation "$operation"
  case "$operation" in
    quality) run_quality ;;
    examples) run_examples ;;
    msrv) run_msrv ;;
    cargo-graph) run_cargo_graph ;;
    feature-contracts) run_feature_contracts ;;
    native) run_native ;;
    native-ibm) run_native_ibm ;;
    native-riscv) run_native_riscv ;;
    platform-amx) run_platform_amx ;;
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
    rsa-linux-x64-asm) run_rsa_linux_x86_64_asm ;;
    *) die "unsupported operation: $operation" ;;
  esac
}

main "$@"
