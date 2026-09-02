#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 2 && $# -le 3 ]] || {
  echo "usage: $0 <platform|amx> <host-target> [shallow|deep]" >&2
  exit 2
}

platform=$1
target=$2
depth=${3:-deep}
[[ "$depth" == shallow || "$depth" == deep ]] || {
  echo "depth must be shallow or deep" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

fail() {
  echo "native platform error: $*" >&2
  exit 1
}

assert_single_libtest() {
  local test_name=$1
  shift
  local listing count
  listing=$("$@" --list) || fail "unable to list the test harness containing $test_name"
  count=$(awk -v expected="$test_name: test" '$0 == expected { count++ } END { print count + 0 }' <<<"$listing")
  [[ "$count" -eq 1 ]] || fail "expected exactly one libtest named $test_name; found $count"
}

require_host() {
  local actual
  actual=$(rustc -vV | sed -n 's/^host: //p')
  [[ "$actual" == "$target" ]] || fail "expected Rust host $target, found $actual"

  case "$platform" in
    linux-arm64 | macos-arm64) [[ "$(uname -m)" == aarch64 || "$(uname -m)" == arm64 ]] ;;
    macos-x64 | windows-x64) [[ "$(uname -m)" == x86_64 ]] ;;
    # Windows Arm64 runs an x86-64 Git Bash; the exact native Rust host above is authoritative.
    windows-arm64) ;;
    ibm-s390x) [[ "$(uname -m)" == s390x ]] ;;
    ibm-power10) [[ "$(uname -m)" == ppc64le ]] ;;
    rise-riscv) [[ "$(uname -m)" == riscv64 ]] ;;
    amx) [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] ;;
    *) fail "unsupported native platform: $platform" ;;
  esac || fail "host architecture does not match $platform"
}

run_amx() {
  local flags test_name
  flags=$(awk '/^flags[[:space:]]*:/ { sub(/^[^:]*:[[:space:]]*/, ""); print; exit }' /proc/cpuinfo)
  [[ " $flags " == *" amx_tile "* ]] || fail "intel-spr runner does not expose AMX-TILE"
  export CARGO_PROFILE_TEST_DEBUG=0

  test_name=linux_x86_64_amx_permission_and_cache_are_process_scoped
  RSCRYPTO_REQUIRE_AMX=1 assert_single_libtest "$test_name" \
    cargo test --locked --test platform_amx_permission --
  RSCRYPTO_REQUIRE_AMX=1 cargo test --locked --test platform_amx_permission \
    "$test_name" -- --exact --nocapture

  test_name=platform::detect::tests::no_std_linux_x86_64_masks_compile_time_amx_without_a_permission_probe
  local rustflags="-C target-feature=+amx-tile,+amx-bf16,+amx-int8"
  RUSTFLAGS="$rustflags" assert_single_libtest "$test_name" \
    cargo test --locked --no-default-features --lib --
  RUSTFLAGS="$rustflags" cargo test --locked --no-default-features --lib \
    "$test_name" -- --exact --nocapture
}

run_native_runtime() {
  export RSCRYPTO_TEST_MODE=commit
  if [[ "$platform" == rise-riscv ]]; then
    export RSCRYPTO_CI_RESOURCE_PROFILE=constrained
    apply_ci_resource_profile
  elif [[ "$platform" == ibm-s390x ]]; then
    export CARGO_TARGET_S390X_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C target-feature=+vector"
  fi

  cargo test --locked --lib --all-features
  cargo run --locked --example introspect --features 'crc32,sha2,chacha20poly1305,diag'

  if [[ "$depth" == deep ]]; then
    cargo test --locked --all-features \
      --test aead_kernel_equivalence \
      --test portable_fallback \
      --test vectored_dispatch
  fi
}

echo "Native platform proof: $platform ($target, $depth)"
uname -a
rustc -vV
command -v lscpu >/dev/null 2>&1 && lscpu || true
require_host

case "$platform" in
  windows-arm64)
    cargo clippy --locked --workspace --lib --all-features
    cargo test --locked --workspace --all-features --no-run
    ;;
  windows-x64)
    cargo clippy --locked --workspace --lib --all-features
    cargo test --locked --lib --all-features
    cargo test --locked --features blake3 \
      --test blake3_official_vectors --test blake3_differential
    ;;
  linux-arm64 | macos-x64 | macos-arm64 | ibm-s390x | ibm-power10 | rise-riscv)
    run_native_runtime
    ;;
  amx) run_amx ;;
  *) fail "unsupported native platform: $platform" ;;
esac

echo "Native platform proof passed: $platform ($target, $depth)"
