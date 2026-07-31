#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

die() {
  echo "RSA macOS assembly evidence error: $*" >&2
  exit 1
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

[[ $# -eq 0 ]] || die "usage: scripts/test/test-rsa-macos-asm.sh"

evidence_dir="$REPO_ROOT/target/rsa-macos-asm"
mkdir -p "$evidence_dir"
{
  uname -a
  [[ "$(uname -s)" == Darwin ]] || die "local RSA macOS assembly evidence requires macOS"
  [[ "$(uname -m)" == arm64 ]] || die "local RSA macOS assembly evidence requires Arm64"
  [[ "$(rustc -vV | sed -n 's/^host: //p')" == aarch64-apple-darwin ]] \
    || die "local RSA macOS assembly evidence requires the aarch64-apple-darwin Rust host"
  cpu_brand=$(sysctl -n machdep.cpu.brand_string) \
    || die "unable to read the macOS CPU identity"
  hypervisor_present=$(sysctl -n kern.hv_vmm_present) \
    || die "unable to determine whether macOS is virtualized"
  printf 'CPU: %s\nHypervisor present: %s\n' "$cpu_brand" "$hypervisor_present"
  [[ "$hypervisor_present" == 0 ]] \
    || die "local RSA macOS assembly evidence requires a physical Apple Silicon Mac"
  rustc -vV

  assert_single_libtest \
    auth::rsa::tests::aarch64_macos_rsa_montgomery_asm_matches_portable_across_supported_widths \
    cargo test --locked --features rsa,diag,getrandom --lib --
  cargo test --locked --features rsa,diag,getrandom --lib \
    auth::rsa::tests::aarch64_macos_rsa_montgomery_asm_matches_portable_across_supported_widths \
    -- --exact --nocapture
  assert_single_libtest \
    auth::rsa::tests::aarch64_macos_rsa_montgomery_asm_matches_portable_across_supported_widths \
    cargo test --locked --release --features rsa,diag,getrandom --lib --
  cargo test --locked --release --features rsa,diag,getrandom --lib \
    auth::rsa::tests::aarch64_macos_rsa_montgomery_asm_matches_portable_across_supported_widths \
    -- --exact --nocapture

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

  binary_description=$(file "$binary") || die "unable to inspect the optimized rsa_public_key test binary"
  [[ "$binary_description" == *"Mach-O 64-bit executable arm64"* ]] \
    || die "optimized rsa_public_key test binary is not Arm64 Mach-O"
  binary_symbols=$(nm -m "$binary") || die "unable to read the optimized rsa_public_key symbol table"
  [[ "$binary_symbols" == *"_rscrypto_rsa_bn_mul_mont_words_apple"* ]] \
    || die "optimized rsa_public_key test binary lacks the Apple Montgomery multiply"
  [[ "$binary_symbols" == *"_rscrypto_rsa_mont_reduce_cios_32_aarch64_apple_darwin"* ]] \
    || die "optimized rsa_public_key test binary lacks the Apple 32-word Montgomery reduction"
  [[ "$binary_symbols" == *"_rscrypto_rsa_mont_reduce_cios_words_aarch64_apple_darwin"* ]] \
    || die "optimized rsa_public_key test binary lacks the Apple generic Montgomery reduction"
} 2>&1 | tee "$evidence_dir/evidence.log"
