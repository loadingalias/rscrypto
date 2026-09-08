#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

fail() {
  echo "RSA assembly evidence error: $*" >&2
  exit 1
}

[[ $# -eq 1 && ( "$1" == linux || "$1" == macos ) ]] || fail "usage: $0 {linux|macos}"
platform=$1

run_named_test() {
  local name=$1 listing count
  shift
  listing=$("$@" --list) || fail "unable to list $name"
  count=$(awk -v expected="$name: test" '$0 == expected { count++ } END { print count + 0 }' <<<"$listing")
  [[ "$count" -eq 1 ]] || fail "expected exactly one $name test; found $count"
  "$@" "$name" --exact --nocapture
}

run_gate() {
  local test_name flags cpu_brand hypervisor_present profile build_output binary symbols description symbol
  local expected_symbols=() nm_args=()
  if [[ "$platform" == linux ]]; then
    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail "requires a Linux x86-64 host"
    flags=$(awk '/^flags[[:space:]]*:/ { sub(/^[^:]*:[[:space:]]*/, ""); print; exit }' /proc/cpuinfo)
    [[ " $flags " == *" bmi2 "* && " $flags " == *" adx "* ]] || fail "requires BMI2 and ADX"
    test_name=auth::rsa::tests::x86_64_linux_rsa_montgomery_asm_matches_portable_across_supported_widths
    expected_symbols=(rscrypto_rsa_bn_mulx4x_mont_x86_64_elf rscrypto_rsa_bn_sqr8x_mont_x86_64_elf)
  else
    uname -a
    [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || fail "requires an Arm64 macOS host"
    [[ "$(rustc -vV | sed -n 's/^host: //p')" == aarch64-apple-darwin ]] || fail "requires the aarch64-apple-darwin Rust host"
    cpu_brand=$(sysctl -n machdep.cpu.brand_string) || fail "unable to read CPU identity"
    hypervisor_present=$(sysctl -n kern.hv_vmm_present) || fail "unable to determine whether macOS is virtualized"
    printf 'CPU: %s\nHypervisor present: %s\n' "$cpu_brand" "$hypervisor_present"
    [[ "$hypervisor_present" == 0 ]] || fail "requires a physical Apple Silicon Mac"
    rustc -vV
    test_name=auth::rsa::tests::aarch64_macos_rsa_montgomery_asm_matches_portable_across_supported_widths
    nm_args=(-m)
    expected_symbols=(_rscrypto_rsa_bn_mul_mont_words_apple
      _rscrypto_rsa_mont_reduce_cios_32_aarch64_apple_darwin
      _rscrypto_rsa_mont_reduce_cios_words_aarch64_apple_darwin)
  fi

  for profile in debug release; do
    local args=(test --locked --features "rsa,diag,getrandom" --lib)
    [[ "$profile" == debug ]] || args+=(--release)
    run_named_test "$test_name" cargo "${args[@]}" --
  done

  build_output=$(cargo test --locked --release --features rsa,diag --test rsa_public_key --no-run --message-format=json)
  binary=$(jq -ers '[.[] | select(.reason == "compiler-artifact" and .target.name == "rsa_public_key" and .executable != null) | .executable]
    | if length == 1 then .[0] else error("expected one RSA test executable") end' <<<"$build_output")
  [[ -n "$binary" && -x "$binary" ]] || fail "could not resolve optimized RSA test binary"
  if [[ "$platform" == macos ]]; then
    description=$(file "$binary") || fail "unable to inspect the optimized binary"
    [[ "$description" == *"Mach-O 64-bit executable arm64"* ]] || fail "optimized binary is not Arm64 Mach-O"
  fi
  symbols=$(nm "${nm_args[@]:+${nm_args[@]}}" "$binary") || fail "unable to read optimized symbols"
  for symbol in "${expected_symbols[@]}"; do
    [[ "$symbols" == *"$symbol"* ]] || fail "optimized binary lacks $symbol"
  done
  run_named_test public_operation_montgomery_candidates_match_current_path "$binary"
}

if [[ "$platform" == macos ]]; then
  mkdir -p target/rsa-macos-asm
  run_gate 2>&1 | tee target/rsa-macos-asm/evidence.log
else
  run_gate
fi
