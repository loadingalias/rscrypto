#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "RSA assembly evidence error: $*" >&2
  exit 1
}

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] \
  || fail "requires a Linux x86-64 host"
flags=$(sed -n 's/^flags[[:space:]]*: //p' /proc/cpuinfo | head -n 1)
[[ " $flags " == *" bmi2 "* && " $flags " == *" adx "* ]] \
  || fail "requires BMI2 and ADX"

test_name=auth::rsa::tests::x86_64_linux_rsa_montgomery_asm_matches_portable_across_supported_widths
for profile in debug release; do
  args=(test --locked --features rsa,diag,getrandom --lib)
  [[ "$profile" == debug ]] || args+=(--release)
  listing=$(cargo "${args[@]}" -- --list)
  [[ $(awk -v expected="$test_name: test" '$0 == expected { count++ } END { print count + 0 }' <<<"$listing") -eq 1 ]] \
    || fail "expected exactly one $test_name test"
  cargo "${args[@]}" "$test_name" -- --exact --nocapture
done

build_output=$(cargo test --locked --release --features rsa,diag \
  --test rsa_public_key --no-run --message-format=json)
binary=$(sed -n 's/.*"executable":"\([^"]*rsa_public_key-[^"]*\)".*/\1/p' <<<"$build_output" | tail -n 1)
[[ -n "$binary" && -x "$binary" ]] || fail "could not resolve optimized RSA test binary"
symbols=$(nm "$binary")
[[ "$symbols" == *rscrypto_rsa_bn_mulx4x_mont_x86_64_elf* ]] \
  || fail "optimized binary lacks the Montgomery multiply"
[[ "$symbols" == *rscrypto_rsa_bn_sqr8x_mont_x86_64_elf* ]] \
  || fail "optimized binary lacks the Montgomery square"

test_name=public_operation_montgomery_candidates_match_current_path
[[ $("$binary" "$test_name" --list | awk -v expected="$test_name: test" '$0 == expected { count++ } END { print count + 0 }') -eq 1 ]] \
  || fail "expected exactly one $test_name test"
"$binary" "$test_name" --exact --nocapture
