#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MANIFEST="$REPO_ROOT/src/auth/asm/rscrypto_rsa_assembly_provenance.tsv"
ARCHIVE_PREFIX="aws-lc-sys-0.41.0/"
TRANSFORM="rsa-aws-lc-sys-0.41.0-transform-v1"
TRANSFORM_MANIFEST_SHA256="f0bc00139c7f44e793b5efb292a73650254a7335e739e95b6b46aff429bf8932"

die() {
  printf 'rsa-asm-provenance: %s\n' "$*" >&2
  exit 1
}

usage() {
  printf 'usage: scripts/check/rsa-asm-provenance.sh [--archive PATH]\n' >&2
  exit 2
}

if command -v sha256sum >/dev/null 2>&1; then
  hash_file() {
    sha256sum "$1" | awk '{ print $1 }'
  }
elif command -v shasum >/dev/null 2>&1; then
  hash_file() {
    shasum -a 256 "$1" | awk '{ print $1 }'
  }
else
  die "sha256sum or shasum is required"
fi

require_shape() {
  local path=$1
  local expected_lines=$2
  local expected_bytes=$3
  local expected_hash=$4
  local actual_lines actual_bytes actual_hash

  [[ -f "$path" ]] || die "missing file: $path"
  actual_lines=$(wc -l <"$path" | tr -d ' ')
  actual_bytes=$(wc -c <"$path" | tr -d ' ')
  actual_hash=$(hash_file "$path")
  [[ "$actual_lines" == "$expected_lines" ]] \
    || die "$path has $actual_lines lines; expected $expected_lines"
  [[ "$actual_bytes" == "$expected_bytes" ]] \
    || die "$path has $actual_bytes bytes; expected $expected_bytes"
  [[ "$actual_hash" == "$expected_hash" ]] \
    || die "$path has SHA-256 $actual_hash; expected $expected_hash"
}

require_lf_file() {
  local path=$1
  [[ -s "$path" ]] || die "empty transform input: $path"
  if LC_ALL=C grep -q $'\r' "$path"; then
    die "transform input is not LF-only: $path"
  fi
  [[ -z "$(tail -c 1 "$path")" ]] || die "transform input lacks a final LF: $path"
}

validate_manifest() {
  [[ -f "$MANIFEST" ]] || die "missing provenance manifest: $MANIFEST"
  LC_ALL=C awk -F '\t' -v transform="$TRANSFORM" '
    BEGIN {
      expected_member["aws-lc-sys-0.41.0/aws-lc/generated-src/ios-aarch64/crypto/fipsmodule/armv8-mont.S"] = 1
      expected_member["aws-lc-sys-0.41.0/aws-lc/generated-src/linux-x86_64/crypto/fipsmodule/x86_64-mont.S"] = 1
      expected_member["aws-lc-sys-0.41.0/aws-lc/generated-src/linux-x86_64/crypto/fipsmodule/x86_64-mont5.S"] = 1
      expected_member["aws-lc-sys-0.41.0/aws-lc/crypto/fipsmodule/bn/asm/armv8-mont.pl"] = 1
      expected_member["aws-lc-sys-0.41.0/aws-lc/crypto/fipsmodule/bn/asm/x86_64-mont.pl"] = 1
      expected_member["aws-lc-sys-0.41.0/aws-lc/crypto/fipsmodule/bn/asm/x86_64-mont5.pl"] = 1
      expected_member["aws-lc-sys-0.41.0/aws-lc/crypto/perlasm/arm-xlate.pl"] = 1
      expected_member["aws-lc-sys-0.41.0/aws-lc/crypto/perlasm/x86_64-xlate.pl"] = 1
      expected_output["src/auth/asm/rscrypto_rsa_bignum_mont_apple.s"] = 1
      expected_output["src/auth/asm/rscrypto_rsa_bignum_mont_aarch64_elf.s"] = 1
      expected_output["src/auth/asm/rscrypto_rsa_x86_64_elf.S"] = 1
    }
    /^#/ { next }
    $1 == "schema" {
      if (NF != 2 || $2 != "1") exit 2
      schema++
      next
    }
    $1 == "transform" {
      if (NF != 2 || $2 != transform) exit 2
      transforms++
      next
    }
    $1 == "archive" {
      if (NF != 4 || $2 != "aws-lc-sys-0.41.0" ||
          length($3) != 64 || $3 !~ /^[0-9a-f]+$/ ||
          length($4) != 40 || $4 !~ /^[0-9a-f]+$/) exit 2
      archives++
      next
    }
    $1 == "member" {
      if (NF != 3 || index($2, "aws-lc-sys-0.41.0/aws-lc/") != 1 ||
          length($3) != 64 || $3 !~ /^[0-9a-f]+$/ ||
          !($2 in expected_member) || seen_member[$2]++) exit 2
      members++
      next
    }
    $1 == "output" {
      if (NF != 5 || index($2, "src/auth/asm/rscrypto_rsa_") != 1 ||
          $3 !~ /^[1-9][0-9]*$/ || $4 !~ /^[1-9][0-9]*$/ ||
          length($5) != 64 || $5 !~ /^[0-9a-f]+$/ ||
          !($2 in expected_output) || seen_output[$2]++) exit 2
      outputs++
      next
    }
    { exit 2 }
    END {
      if (schema != 1 || transforms != 1 || archives != 1 || members != 8 || outputs != 3) exit 2
      for (path in expected_member) if (!(path in seen_member)) exit 2
      for (path in expected_output) if (!(path in seen_output)) exit 2
    }
  ' "$MANIFEST" || die "invalid provenance manifest"
  [[ "$(hash_file "$MANIFEST")" == "$TRANSFORM_MANIFEST_SHA256" ]] \
    || die "provenance manifest changed without a new transform identity"
}

verify_external_rsa_coverage() {
  local manifest_outputs external_outputs

  manifest_outputs="$(
    awk -F '\t' '$1 == "output" { print $2 }' "$MANIFEST" |
      LC_ALL=C sort
  )"
  external_outputs="$(
    find "$REPO_ROOT/src" -type f \
      \( -name '*.s' -o -name '*.S' \) |
      while IFS= read -r path; do
        relative=${path#"$REPO_ROOT/"}
        if grep -Eq \
          '^(//|[[:space:]]*\*) Adapted for rscrypto|^[[:space:]]*\* The butterfly schedule is auto-derived from' \
          "$path" &&
          {
            grep -Eq '(^|/)rsa([/_.]|$)|_rsa([_.]|$)' <<<"$relative" ||
              grep -Eq \
                '^[[:space:]]*\.(globl|global)[[:space:]]+_?rscrypto_rsa_[A-Za-z0-9_]+([[:space:]]|$)|^_?rscrypto_rsa_[A-Za-z0-9_]+:' \
                "$path"
          }; then
          printf '%s\n' "$relative"
        fi
      done |
      LC_ALL=C sort
  )"
  [[ "$external_outputs" == "$manifest_outputs" ]] \
    || die "external-derived RSA assembly does not exactly match manifest outputs"
}

verify_rsa_wrapper_fingerprints() {
  local wrapper expected actual

  while IFS=$'\t' read -r wrapper expected; do
    actual=$(hash_file "$REPO_ROOT/$wrapper")
    [[ "$actual" == "$expected" ]] \
      || die "$wrapper changed without a provenance review"
  done <<'EOF'
src/auth/rsa_aarch64_asm.rs	2f3207e455cd800cd0ae8c833f1f743b2f830e004e65eea2f8099db28edbb846
src/auth/rsa_aarch64_linux_asm.rs	4cc4619203bab93baeb87511924a65853a83773ce5227d6caf94019f00a87d4a
src/auth/rsa_x86_64_asm.rs	7c610b07b75a25900393efaf6e23f817edc756dbbb3459daa4280a79546d7c61
EOF
}

verify_rsa_wrapper_coverage() {
  local manifest_outputs wrapper_targets wrapper include path

  manifest_outputs="$(
    awk -F '\t' '$1 == "output" { print $2 }' "$MANIFEST" |
      LC_ALL=C sort
  )"
  wrapper_targets="$(
    for wrapper in \
      "$REPO_ROOT/src/auth/rsa_aarch64_asm.rs" \
      "$REPO_ROOT/src/auth/rsa_aarch64_linux_asm.rs" \
      "$REPO_ROOT/src/auth/rsa_x86_64_asm.rs"; do
      [[ -f "$wrapper" ]] || die "missing RSA assembly wrapper: $wrapper"
      while IFS= read -r include; do
        [[ "$include" != /* && "$include" != *..* ]] \
          || die "unsupported RSA assembly include path: $include"
        path="$REPO_ROOT/src/auth/$include"
        [[ -f "$path" ]] || die "missing RSA assembly include target: $path"
        if grep -Eq \
          '^(//|[[:space:]]*\*) Adapted for rscrypto|^[[:space:]]*\* The butterfly schedule is auto-derived from' \
          "$path"; then
          printf 'src/auth/%s\n' "$include"
        fi
      done < <(
        awk '
          {
            line = $0
            while (match(line, /include_str!\("[^"]+"\)/)) {
              token = substr(line, RSTART, RLENGTH)
              sub(/^include_str!\("/, "", token)
              sub(/"\)$/, "", token)
              print token
              line = substr(line, RSTART + RLENGTH)
            }
          }
        ' "$wrapper"
      )
    done |
      LC_ALL=C sort
  )"
  [[ "$wrapper_targets" == "$manifest_outputs" ]] \
    || die "external RSA assembly wrapper targets do not exactly match manifest outputs"
}

verify_committed_outputs() {
  while IFS=$'\t' read -r kind path lines bytes digest; do
    [[ "$kind" == "output" ]] || continue
    case "$path" in
      src/auth/asm/rscrypto_rsa_bignum_mont_apple.s \
        | src/auth/asm/rscrypto_rsa_bignum_mont_aarch64_elf.s \
        | src/auth/asm/rscrypto_rsa_x86_64_elf.S) ;;
      *) die "unexpected RSA provenance output: $path" ;;
    esac
    require_shape "$REPO_ROOT/$path" "$lines" "$bytes" "$digest"
  done <"$MANIFEST"
}

validate_manifest
verify_external_rsa_coverage
verify_rsa_wrapper_fingerprints
verify_rsa_wrapper_coverage
verify_committed_outputs

if [[ $# -eq 0 ]]; then
  printf 'rsa-asm-provenance: committed RSA assembly matches the provenance manifest\n'
  exit 0
fi
[[ $# -eq 2 && $1 == "--archive" ]] || usage
ARCHIVE=$2
[[ -f "$ARCHIVE" ]] || die "archive is not a regular file: $ARCHIVE"
export LC_ALL=C

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rscrypto-rsa-asm.XXXXXXXX")"
[[ -d "$TEMP_DIR" ]] || die "failed to create temporary directory"
trap 'rm -rf -- "$TEMP_DIR"' EXIT

archive_digest=$(awk -F '\t' '$1 == "archive" { print $3 }' "$MANIFEST")
actual_archive_digest=$(hash_file "$ARCHIVE")
[[ "$actual_archive_digest" == "$archive_digest" ]] \
  || die "archive SHA-256 $actual_archive_digest does not match $archive_digest"

ARCHIVE_LIST="$TEMP_DIR/archive.list"
tar -tzf "$ARCHIVE" >"$ARCHIVE_LIST"

members=()
while IFS=$'\t' read -r kind member digest; do
  [[ "$kind" == "member" ]] || continue
  members+=("$member")
  exact_count=$(awk -v member="$member" '$0 == member { count++ } END { print count + 0 }' "$ARCHIVE_LIST")
  [[ "$exact_count" == 1 ]] || die "archive must contain exactly one $member"
  relative=${member#"$ARCHIVE_PREFIX"}
  suffix_count=$(awk -v suffix="/$relative" '
    length($0) >= length(suffix) && substr($0, length($0) - length(suffix) + 1) == suffix { count++ }
    END { print count + 0 }
  ' "$ARCHIVE_LIST")
  [[ "$suffix_count" == 1 ]] || die "archive contains required member under another prefix: $relative"
done <"$MANIFEST"
[[ ${#members[@]} -eq 8 ]] || die "manifest member count changed during extraction"

tar -xzf "$ARCHIVE" -C "$TEMP_DIR" "${members[@]}"
while IFS=$'\t' read -r kind member digest; do
  [[ "$kind" == "member" ]] || continue
  require_shape "$TEMP_DIR/$member" "$(wc -l <"$TEMP_DIR/$member" | tr -d ' ')" \
    "$(wc -c <"$TEMP_DIR/$member" | tr -d ' ')" "$digest"
done <"$MANIFEST"

SOURCE_ROOT="$TEMP_DIR/$ARCHIVE_PREFIX/aws-lc"
APPLE_INPUT="$SOURCE_ROOT/generated-src/ios-aarch64/crypto/fipsmodule/armv8-mont.S"
X86_MONT_INPUT="$SOURCE_ROOT/generated-src/linux-x86_64/crypto/fipsmodule/x86_64-mont.S"
X86_MONT5_INPUT="$SOURCE_ROOT/generated-src/linux-x86_64/crypto/fipsmodule/x86_64-mont5.S"
require_lf_file "$APPLE_INPUT"
require_lf_file "$X86_MONT_INPUT"
require_lf_file "$X86_MONT5_INPUT"

APPLE_CLOSE="#endif  // !OPENSSL_NO_ASM && defined(OPENSSL_AARCH64) && defined(__APPLE__)"
APPLE_OUTER="#if !defined(OPENSSL_NO_ASM) && defined(OPENSSL_AARCH64) && defined(__APPLE__)"
APPLE_SIGNATURE=$'.byte\t77,111,110,116,103,111,109,101,114,121,32,77,117,108,116,105,112,108,105,99,97,116,105,111,110,32,102,111,114,32,65,82,77,118,56,44,32,67,82,89,80,84,79,71,65,77,83,32,98,121,32,60,97,112,112,114,111,64,111,112,101,110,115,115,108,46,111,114,103,62,0'

awk -v closing="$APPLE_CLOSE" -v outer="$APPLE_OUTER" -v signature="$APPLE_SIGNATURE" '
  function is_ident(c) { return c ~ /^[A-Za-z0-9_]$/ }
  function token_count(s, token, count, pos, before, after) {
    count = 0
    while ((pos = index(s, token)) != 0) {
      before = pos == 1 ? "" : substr(s, pos - 1, 1)
      after = substr(s, pos + length(token), 1)
      if ((before == "" || !is_ident(before)) && (after == "" || !is_ident(after))) count++
      s = substr(s, pos + length(token))
    }
    return count
  }
  $0 == outer { outer_count++; outer_seen = 1; next }
  $0 == ".text" {
    starts++
    if (!outer_seen || inside) exit 2
    inside = 1
  }
  $0 == closing {
    closes++
    if (!inside) exit 2
    inside = 0
    next
  }
  inside {
    lines++
    if ($0 ~ /^[[:space:]]*[.]cfi_/) cfi++
    if ($0 == "\tAARCH64_SIGN_LINK_REGISTER") sign++
    if ($0 == "\tAARCH64_VALIDATE_LINK_REGISTER") validate++
    if ($0 == "\t// Not adding AARCH64_SIGN_LINK_REGISTER here because __bn_sqr8x_mont is jumped to" ||
        $0 == "\t// Not adding AARCH64_SIGN_LINK_REGISTER here because __bn_mul4x_mont is jumped to") pac_comments++
    if (index($0, "AARCH64_SIGN_LINK_REGISTER") || index($0, "AARCH64_VALIDATE_LINK_REGISTER")) pac_text++
    if ($0 == "\tmov\tx0,#1") returns++
    bn_mul += token_count($0, "_bn_mul_mont")
    sqr += token_count($0, "__bn_sqr8x_mont")
    mul4 += token_count($0, "__bn_mul4x_mont")
    rewinded += gsub(/rewinded/, "&")
    only_which += gsub(/only from bn_mul_mont which/, "&")
    only_or += gsub(/only from bn_mul_mont or/, "&")
    if ($0 == "\tmov\tsp,x22\t\t\t// alloca") stack8++
    if ($0 == "\tsub\tx2,sp,x5,lsl#4") stack16++
    if ($0 == "\tsub\tx26,sp,x5,lsl#3") stack8b++
    last3 = last2
    last2 = last1
    last1 = $0
  }
  END {
    if (outer_count != 1 || starts != 1 || closes != 1 || inside ||
        lines != 1495 || cfi != 79 || sign != 1 || validate != 3 ||
        pac_comments != 2 || pac_text != 6 || returns != 3 ||
        bn_mul != 3 || sqr != 3 || mul4 != 4 || rewinded != 4 ||
        only_which != 1 || only_or != 1 || stack8 != 1 || stack16 != 1 || stack8b != 1 ||
        last3 != signature || last2 != ".align\t2" || last1 != ".align\t4") exit 2
  }
' "$APPLE_INPUT" || die "Apple AArch64 transform preflight failed"

APPLE_BODY="$TEMP_DIR/apple.body"
awk -v closing="$APPLE_CLOSE" '
  function is_ident(c) { return c ~ /^[A-Za-z0-9_]$/ }
  function replace_token(s, old, new, out, pos, before, after) {
    out = ""
    while ((pos = index(s, old)) != 0) {
      before = pos == 1 ? "" : substr(s, pos - 1, 1)
      after = substr(s, pos + length(old), 1)
      if ((before == "" || !is_ident(before)) && (after == "" || !is_ident(after))) {
        out = out substr(s, 1, pos - 1) new
        s = substr(s, pos + length(old))
        replacements++
      } else {
        out = out substr(s, 1, pos)
        s = substr(s, pos + 1)
      }
    }
    return out s
  }
  $0 == ".text" { inside = 1 }
  $0 == closing { inside = 0; next }
  !inside { next }
  /^[[:space:]]*[.]cfi_/ { cfi++; next }
  $0 == "\tAARCH64_SIGN_LINK_REGISTER" || $0 == "\tAARCH64_VALIDATE_LINK_REGISTER" { pac++; next }
  $0 == "\t// Not adding AARCH64_SIGN_LINK_REGISTER here because __bn_sqr8x_mont is jumped to" ||
    $0 == "\t// Not adding AARCH64_SIGN_LINK_REGISTER here because __bn_mul4x_mont is jumped to" {
      pac_comments++
      next
    }
  {
    line = $0
    if (line == "\tmov\tx0,#1") {
      line = "\t// No return value"
      returns++
    }
    which += gsub(/only from bn_mul_mont which/, "only from bn_mul_mont_words which", line)
    only_or += gsub(/only from bn_mul_mont or/, "only from bn_mul_mont_words or", line)
    rewinded += gsub(/rewinded/, "rewound", line)
    before = replacements
    line = replace_token(line, "_bn_mul_mont", "_rscrypto_rsa_bn_mul_mont_words_apple")
    bn_mul += replacements - before
    before = replacements
    line = replace_token(line, "__bn_sqr8x_mont", "Lrscrypto_rsa_bn_sqr8x_mont")
    sqr += replacements - before
    before = replacements
    line = replace_token(line, "__bn_mul4x_mont", "Lrscrypto_rsa_bn_mul4x_mont")
    mul4 += replacements - before
    if (line == "\tmov\tsp,x22\t\t\t// alloca") {
      print "\t// This can allocate at most 8 * BN_MONTGOMERY_MAX_WORDS on the stack,"
      print "\t// or 2 KiB. This fits well within a page, so it is not necessary to"
      print "\t// fault pages in the correct order."
      insert8++
    }
    if (line == "\tsub\tx2,sp,x5,lsl#4") {
      print "\t// This can allocate at most 16 * BN_MONTGOMERY_MAX_WORDS on the stack,"
      print "\t// or 4 KiB. The fixed allocation above pushes to just above a page. On"
      print "\t// Windows, we must ensure new pages are first accessed in order. See"
      print "\t// https://learn.microsoft.com/en-us/cpp/build/arm64-windows-abi-conventions?view=msvc-170#stack"
      print "\t//"
      print "\t// The order is correct, but precariously so: the code above access as"
      print "\t// low as [sp,#16]. This leaves a jump of 16 + 4096 = 4112 bytes. If"
      print "\t// [sp,#16] were at page boundary, those 4112 bytes would span two"
      print "\t// pages. If [x2] were the next access, we would skip a guard page."
      print "\t//"
      print "\t// Fortunately, the first access is [x2,#8*8], at .Lsqr8x_zero_start."
      print "\t// We jump at most 4112 - 64 = 4048 bytes, less than a page. If any of"
      print "\t// this changes, we must insert a no-op access or call __chkstk."
      insert16++
    }
    if (line == "\tsub\tx26,sp,x5,lsl#3") {
      print "\t// This can allocate at most 8 * BN_MONTGOMERY_MAX_WORDS on the stack,"
      print "\t// or 2 KiB. This fits well within a page, so it is not necessary to"
      print "\t// fault pages in the correct order."
      insert8b++
    }
    print line
  }
  END {
    if (inside || cfi != 79 || pac != 4 || pac_comments != 2 || returns != 3 ||
        which != 1 || only_or != 1 || rewinded != 4 || bn_mul != 3 || sqr != 2 || mul4 != 3 ||
        insert8 != 1 || insert16 != 1 || insert8b != 1) exit 2
  }
' "$APPLE_INPUT" >"$APPLE_BODY" || die "Apple AArch64 transform failed"
require_shape "$APPLE_BODY" 1429 31667 1238c64546b882e5763689d6e8c7bb38999341cc2bfd8a344a4f044722cb9669

APPLE_OUTPUT="$TEMP_DIR/rscrypto_rsa_bignum_mont_apple.s"
{
  printf '%s\n' \
    '// Copyright 2015-2016 The OpenSSL Project Authors. All Rights Reserved.' \
    '// SPDX-License-Identifier: Apache-2.0' \
    '//' \
    '// Adapted for rscrypto from BoringSSL generated armv8-mont-apple.S.' \
    '// The public symbol is renamed into the rscrypto namespace and embedded with Rust global_asm!.' \
    '' \
    '// This file is generated from a similarly-named Perl script in the BoringSSL' \
    '// source tree. Do not edit by hand.' \
    '' \
    ''
  awk '{ print }' "$APPLE_BODY"
} >"$APPLE_OUTPUT"
require_shape "$APPLE_OUTPUT" 1439 32065 3e723bd775c6e216d9525d470d6d082a83e989f7988db0355c0ad9cd5bfa5072
cmp -s "$APPLE_OUTPUT" "$REPO_ROOT/src/auth/asm/rscrypto_rsa_bignum_mont_apple.s" \
  || die "reconstructed Apple assembly differs from the committed snapshot"

AARCH64_ELF_OUTPUT="$TEMP_DIR/rscrypto_rsa_bignum_mont_aarch64_elf.s"
awk '
  function is_ident(c) { return c ~ /^[A-Za-z0-9_]$/ }
  function replace_token(s, old, new, out, pos, before, after) {
    out = ""
    while ((pos = index(s, old)) != 0) {
      before = pos == 1 ? "" : substr(s, pos - 1, 1)
      after = substr(s, pos + length(old), 1)
      if ((before == "" || !is_ident(before)) && (after == "" || !is_ident(after))) {
        out = out substr(s, 1, pos - 1) new
        s = substr(s, pos + length(old))
        replacements++
      } else {
        out = out substr(s, 1, pos)
        s = substr(s, pos + 1)
      }
    }
    return out s
  }
  {
    line = $0
    if (line == "// Adapted for rscrypto from BoringSSL generated armv8-mont-apple.S.") {
      line = "// Adapted for rscrypto from BoringSSL generated armv8-mont-apple.S and retargeted for ELF."
      header++
    }
    before = replacements
    line = replace_token(line, "_rscrypto_rsa_bn_mul_mont_words_apple", "rscrypto_rsa_bn_mul_mont_words_aarch64_elf")
    symbol += replacements - before
    before = replacements
    line = replace_token(line, ".private_extern", ".hidden")
    directive += replacements - before
    print line
  }
  END {
    if (header != 1 || symbol != 3 || directive != 1) exit 2
  }
' "$APPLE_OUTPUT" >"$AARCH64_ELF_OUTPUT" || die "AArch64 ELF retarget transform failed"
require_shape "$AARCH64_ELF_OUTPUT" 1439 32095 f28fa8f4f02e0427288c717fc5666b7c07f0f8fc0c7ea5bb21721b1ffe3dc4dc
cmp -s "$AARCH64_ELF_OUTPUT" "$REPO_ROOT/src/auth/asm/rscrypto_rsa_bignum_mont_aarch64_elf.s" \
  || die "reconstructed AArch64 ELF assembly differs from the committed snapshot"

X86_OUTER="#if !defined(OPENSSL_NO_ASM) && defined(OPENSSL_X86_64) && defined(__ELF__)"
X86_SIGNATURE=$'.byte\t77,111,110,116,103,111,109,101,114,121,32,77,117,108,116,105,112,108,105,99,97,116,105,111,110,32,102,111,114,32,120,56,54,95,54,52,44,32,67,82,89,80,84,79,71,65,77,83,32,98,121,32,60,97,112,112,114,111,64,111,112,101,110,115,115,108,46,111,114,103,62,0'
awk -v outer="$X86_OUTER" -v signature="$X86_SIGNATURE" '
  function is_ident(c) { return c ~ /^[A-Za-z0-9_]$/ }
  function token_count(s, token, count, pos, before, after) {
    count = 0
    while ((pos = index(s, token)) != 0) {
      before = pos == 1 ? "" : substr(s, pos - 1, 1)
      after = substr(s, pos + length(token), 1)
      if ((before == "" || !is_ident(before)) && (after == "" || !is_ident(after))) count++
      s = substr(s, pos + length(token))
    }
    return count
  }
  $0 == outer { outer_count++; outer_seen = 1; next }
  $0 == ".text\t" {
    starts++
    if (!outer_seen || inside) exit 2
    inside = 1
  }
  inside {
    lines++
    if ($0 == "_CET_ENDBR") cet++
    if ($0 == "#ifndef MY_ASSEMBLER_IS_TOO_OLD_FOR_512AVX") {
      if (guard_depth != 0) exit 2
      guard_depth = 1
      guards++
    }
    if ($0 == "#endif") {
      if (guard_depth != 1) exit 2
      guard_depth = 0
      guard_ends++
    }
    nohw += token_count($0, "bn_mul_mont_nohw")
    mul4 += token_count($0, "bn_mul4x_mont")
    sqr8 += token_count($0, "bn_sqr8x_mont")
    mulx4 += token_count($0, "bn_mulx4x_mont")
    sqrx += token_count($0, "bn_sqrx8x_internal")
    sqr += token_count($0, "bn_sqr8x_internal")
    if ($0 == signature) {
      signatures++
      after_signature = 1
    } else if (after_signature && $0 == ".align\t16") {
      if (guard_depth != 0) exit 2
      terminals++
      inside = 0
      expect_close = 1
    }
    next
  }
  expect_close {
    if ($0 != "#endif") exit 2
    outer_close++
    expect_close = 0
  }
  END {
    if (outer_count != 1 || starts != 1 || signatures != 1 || terminals != 1 ||
        outer_close != 1 || inside || expect_close || lines != 1236 ||
        guard_depth != 0 ||
        cet != 4 || guards != 3 || guard_ends != 3 ||
        nohw != 6 || mul4 != 6 || sqr8 != 6 || mulx4 != 6 || sqrx != 3 || sqr != 3) exit 2
  }
' "$X86_MONT_INPUT" || die "x86-64 mont transform preflight failed"

X86_MONT_BODY="$TEMP_DIR/x86-mont.body"
awk -v signature="$X86_SIGNATURE" '
  function is_ident(c) { return c ~ /^[A-Za-z0-9_]$/ }
  function replace_token(s, old, new, out, pos, before, after) {
    out = ""
    while ((pos = index(s, old)) != 0) {
      before = pos == 1 ? "" : substr(s, pos - 1, 1)
      after = substr(s, pos + length(old), 1)
      if ((before == "" || !is_ident(before)) && (after == "" || !is_ident(after))) {
        out = out substr(s, 1, pos - 1) new
        s = substr(s, pos + length(old))
        replacements++
      } else {
        out = out substr(s, 1, pos)
        s = substr(s, pos + 1)
      }
    }
    return out s
  }
  $0 == ".text\t" { inside = 1 }
  !inside { next }
  $0 == "_CET_ENDBR" { cet++; next }
  $0 == "#ifndef MY_ASSEMBLER_IS_TOO_OLD_FOR_512AVX" {
    if (guard_depth != 0) exit 2
    guard_depth = 1
    guards++
    next
  }
  $0 == "#endif" {
    if (guard_depth != 1) exit 2
    guard_depth = 0
    guards++
    next
  }
  {
    line = $0
    before = replacements
    line = replace_token(line, "bn_mul_mont_nohw", "rscrypto_rsa_bn_mul_mont_nohw_x86_64_elf")
    nohw += replacements - before
    before = replacements
    line = replace_token(line, "bn_mul4x_mont", "rscrypto_rsa_bn_mul4x_mont_x86_64_elf")
    mul4 += replacements - before
    before = replacements
    line = replace_token(line, "bn_sqr8x_mont", "rscrypto_rsa_bn_sqr8x_mont_x86_64_elf")
    sqr8 += replacements - before
    before = replacements
    line = replace_token(line, "bn_mulx4x_mont", "rscrypto_rsa_bn_mulx4x_mont_x86_64_elf")
    mulx4 += replacements - before
    before = replacements
    line = replace_token(line, "bn_sqrx8x_internal", "rscrypto_rsa_bn_sqrx8x_internal_x86_64_elf")
    sqrx += replacements - before
    before = replacements
    line = replace_token(line, "bn_sqr8x_internal", "rscrypto_rsa_bn_sqr8x_internal_x86_64_elf")
    sqr += replacements - before
    print line
    if ($0 == signature) after_signature = 1
    else if (after_signature && $0 == ".align\t16") {
      if (guard_depth != 0) exit 2
      inside = 0
    }
  }
  END {
    if (inside || guard_depth != 0 || cet != 4 || guards != 6 || nohw != 6 || mul4 != 6 ||
        sqr8 != 6 || mulx4 != 6 || sqrx != 3 || sqr != 3) exit 2
  }
' "$X86_MONT_INPUT" >"$X86_MONT_BODY" || die "x86-64 mont transform failed"
require_shape "$X86_MONT_BODY" 1226 20955 2e9a43f8690820be1c17aac0b68c6ae9092e09bc28775460d6b93334e4f7dd96

X86_MONT5_BODY="$TEMP_DIR/x86-mont5.body"
awk '
  function is_ident(c) { return c ~ /^[A-Za-z0-9_]$/ }
  function replace_token(s, old, new, out, pos, before, after) {
    out = ""
    while ((pos = index(s, old)) != 0) {
      before = pos == 1 ? "" : substr(s, pos - 1, 1)
      after = substr(s, pos + length(old), 1)
      if ((before == "" || !is_ident(before)) && (after == "" || !is_ident(after))) {
        out = out substr(s, 1, pos - 1) new
        s = substr(s, pos + length(old))
        replacements++
      } else {
        out = out substr(s, 1, pos)
        s = substr(s, pos + 1)
      }
    }
    return out s
  }
  $0 == "#if !defined(OPENSSL_NO_ASM) && defined(OPENSSL_X86_64) && defined(__ELF__)" { openings++; outer_depth++ }
  $0 == "#ifndef MY_ASSEMBLER_IS_TOO_OLD_FOR_512AVX" { openings++; my_depth++ }
  $0 == "#endif" {
    closings++
    if (my_depth > 0) my_depth--
    else if (outer_depth > 0) outer_depth--
  }
  $0 == ".globl\tbn_sqr8x_internal" {
    start1++
    if (my_depth != 0 || block) exit 2
    block = 1
  }
  $0 == ".globl\tbn_sqrx8x_internal" {
    start2++
    if (my_depth != 1 || block) exit 2
    block = 2
  }
  block {
    raw_lines++
    if (expect_alias) {
      expected_alias = block == 1 ? "__bn_sqr8x_internal:" : "__bn_sqrx8x_internal:"
      if ($0 != expected_alias) exit 2
      aliases++
      expect_alias = 0
      expect_cfi = 1
      next
    }
    if (expect_cfi) {
      if ($0 != ".cfi_startproc\t") exit 2
      expect_cfi = 0
      expect_cet = 1
    } else if (expect_cet) {
      if ($0 != "_CET_ENDBR") exit 2
      expect_cet = 0
      cet++
      skip_blanks = 1
      blank_count = 0
      next
    }
    if (skip_blanks) {
      if ($0 == "") {
        blank_count++
        next
      }
      expected_blanks = cet == 1 ? 73 : 40
      expected_anchor = cet == 1 ? "\tleaq\t32(%r10),%rbp" : "\tleaq\t48+8(%rsp),%rdi"
      if (blank_count != expected_blanks || $0 != expected_anchor) exit 2
      skip_blanks = 0
      blank_runs++
    }
    if ($0 == "__bn_sqr8x_internal:" || $0 == "__bn_sqrx8x_internal:") {
      exit 2
    }
    if ($0 == "_CET_ENDBR") {
      exit 2
    }
    if ($0 == "bn_sqr8x_internal:" || $0 == "bn_sqrx8x_internal:") {
      expected_public = block == 1 ? "bn_sqr8x_internal:" : "bn_sqrx8x_internal:"
      if ($0 != expected_public || public_labels == block) exit 2
      public_labels = block
      expect_alias = 1
    }
    line = $0
    before = replacements
    line = replace_token(line, "bn_sqr8x_internal", "rscrypto_rsa_bn_sqr8x_internal_x86_64_elf")
    sqr += replacements - before
    before = replacements
    line = replace_token(line, "bn_sqrx8x_internal", "rscrypto_rsa_bn_sqrx8x_internal_x86_64_elf")
    sqrx += replacements - before
    before = replacements
    line = replace_token(line, "__bn_sqr8x_reduction", "rscrypto_rsa_bn_sqr8x_reduction_x86_64_elf")
    sqr_red += replacements - before
    before = replacements
    line = replace_token(line, "__bn_sqrx8x_reduction", "rscrypto_rsa_bn_sqrx8x_reduction_x86_64_elf")
    sqrx_red += replacements - before
    print line
    if ($0 == ".size\tbn_sqr8x_internal,.-bn_sqr8x_internal") {
      end1++
      block = 0
    } else if ($0 == ".size\tbn_sqrx8x_internal,.-bn_sqrx8x_internal") {
      end2++
      block = 0
    }
  }
  END {
    if (openings != 5 || closings != 5 || outer_depth || my_depth || block ||
        start1 != 1 || end1 != 1 || start2 != 1 || end2 != 1 || raw_lines != 1407 ||
        public_labels != 2 || aliases != 2 || cet != 2 || blank_runs != 2 || skip_blanks ||
        expect_alias || expect_cfi || expect_cet ||
        sqr != 7 || sqrx != 7 || sqr_red != 1 || sqrx_red != 1) exit 2
  }
' "$X86_MONT5_INPUT" >"$X86_MONT5_BODY" || die "x86-64 mont5 transform failed"
require_shape "$X86_MONT5_BODY" 1290 21372 5baa0aa7f587bee6049d66e1d148b31f73ff6a4cd5a4864310770d7c854d9552

X86_OUTPUT="$TEMP_DIR/rscrypto_rsa_x86_64_elf.S"
{
  printf '%s\n' \
    '// Copyright 2005-2016 The OpenSSL Project Authors. All Rights Reserved.' \
    '// SPDX-License-Identifier: Apache-2.0' \
    '//' \
    '// Adapted for rscrypto from AWS-LC/BoringSSL generated x86_64-mont.S and x86_64-mont5.S.' \
    '// Public symbols are renamed into the rscrypto namespace and embedded with Rust global_asm!.' \
    '' \
    '// This file is generated from a similarly-named Perl script in the BoringSSL' \
    '// source tree. Do not edit by hand.' \
    '' \
    ''
  awk '{ print }' "$X86_MONT_BODY"
  printf '\n%s\n' '// Square internals required by rscrypto_rsa_bn_sqr8x_mont_x86_64_elf.'
  awk '{ print }' "$X86_MONT5_BODY"
  printf '\n%s\n' '.section .note.GNU-stack,"",@progbits'
} >"$X86_OUTPUT"

if grep -Eq '(^|[^A-Za-z0-9_])(bn_mul_mont_nohw|bn_mul4x_mont|bn_sqr8x_mont|bn_mulx4x_mont|bn_sqrx8x_internal|bn_sqr8x_internal)([^A-Za-z0-9_]|$)|^_CET_ENDBR$|^#(if|endif)' "$X86_OUTPUT"; then
  die "x86-64 postflight found an old token, CET marker, or preprocessor guard"
fi
require_shape "$X86_OUTPUT" 2530 42855 1b59e82724d10b06decbc8a9fbc88f9e04cefdee0524ef752c37ccf422f729a3
cmp -s "$X86_OUTPUT" "$REPO_ROOT/src/auth/asm/rscrypto_rsa_x86_64_elf.S" \
  || die "reconstructed x86-64 assembly differs from the committed snapshot"

printf 'rsa-asm-provenance: archive, members, transforms, and committed snapshots verified\n'
