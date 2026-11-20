#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../lib/fuzz.sh
source "$SCRIPT_DIR/../lib/fuzz.sh"

RUSTUP_TOOLCHAIN=$("$SCRIPT_DIR/../lib/toolchain.sh")
export RUSTUP_TOOLCHAIN
export CARGO_RAIL_CACHE=off

usage() {
  cat >&2 <<EOF
Usage: scripts/test/test-fuzz-asan.sh

Replays committed fuzz corpora under AddressSanitizer.

Environment:
  RSCRYPTO_FUZZ_CORPUS        committed (default) or local (include discoveries)
  RSCRYPTO_ASAN_TARGET_DIR     Cargo target dir for ASan builds
EOF
}

case "${1:-}" in
  '') ;;
  -h|--help) usage; exit 0 ;;
  *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
esac

if ! cargo fuzz --version &>/dev/null; then
  echo "cargo-fuzz is required for ASan fuzz replay" >&2
  exit 1
fi

export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:abort_on_error=1:strict_init_order=1}"
export RUSTFLAGS="${RUSTFLAGS:-} -Zsanitizer=address -Cforce-frame-pointers=yes"
# The rebuilt AArch64 std calls an instrumented outline-atomic helper during
# __asan_init, before shadow memory exists. Use inline LL/SC atomics so the
# standard library and corpus tests can retain their address instrumentation.
if [[ "$(fuzz_host_target)" == "aarch64-unknown-linux-gnu" ]]; then
  export RUSTFLAGS="$RUSTFLAGS -Ctarget-feature=-outline-atomics"
fi
export RUSTDOCFLAGS="${RUSTDOCFLAGS:-} -Zsanitizer=address"
export CARGO_TARGET_DIR="${RSCRYPTO_ASAN_TARGET_DIR:-$FUZZ_ROOT/target-asan}"

targets=$(fuzz_list_targets) || exit 2
missing=0
while IFS= read -r target; do
  [[ -z "$target" ]] && continue
  if [[ ! -d "$FUZZ_ROOT/corpus/$target" ]]; then
    echo "missing ASan corpus: $target ($FUZZ_ROOT/corpus/$target)" >&2
    missing=1
  fi
done <<<"$targets"
[[ "$missing" -eq 0 ]] || { echo "ASan fuzz corpus replay failed" >&2; exit 1; }

# The library under test builds at opt-level 1, the usual level for ASan. Unoptimized, the
# unrolled AArch64 Keccak rounds keep every temporary in memory, and the SLH-DSA replay alone
# exceeded the CI job limit. Debug assertions and overflow checks stay on.
cargo_test=(cargo test
  --locked
  -Zbuild-std
  --config 'profile.test.package.rscrypto.opt-level=1'
  --target "$(fuzz_host_target)"
  --manifest-path "$FUZZ_ROOT/Cargo.toml"
  --all-features
  --test corpus_replay)
replay_list=$("${cargo_test[@]}" -- --list --format terse) || { echo "ASan fuzz corpus replay failed" >&2; exit 1; }
expected=$(printf '%s\n' "$targets" | sed '/^$/d; s/^/replay_/; s/$/_corpus: test/' | LC_ALL=C sort)
actual=$(printf '%s\n' "$replay_list" | sed -n '/: test$/p' | LC_ALL=C sort)
if [[ "$expected" != "$actual" ]]; then
  echo "ASan replay inventory mismatch" >&2
  diff -u <(printf '%s\n' "$expected") <(printf '%s\n' "$actual") >&2 || true
  exit 1
fi
"${cargo_test[@]}" -- --include-ignored --nocapture || { echo "ASan fuzz corpus replay failed" >&2; exit 1; }
total=$(printf '%s\n' "$actual" | wc -l | tr -d ' ')
echo "ASan fuzz corpus replay passed for $total targets"
