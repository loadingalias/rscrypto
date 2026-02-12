#!/usr/bin/env bash
set -euo pipefail

# IBM cross-compilation checks via zig (s390x, POWER)
# Usage: check-ibm.sh [--all] [crate1 crate2 ...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/targets.sh
source "$SCRIPT_DIR/../lib/targets.sh"

# Check for zig
if ! command -v zig >/dev/null 2>&1; then
	echo "IBM targets ${DIM}(skipped)${RESET}"
	skip "zig not installed" "brew install zig"
	exit 0
fi

# Parse args and set CRATE_FLAGS, SCOPE_DESC
get_crate_flags "$@"

export ZIG_CC="$SCRIPT_DIR/zig-cc.sh"

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT

echo "IBM targets ${DIM}($SCOPE_DESC)${RESET}"

if [[ ${#IBM_TARGETS[@]} -eq 0 ]]; then
	skip "no IBM targets configured" "config/target-matrix.toml"
	exit 0
fi

for target in "${IBM_TARGETS[@]}"; do
	ensure_target "$target"
done

for target in "${IBM_TARGETS[@]}"; do
	mkdir -p "target/cross-check/$target"
done

pids=()
logs=()
targets=()

for i in "${!IBM_TARGETS[@]}"; do
	target="${IBM_TARGETS[$i]}"
	target_dir="target/cross-check/$target"
	log_file="$LOG_DIR/$target.log"
	logs[$i]="$log_file"
	targets[$i]="$target"

	(
		# shellcheck disable=SC2086
		if ! CC="$ZIG_CC" RUSTC_WRAPPER="" CARGO_TARGET_DIR="$target_dir" \
			cargo clippy $CRATE_FLAGS --lib --all-features --target "$target" -- -D warnings \
			>"$log_file" 2>&1; then
			exit 1
		fi
	) &
	pids[$i]=$!
done

FAILED=0
for i in "${!targets[@]}"; do
	target="${targets[$i]}"
	short_name="${target/unknown-linux-/}" # s390x-gnu / powerpc64le-gnu

	step "$short_name clippy"
	if wait "${pids[$i]}"; then
		ok
	else
		fail
		show_error "${logs[$i]}"
		FAILED=1
	fi
done

if [ $FAILED -ne 0 ]; then
	exit 1
fi

echo "${GREEN}✓${RESET} IBM targets passed"
