#!/usr/bin/env bash
set -euo pipefail

# Complete cross-platform check: host + windows + linux + constrained targets.
# Usage: check-all.sh [--all] [crate1 crate2 ...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/targets.sh
source "$SCRIPT_DIR/../lib/targets.sh"

DEFAULT_CONSTRAINED_CRATES=(
	"platform"
	"backend"
	"traits"
	"checksum"
	"hashes"
)

CONSTRAINED_CRATES=()
CONSTRAINED_SCOPE=""

select_constrained_crates() {
	local all_flag=false
	local crates=()

	for arg in "$@"; do
		if [[ "$arg" == "--all" ]]; then
			all_flag=true
		else
			crates+=("$arg")
		fi
	done

	if [[ ${#crates[@]} -gt 0 ]]; then
		CONSTRAINED_CRATES=("${crates[@]}")
		CONSTRAINED_SCOPE="${crates[*]}"
		return 0
	fi

	if [[ "$all_flag" == true ]]; then
		CONSTRAINED_CRATES=("${DEFAULT_CONSTRAINED_CRATES[@]}")
		CONSTRAINED_SCOPE="workspace"
		return 0
	fi

	local affected
	affected="$(rail_plan_crates)"

	if [[ -z "$affected" ]]; then
		CONSTRAINED_CRATES=("${DEFAULT_CONSTRAINED_CRATES[@]}")
		CONSTRAINED_SCOPE="workspace (no changes)"
	else
		CONSTRAINED_CRATES=()
		for crate in $affected; do
			CONSTRAINED_CRATES+=("$crate")
		done
		CONSTRAINED_SCOPE="affected"
	fi
}

crate_supports_alloc() {
	local crate=$1
	local manifest="crates/$crate/Cargo.toml"
	[[ -f "$manifest" ]] && grep -q '^[[:space:]]*alloc[[:space:]]*=' "$manifest"
}

run_constrained_target() {
	local target=$1
	local log_dir=$2

	ensure_target "$target"

	local target_dir="target/cross-check/$target"
	mkdir -p "$target_dir"

	local log_file="$log_dir/$target.log"
	: >"$log_file"

	step "$target check (no features)"
	for crate in "${CONSTRAINED_CRATES[@]}"; do
		if ! RUSTC_WRAPPER="" CARGO_TARGET_DIR="$target_dir" \
			cargo check -p "$crate" --no-default-features --target "$target" --lib \
			>>"$log_file" 2>&1; then
			fail
			show_error "$log_file"
			return 1
		fi
	done
	ok

	local alloc_crates=()
	for crate in "${CONSTRAINED_CRATES[@]}"; do
		if crate_supports_alloc "$crate"; then
			alloc_crates+=("$crate")
		fi
	done

	if [[ ${#alloc_crates[@]} -gt 0 ]]; then
		step "$target check (alloc)"
		for crate in "${alloc_crates[@]}"; do
			if ! RUSTC_WRAPPER="" CARGO_TARGET_DIR="$target_dir" \
				cargo check -p "$crate" --no-default-features --features alloc --target "$target" --lib \
				>>"$log_file" 2>&1; then
				fail
				show_error "$log_file"
				return 1
			fi
		done
		ok
	fi

	step "$target build (no features)"
	for crate in "${CONSTRAINED_CRATES[@]}"; do
		if ! RUSTC_WRAPPER="" CARGO_TARGET_DIR="$target_dir" \
			cargo build -p "$crate" --no-default-features --target "$target" --lib --release \
			>>"$log_file" 2>&1; then
			fail
			show_error "$log_file"
			return 1
		fi
	done
	ok

	if [[ ${#alloc_crates[@]} -gt 0 ]]; then
		step "$target build (alloc)"
		for crate in "${alloc_crates[@]}"; do
			if ! RUSTC_WRAPPER="" CARGO_TARGET_DIR="$target_dir" \
				cargo build -p "$crate" --no-default-features --features alloc --target "$target" --lib --release \
				>>"$log_file" 2>&1; then
				fail
				show_error "$log_file"
				return 1
			fi
		done
		ok
	fi
}

run_constrained_checks() {
	local log_dir
	log_dir=$(mktemp -d)
	trap 'rm -rf "$log_dir"' RETURN

	local constrained_targets=("${NOSTD_TARGETS[@]}" "${WASM_TARGETS[@]}")

	echo ""
	echo "Constrained targets ${DIM}($CONSTRAINED_SCOPE)${RESET}"

	if [[ ${#constrained_targets[@]} -eq 0 ]]; then
		skip "no constrained targets configured" "config/target-matrix.toml"
		return 0
	fi

	local pids=()
	local targets=()
	for i in "${!constrained_targets[@]}"; do
		target="${constrained_targets[$i]}"
		targets[$i]="$target"
		(
			run_constrained_target "$target" "$log_dir"
		) &
		pids[$i]=$!
	done

	local failures=0
	for i in "${!targets[@]}"; do
		target="${targets[$i]}"
		step "$target group"
		if wait "${pids[$i]}"; then
			ok
		else
			fail
			failures=1
		fi
	done

	if [[ $failures -ne 0 ]]; then
		return 1
	fi

	echo "${GREEN}✓${RESET} Constrained targets passed"
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Cross-platform checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

maybe_disable_sccache
select_constrained_crates "$@"

# Run host checks first (most likely to fail, fastest feedback)
"$SCRIPT_DIR/check.sh" "$@"
echo ""

echo "Cross targets ${DIM}(parallel)${RESET}"
log_dir=$(mktemp -d)
trap 'rm -rf "$log_dir"' EXIT

(
	"$SCRIPT_DIR/check-win.sh" "$@"
) >"$log_dir/windows.log" 2>&1 &
pid_windows=$!

(
	"$SCRIPT_DIR/check-linux.sh" "$@"
) >"$log_dir/linux.log" 2>&1 &
pid_linux=$!

(
	"$SCRIPT_DIR/check-ibm.sh" "$@"
) >"$log_dir/ibm.log" 2>&1 &
pid_ibm=$!

(
	run_constrained_checks
) >"$log_dir/constrained.log" 2>&1 &
pid_constrained=$!

failures=0
for job in windows linux ibm constrained; do
	pid_var="pid_${job}"
	log_file="$log_dir/$job.log"
	case "$job" in
	windows) display="Windows group" ;;
	linux) display="Linux group" ;;
	ibm) display="IBM group" ;;
	constrained) display="Constrained group" ;;
	esac
	step "$display"
	if wait "${!pid_var}"; then
		ok
	else
		fail
		show_error "$log_file"
		failures=1
	fi
done

if [[ $failures -ne 0 ]]; then
	exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "${GREEN}✓${RESET} All cross-platform checks passed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
