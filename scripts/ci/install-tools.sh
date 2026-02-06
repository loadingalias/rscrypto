#!/usr/bin/env bash
# Install cargo tools for CI
# Usage: install-tools.sh [standard|namespace|runson|bench|ibm|minimal|none]

set -euo pipefail

MODE="${1:-standard}"

echo "Installing cargo tools (mode: $MODE)"

# Tool-free jobs: do nothing (fast path).
if [[ "$MODE" == "none" ]]; then
	echo "Skipping tool installation (mode: none)"
	exit 0
fi

# Prefer cargo-installed tools over any preinstalled runner tools.
export PATH="$HOME/.cargo/bin:$PATH"

# Check if cargo-binstall is available
install_binstall() {
	if command -v cargo-binstall &>/dev/null; then
		echo "cargo-binstall already installed"
		return 0
	fi

	echo "Installing cargo-binstall..."

	# Downloads and installs cargo-binstall from a GitHub release.
	# Uses a subshell to isolate the EXIT trap and temporary directory handling.
	install_binstall_from_release() (
		set -euo pipefail
		local target="$1"
		local tmpdir
		tmpdir="$(mktemp -d)"
		trap 'rm -rf "$tmpdir"' EXIT

		local base_url
		if [[ -n "${BINSTALL_VERSION:-}" ]]; then
			base_url="https://github.com/cargo-bins/cargo-binstall/releases/download/v${BINSTALL_VERSION}/cargo-binstall-"
		else
			base_url="https://github.com/cargo-bins/cargo-binstall/releases/latest/download/cargo-binstall-"
		fi

		local url="${base_url}${target}.tgz"

		echo "  trying $target"
		cd "$tmpdir"

		# Avoid `curl | tar` here: with `set -euo pipefail`, a 404 can terminate the whole script
		# on some bash versions even when the function is invoked under `if ...; then`.
		if ! curl -L --proto '=https' --tlsv1.2 -sSf -o cargo-binstall.tgz "$url"; then
			exit 1
		fi
		tar -xzf cargo-binstall.tgz
		mkdir -p "$HOME/.cargo/bin"
		mv cargo-binstall "$HOME/.cargo/bin/"
		chmod +x "$HOME/.cargo/bin/cargo-binstall"
	)

	# Detect Windows ARM64 specially - uname -m returns x86_64 due to emulation layer
	# PROCESSOR_ARCHITECTURE is the reliable way to detect native arch on Windows
	if [[ "${PROCESSOR_ARCHITECTURE:-}" == "ARM64" ]]; then
		echo "Detected Windows ARM64 (via PROCESSOR_ARCHITECTURE)"
		BINSTALL_URL="https://github.com/cargo-bins/cargo-binstall/releases/latest/download/cargo-binstall-aarch64-pc-windows-msvc.zip"
		BINSTALL_ZIP="cargo-binstall-aarch64-pc-windows-msvc.zip"

		# Download and extract manually
		curl -L --proto '=https' --tlsv1.2 -sSf -o "$BINSTALL_ZIP" "$BINSTALL_URL"
		unzip -q "$BINSTALL_ZIP"
		mkdir -p "$HOME/.cargo/bin"
		mv cargo-binstall.exe "$HOME/.cargo/bin/"
		rm -f "$BINSTALL_ZIP"
		echo "✅ cargo-binstall installed (Windows ARM64)"
	else
		# The upstream bootstrap script tries `*-unknown-linux-musl` on many arches;
		# cargo-binstall doesn't publish MUSL binaries for all CI targets (notably s390x/ppc64le).
		# Prefer a release binary when available; otherwise fall back to `cargo install`.

		local os machine
		local -a targets
		os="$(uname -s)"
		machine="$(uname -m)"

		case "$os" in
		Linux)
			case "$machine" in
			x86_64) targets=(x86_64-unknown-linux-musl x86_64-unknown-linux-gnu) ;;
			aarch64) targets=(aarch64-unknown-linux-musl aarch64-unknown-linux-gnu) ;;
			armv7l) targets=(armv7-unknown-linux-musleabihf armv7-unknown-linux-gnueabihf) ;;
			s390x) targets=(s390x-unknown-linux-gnu) ;;
			ppc64le | powerpc64le) targets=(powerpc64le-unknown-linux-gnu) ;;
			*) targets=() ;;
			esac
			;;
		Darwin)
			case "$machine" in
			x86_64) targets=(x86_64-apple-darwin) ;;
			arm64) targets=(aarch64-apple-darwin) ;;
			*) targets=() ;;
			esac
			;;
		*)
			targets=()
			;;
		esac

		for t in "${targets[@]}"; do
			if install_binstall_from_release "$t"; then
				echo "✅ cargo-binstall installed ($t)"
				return 0
			fi
		done

		echo "  no prebuilt cargo-binstall for ${os}/${machine}; building from source..."
		if [[ -n "${BINSTALL_VERSION:-}" ]]; then
			cargo install cargo-binstall --locked --version "${BINSTALL_VERSION}"
		else
			cargo install cargo-binstall --locked
		fi
	fi
}

# Install a tool if not already present
install_if_missing() {
	local tool="$1"
	local binary="${2:-$tool}"

	if command -v "$binary" &>/dev/null; then
		echo "  $tool: cached"
		return 0
	fi

	echo "  $tool: installing..."
	cargo binstall "$tool" --no-confirm --force 2>/dev/null || cargo install "$tool" --locked
}

# Install cargo-audit with version validation (requires 0.22+ for CVSS 4.0)
install_cargo_audit() {
	local min_major=0
	local min_minor=22

	if command -v cargo-audit &>/dev/null; then
		# Extract version: "cargo-audit 0.20.1" -> "0.20.1"
		local version
		version=$(cargo-audit --version 2>/dev/null | awk '{print $2}')
		local major minor
		major=$(echo "$version" | cut -d. -f1)
		minor=$(echo "$version" | cut -d. -f2)

		if [ "$major" -gt "$min_major" ] 2>/dev/null ||
			{ [ "$major" -eq "$min_major" ] && [ "$minor" -ge "$min_minor" ]; } 2>/dev/null; then
			echo "  cargo-audit: cached (v$version >= 0.22)"
			return 0
		fi

		echo "  cargo-audit: outdated (v$version < 0.22), upgrading..."
	else
		echo "  cargo-audit: installing..."
	fi

	# NOTE: On some runners (notably Windows ARM64), `cargo binstall` may not have a matching binary.
	# If `cargo install` runs without `--force` and an old cargo-audit is already present (image/cache),
	# Cargo will *not* replace it; that leads to CVSS 4.0 parse failures when the advisory-db updates.
	cargo binstall cargo-audit --no-confirm --force 2>/dev/null || cargo install cargo-audit --locked --force
}

# Install cargo-binstall first (required for fast installs)
install_binstall

echo ""
echo "Installing tools for mode: $MODE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

case "$MODE" in
standard | namespace | runson)
	# Standard CI tools (same for all CI runners)
	install_if_missing "cargo-nextest" "cargo-nextest"
	install_if_missing "cargo-rail" "cargo-rail"
	install_if_missing "cargo-deny" "cargo-deny"
	install_cargo_audit # version-checked (requires 0.22+ for CVSS 4.0)
	install_if_missing "actionlint" "actionlint"
	install_if_missing "just" "just"
	;;

ibm | ci-lite)
	# IBM runners: only what our IBM CI actually uses (ci-check + test --all).
	# - ci-check requires: cargo-deny, cargo-audit
	# - test --all requires: cargo-nextest
	install_if_missing "cargo-nextest" "cargo-nextest"
	install_if_missing "cargo-deny" "cargo-deny"
	install_cargo_audit # version-checked (requires 0.22+ for CVSS 4.0)
	install_if_missing "actionlint" "actionlint"
	install_if_missing "just" "just"
	;;

bench | runson-bench)
	# Benchmark tools (Criterion + tuning)
	install_if_missing "cargo-criterion" "cargo-criterion"
	install_if_missing "critcmp" "critcmp"
	install_if_missing "just" "just"
	;;

fuzz)
	# Weekly fuzz lane: keep tool set minimal and explicit.
	install_if_missing "cargo-fuzz" "cargo-fuzz"
	install_if_missing "just" "just"
	;;

minimal)
	# Minimal set for quick jobs
	install_if_missing "just" "just"
	;;

none)
	# Tool-free jobs: use whatever is already present in the toolchain image.
	;;

*)
	echo "Unknown mode: $MODE"
	echo "Usage: install-tools.sh [standard|namespace|runson|bench|runson-bench|ibm|ci-lite|fuzz|minimal|none]"
	exit 1
	;;
esac

echo ""
echo "Tool installation complete"
