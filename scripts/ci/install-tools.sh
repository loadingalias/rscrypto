#!/usr/bin/env bash
# Install cargo tools for CI
# Usage: install-tools.sh [standard|namespace|runson|bench|minimal]

set -euo pipefail

MODE="${1:-standard}"

echo "Installing cargo tools (mode: $MODE)"

# Prefer cargo-installed tools over any preinstalled runner tools.
export PATH="$HOME/.cargo/bin:$PATH"

# Check if cargo-binstall is available
install_binstall() {
    if command -v cargo-binstall &>/dev/null; then
        echo "cargo-binstall already installed"
        return 0
    fi

    echo "Installing cargo-binstall..."

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
        # Use upstream bootstrap script for other platforms
        curl -L --proto '=https' --tlsv1.2 -sSf \
            https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash
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

        if [ "$major" -gt "$min_major" ] 2>/dev/null || \
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
    standard|namespace|runson)
        # Standard CI tools (same for all CI runners)
        install_if_missing "cargo-nextest" "cargo-nextest"
        install_if_missing "cargo-rail" "cargo-rail"
        install_if_missing "cargo-deny" "cargo-deny"
        install_cargo_audit  # version-checked (requires 0.22+ for CVSS 4.0)
        install_if_missing "just" "just"
        ;;

    bench|runson-bench)
        # Benchmark tools (Criterion + tuning)
        install_if_missing "cargo-criterion" "cargo-criterion"
        install_if_missing "critcmp" "critcmp"
        install_if_missing "just" "just"
        ;;

    minimal)
        # Minimal set for quick jobs
        install_if_missing "just" "just"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: install-tools.sh [standard|namespace|runson|bench|runson-bench|minimal]"
        exit 1
        ;;
esac

echo ""
echo "Tool installation complete"
