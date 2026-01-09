#!/usr/bin/env bash
# Install cargo tools for CI
# Usage: install-tools.sh [standard|namespace|runson|bench|minimal]

set -euo pipefail

MODE="${1:-standard}"

echo "Installing cargo tools (mode: $MODE)"

# Check if cargo-binstall is available
install_binstall() {
    if command -v cargo-binstall &>/dev/null; then
        echo "cargo-binstall already installed"
        return 0
    fi

    echo "Installing cargo-binstall..."
    curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash
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

# Install cargo-audit with version validation (requires 0.20+ for CVSS 4.0)
install_cargo_audit() {
    local min_major=0
    local min_minor=20

    if command -v cargo-audit &>/dev/null; then
        # Extract version: "cargo-audit 0.20.1" -> "0.20.1"
        local version
        version=$(cargo audit --version 2>/dev/null | awk '{print $2}')
        local major minor
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)

        if [ "$major" -gt "$min_major" ] 2>/dev/null || \
           { [ "$major" -eq "$min_major" ] && [ "$minor" -ge "$min_minor" ]; } 2>/dev/null; then
            echo "  cargo-audit: cached (v$version >= 0.20)"
            return 0
        fi

        echo "  cargo-audit: outdated (v$version < 0.20), upgrading..."
    else
        echo "  cargo-audit: installing..."
    fi

    cargo binstall cargo-audit --no-confirm --force 2>/dev/null || cargo install cargo-audit --locked
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
        install_cargo_audit  # version-checked (requires 0.20+ for CVSS 4.0)
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
