#!/usr/bin/env bash
# Install cargo tools for CI
# Usage: install-tools.sh [standard|namespace|bench|minimal]

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

# Install cargo-binstall first (required for fast installs)
install_binstall

echo ""
echo "Installing tools for mode: $MODE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

case "$MODE" in
    standard)
        # Standard CI tools for commit workflow
        install_if_missing "cargo-nextest" "cargo-nextest"
        install_if_missing "cargo-deny" "cargo-deny"
        install_if_missing "cargo-audit" "cargo-audit"
        install_if_missing "just" "just"
        ;;

    namespace)
        # Namespace runners - same as standard but may have different caching
        install_if_missing "cargo-nextest" "cargo-nextest"
        install_if_missing "cargo-deny" "cargo-deny"
        install_if_missing "cargo-audit" "cargo-audit"
        install_if_missing "just" "just"
        ;;

    bench)
        # Benchmark tools
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
        echo "Usage: install-tools.sh [standard|namespace|bench|minimal]"
        exit 1
        ;;
esac

echo ""
echo "Tool installation complete"
