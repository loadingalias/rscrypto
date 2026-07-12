#!/usr/bin/env bash
# Install cargo tools for CI.
# Usage: install-tools.sh [standard|rail|ci|supply-chain|bench|ibm|fuzz|coverage|ct-linux|minimal|none]

set -euo pipefail

MODE="${1:-standard}"
CARGO_RAIL_VERSION="${CARGO_RAIL_VERSION:-0.17.0}"
CARGO_SEMVER_CHECKS_VERSION="${CARGO_SEMVER_CHECKS_VERSION:-0.48.0}"

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

install_just_portable() {
  if command -v just &>/dev/null; then
    echo "  just: cached"
    return 0
  fi

  echo "  just: installing via cargo"
  cargo install just --locked
}

# Compare dot-separated numeric versions. Returns 0 when $1 >= $2.
version_gte() {
  local lhs="$1"
  local rhs="$2"
  local IFS=.
  local -a a b

  read -r -a a <<<"$lhs"
  read -r -a b <<<"$rhs"

  local i max len_a len_b
  len_a=${#a[@]}
  len_b=${#b[@]}
  max=$((len_a > len_b ? len_a : len_b))

  for ((i = 0; i < max; i++)); do
    local ai="${a[i]:-0}"
    local bi="${b[i]:-0}"
    if ((10#$ai > 10#$bi)); then
      return 0
    fi
    if ((10#$ai < 10#$bi)); then
      return 1
    fi
  done

  return 0
}

ensure_cargo_rail() {
  local required="$1"
  local installed=""

  if command -v cargo-rail &>/dev/null; then
    installed="$(cargo rail --version 2>/dev/null | awk '{print $2}' || true)"
    if [[ -n "$installed" ]] && version_gte "$installed" "$required"; then
      echo "  cargo-rail: cached ($installed)"
      return 0
    fi
    echo "  cargo-rail: stale ($installed), upgrading to >= $required"
  else
    echo "  cargo-rail: installing (required >= $required)"
  fi

  cargo binstall "cargo-rail@${required}" --no-confirm --force 2>/dev/null || cargo install cargo-rail --locked --version "$required" --force
}

ensure_cargo_semver_checks() {
  local required="$1"
  local installed=""

  if command -v cargo-semver-checks &>/dev/null; then
    installed="$(cargo-semver-checks --version 2>/dev/null | awk '{print $2}' || true)"
    if [[ -n "$installed" ]] && version_gte "$installed" "$required"; then
      echo "  cargo-semver-checks: cached ($installed)"
      return 0
    fi
    echo "  cargo-semver-checks: stale ($installed), upgrading to >= $required"
  else
    echo "  cargo-semver-checks: installing (required >= $required)"
  fi

  cargo binstall "cargo-semver-checks@${required}" --no-confirm --force 2>/dev/null || cargo install cargo-semver-checks --locked --version "$required" --force
}

install_binsec_system_packages() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    return 0
  fi
  if ! command -v apt-get &>/dev/null; then
    return 0
  fi
  if [[ "${BINSEC_SYSTEM_PACKAGES_READY:-}" == "1" ]]; then
    return 0
  fi

  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgmp-dev \
    libmpfr-dev \
    m4 \
    opam \
    pkg-config \
    zlib1g-dev
  BINSEC_SYSTEM_PACKAGES_READY=1
}

install_binsec() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "  binsec: skipped (Linux-only CT kernel lane)"
    return 0
  fi

  local package="${BINSEC_OPAM_PACKAGE:-binsec.0.11.1}"
  local decoder_package="${BINSEC_DECODER_OPAM_PACKAGE:-unisim_archisec.0.0.14}"
  local solver_packages_raw="${BINSEC_SOLVER_OPAM_PACKAGES:-bitwuzla.1.0.6 bitwuzla-cxx.0.9.0}"
  local -a solver_packages
  read -r -a solver_packages <<< "$solver_packages_raw"

  install_binsec_system_packages

  echo "  binsec: installing via OPAM"
  if ! command -v opam &>/dev/null; then
    echo "opam is required to install BINSEC on this runner" >&2
    return 1
  fi

  export OPAMYES=1
  export OPAMSWITCH="${OPAMSWITCH:-rscrypto-ct}"

  if [[ ! -d "$HOME/.opam" ]]; then
    opam init --bare --disable-sandboxing -y
  fi

  if ! opam switch list --short | grep -qx "$OPAMSWITCH"; then
    opam switch create "$OPAMSWITCH" ocaml-base-compiler.5.2.1 -y
  fi

  eval "$(opam env --switch="$OPAMSWITCH" --set-switch)"
  local -a required_packages=("$decoder_package" "${solver_packages[@]}" "$package")
  local -a missing_packages=()
  local installed_packages
  installed_packages="$(opam list --installed --short)"

  local required
  for required in "${required_packages[@]}"; do
    if ! grep -qx "${required%%.*}" <<<"$installed_packages"; then
      missing_packages+=("$required")
    fi
  done

  if command -v binsec &>/dev/null && [[ "${#missing_packages[@]}" -eq 0 ]]; then
    echo "  binsec: cached"
  else
    if [[ "${#missing_packages[@]}" -gt 0 ]]; then
      echo "  binsec: installing missing OPAM package(s): ${missing_packages[*]}"
    else
      echo "  binsec: binary missing from PATH; reinstalling $package"
    fi
    opam install "${required_packages[@]}" -y
    # BINSEC links optional solver bindings at build time; reinstall if a cached
    # switch gained solver packages after BINSEC was first installed.
    opam reinstall "$package" -y
  fi
  eval "$(opam env --switch="$OPAMSWITCH" --set-switch)"

  if [[ -n "${GITHUB_PATH:-}" ]]; then
    echo "$HOME/.opam/$OPAMSWITCH/bin" >> "$GITHUB_PATH"
  fi
  export PATH="$HOME/.opam/$OPAMSWITCH/bin:$PATH"

  binsec -version || true
}

install_ct_linux_packages() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    return 0
  fi
  if ! command -v apt-get &>/dev/null; then
    return 0
  fi
  install_binsec_system_packages
  sudo apt-get install -y --no-install-recommends musl-tools
}

echo ""
echo "Installing tools for mode: $MODE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

case "$MODE" in
  standard)
    # Standard CI tools (same for all CI runners)
    install_binstall
    install_if_missing "cargo-nextest" "cargo-nextest"
    install_if_missing "cargo-deny" "cargo-deny"
    install_if_missing "cargo-audit" "cargo-audit"
    ensure_cargo_rail "$CARGO_RAIL_VERSION"
    ensure_cargo_semver_checks "$CARGO_SEMVER_CHECKS_VERSION"
    install_if_missing "just" "just"
    ;;

  rail)
    # Compiler-backed Cargo graph assurance.
    install_binstall
    ensure_cargo_rail "$CARGO_RAIL_VERSION"
    ;;

  ci)
    # Native CI tools: test runner + just wrapper only. Supply-chain tooling
    # belongs to the dedicated supply-chain lane.
    install_binstall
    install_if_missing "cargo-nextest" "cargo-nextest"
    install_if_missing "just" "just"
    ;;

  supply-chain)
    install_binstall
    install_if_missing "cargo-deny" "cargo-deny"
    install_if_missing "cargo-audit" "cargo-audit"
    ;;

  ibm)
    # IBM runners: keep installs minimal and fast.
    # Skip cargo-binstall entirely on these platforms since it requires source
    # compilation; just install 'just' directly via cargo.
    # `cargo-nextest` often lacks prebuilt binaries on s390x/ppc64le and falls
    # back to expensive source builds; test.sh will use `cargo test` fallback.
    if command -v just &>/dev/null; then
      echo "  just: cached"
    else
      echo "  just: installing via cargo (skipping binstall for speed)..."
      cargo install just --locked
    fi
    ;;

  bench)
    # Benchmark tools (Criterion + tuning)
    install_binstall
    install_if_missing "cargo-criterion" "cargo-criterion"
    install_if_missing "critcmp" "critcmp"
    install_if_missing "just" "just"
    ;;

  fuzz)
    # Weekly fuzz lane: keep tool set minimal and explicit.
    install_binstall
    install_if_missing "cargo-fuzz" "cargo-fuzz"
    install_if_missing "just" "just"
    ;;

  coverage)
    # Coverage reporting: nextest + deterministic fuzz corpus replay.
    install_binstall
    install_if_missing "cargo-llvm-cov" "cargo-llvm-cov"
    install_if_missing "cargo-nextest" "cargo-nextest"
    install_if_missing "just" "just"
    # llvm-tools-preview provides llvm-profdata/llvm-cov for coverage reporting.
    rustup component add llvm-tools-preview 2>/dev/null || true
    ;;

  ct-linux)
    install_just_portable
    install_ct_linux_packages
    install_binsec
    ;;

  minimal)
    # Minimal set for quick jobs
    install_binstall
    install_if_missing "just" "just"
    ;;

  none)
    # Tool-free jobs: use whatever is already present in the toolchain image.
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Usage: install-tools.sh [standard|rail|ci|supply-chain|bench|ibm|fuzz|coverage|ct-linux|minimal|none]"
    exit 1
    ;;
esac

echo ""
echo "Tool installation complete"
