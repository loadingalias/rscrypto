#!/usr/bin/env bash
# Install the few specialized tools not provided by runner images or rustup.
# Usage: install-tools.sh [ci|supply-chain|bench|structural-bench|profile|fuzz|coverage|ct-linux|minimal|none]
#        install-tools.sh --check-mode MODE

set -euo pipefail

MODE=${1:-}

mode_is_supported() {
  case "$1" in
    ci | supply-chain | bench | structural-bench | profile | fuzz | coverage | ct-linux | minimal | none) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ "$MODE" == --check-mode ]]; then
  [[ $# -eq 2 ]] || {
    echo "Usage: install-tools.sh --check-mode MODE" >&2
    exit 2
  }
  mode_is_supported "$2" || {
    echo "Unsupported install-tools mode: $2" >&2
    exit 2
  }
  exit 0
fi

mode_is_supported "$MODE" || {
  echo "Unknown mode: $MODE" >&2
  echo "Usage: install-tools.sh [ci|supply-chain|bench|structural-bench|profile|fuzz|coverage|ct-linux|minimal|none]" >&2
  exit 2
}

CARGO_NEXTEST_VERSION=0.9.143
CARGO_DENY_VERSION=0.20.2
CARGO_AUDIT_VERSION=0.22.2
CARGO_RAIL_VERSION=0.25.0
JUST_VERSION=1.58.0
JUST_RELEASE_BASE=https://github.com/casey/just/releases/download
GUNGRAUN_RUNNER_VERSION=0.19.4
CARGO_SHOW_ASM_VERSION=0.2.62
SAMPLY_VERSION=0.13.1
CARGO_LLVM_LINES_VERSION=0.4.48
CARGO_FUZZ_VERSION=0.13.2
CARGO_LLVM_COV_VERSION=0.9.0

OPAM_REPOSITORY_COMMIT=607f49d990590190e047dba24bd53b28e8195c7b
OPAM_REPOSITORY_REMOTE=https://github.com/ocaml/opam-repository.git
OCAML_COMPILER_PACKAGE=ocaml-base-compiler.5.2.1
BINSEC_PACKAGE=binsec.0.11.3
BINSEC_DECODER_PACKAGE=unisim_archisec.0.0.14
BINSEC_SOLVER_PACKAGES=(bitwuzla.1.0.6 bitwuzla-cxx.0.9.0)

BINSEC_APT_PACKAGES=(
  build-essential
  git
  libgmp-dev
  libmpfr-dev
  m4
  opam
  pkg-config
  zlib1g-dev
)
MUSL_APT_PACKAGE=musl-tools

RSCRYPTO_TOOL_TEMP=${RUNNER_TEMP:-${TMPDIR:-/tmp}}
[[ -d "$RSCRYPTO_TOOL_TEMP" ]] || {
  echo "CI tool install error: temporary root is not a directory: $RSCRYPTO_TOOL_TEMP" >&2
  exit 1
}
RSCRYPTO_TOOL_ROOT=$(mktemp -d "$RSCRYPTO_TOOL_TEMP/rscrypto-ci-tools.XXXXXX")
RSCRYPTO_CARGO_HOME="$RSCRYPTO_TOOL_ROOT/cargo"
RSCRYPTO_CARGO_BIN="$RSCRYPTO_CARGO_HOME/bin"
mkdir -p "$RSCRYPTO_CARGO_BIN"
export CARGO_HOME="$RSCRYPTO_CARGO_HOME"
export CARGO_TARGET_DIR="$RSCRYPTO_TOOL_ROOT/cargo-target"
export PATH="$RSCRYPTO_CARGO_BIN:$PATH"

fail() {
  echo "CI tool install error: $*" >&2
  exit 1
}

extract_version() {
  local output=$1
  if [[ "$output" =~ ([0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    return 1
  fi
}

cargo_tool_path() {
  local binary=$1
  if [[ -x "$RSCRYPTO_CARGO_BIN/$binary" ]]; then
    printf '%s\n' "$RSCRYPTO_CARGO_BIN/$binary"
  elif [[ -x "$RSCRYPTO_CARGO_BIN/$binary.exe" ]]; then
    printf '%s\n' "$RSCRYPTO_CARGO_BIN/$binary.exe"
  else
    return 1
  fi
}

cargo_tool_version() {
  local binary=$1
  local path=$2
  local output
  case "$binary" in
    cargo-rail) output=$("$path" rail --version 2>&1) ;;
    cargo-llvm-cov) output=$("$path" llvm-cov --version 2>&1) ;;
    *) output=$("$path" --version 2>&1) ;;
  esac
  extract_version "$output"
}

cargo_metadata_has() {
  local package=$1
  local version=$2
  local installed
  installed=$(cargo install --list)
  grep -Fqx "$package v$version:" <<<"$installed"
}

verify_cargo_tool() {
  local package=$1
  local version=$2
  local binary=$3
  local path actual

  cargo_metadata_has "$package" "$version" \
    || fail "$package install metadata does not record exact version $version"
  path=$(cargo_tool_path "$binary") \
    || fail "$package metadata exists but $binary is missing or non-executable"
  actual=$(cargo_tool_version "$binary" "$path") \
    || fail "unable to read $binary version after authentication"
  [[ "$actual" == "$version" ]] \
    || fail "$binary reports $actual, expected $version"
}

install_cargo_tool() {
  local package=$1
  local version=$2
  local binary=${3:-$package}

  echo "  $package: installing $version from crates.io into a fresh root"
  cargo install --registry crates-io "$package" --locked --version "=$version" --force
  verify_cargo_tool "$package" "$version" "$binary"
}

sha256_file() {
  local path=$1
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    fail "neither sha256sum nor shasum is available"
  fi
}

just_release_asset() {
  local os arch
  os=$(uname -s)
  arch=$(uname -m)

  case "$os:$arch" in
    Linux:x86_64)
      printf '%s\t%s\t%s\n' \
        "just-$JUST_VERSION-x86_64-unknown-linux-musl.tar.gz" \
        4a5cc2f53e6f0f8c59092a6cc38291eb729d46a7dd95d3ae582008881b84931d \
        just
      ;;
    Linux:aarch64 | Linux:arm64)
      printf '%s\t%s\t%s\n' \
        "just-$JUST_VERSION-aarch64-unknown-linux-musl.tar.gz" \
        748237128c4c40cbdabc65e841d05ceba13cc23a91eaba395495894c1d9764df \
        just
      ;;
    Linux:riscv64 | Linux:riscv64gc)
      printf '%s\t%s\t%s\n' \
        "just-$JUST_VERSION-riscv64gc-unknown-linux-musl.tar.gz" \
        1cbca0ce9880d5d1050115a6e2ced510927f85d1797a204ef6bccb319d923d8d \
        just
      ;;
    Darwin:arm64 | Darwin:aarch64)
      printf '%s\t%s\t%s\n' \
        "just-$JUST_VERSION-aarch64-apple-darwin.tar.gz" \
        50ae3e996c974a0bf32ea7d10f495070df33f1b43e0616b2769e3d4821ed8f48 \
        just
      ;;
    MINGW*:x86_64 | MSYS*:x86_64 | CYGWIN*:x86_64)
      printf '%s\t%s\t%s\n' \
        "just-$JUST_VERSION-x86_64-pc-windows-msvc.zip" \
        759f16fb7aa17c5c8b9594b6d4a8c1a6630dfd042cf2b3ff84841454d3d188dc \
        just.exe
      ;;
    MINGW*:aarch64 | MSYS*:aarch64 | CYGWIN*:aarch64 | \
      MINGW*:arm64 | MSYS*:arm64 | CYGWIN*:arm64)
      printf '%s\t%s\t%s\n' \
        "just-$JUST_VERSION-aarch64-pc-windows-msvc.zip" \
        3a39ed629eb67678976c811a4da46f7985a2c22f4dbabe017b8b2eb5ceb5d01c \
        just.exe
      ;;
    *) return 1 ;;
  esac
}

install_just_release() {
  local release asset expected binary archive actual reported
  release=$(just_release_asset) || {
    echo "  just: no upstream binary for $(uname -s)/$(uname -m); compiling the exact release"
    install_cargo_tool just "$JUST_VERSION"
    return
  }
  IFS=$'\t' read -r asset expected binary <<<"$release"
  archive="$RSCRYPTO_TOOL_ROOT/$asset"

  echo "  just: installing checksummed $JUST_VERSION release asset $asset"
  curl --proto '=https' --tlsv1.2 -fsSL \
    "$JUST_RELEASE_BASE/$JUST_VERSION/$asset" -o "$archive"
  actual=$(sha256_file "$archive")
  [[ "$actual" == "$expected" ]] \
    || fail "just release digest is $actual, expected $expected"
  case "$asset" in
    *.zip)
      command -v unzip >/dev/null 2>&1 \
        || fail "unzip is required for the Windows just release"
      unzip -q "$archive" "$binary" -d "$RSCRYPTO_CARGO_BIN"
      ;;
    *) tar -xf "$archive" -C "$RSCRYPTO_CARGO_BIN" "$binary" ;;
  esac
  chmod 755 "$RSCRYPTO_CARGO_BIN/$binary"
  reported=$(cargo_tool_version just "$RSCRYPTO_CARGO_BIN/$binary") \
    || fail "unable to read just version after release extraction"
  [[ "$reported" == "$JUST_VERSION" ]] \
    || fail "just reports $reported, expected $JUST_VERSION"
}

ensure_cargo_rail() {
  local path actual
  if [[ "${RSCRYPTO_AUTHENTICATED_CARGO_RAIL:-false}" == true ]]; then
    path=$(command -v cargo-rail 2>/dev/null || true)
    if [[ -n "$path" ]]; then
      actual=$(cargo_tool_version cargo-rail "$path" 2>/dev/null || true)
      if [[ "$actual" == "$CARGO_RAIL_VERSION" ]]; then
        echo "  cargo-rail: reusing authenticated $actual from cargo-rail-action"
        return 0
      fi
    fi
    fail "cargo-rail-action reported an authenticated Cargo Rail install, but the exact binary is unavailable"
  fi
  install_cargo_tool cargo-rail "$CARGO_RAIL_VERSION"
}

JUST_INSTALLED=false
ensure_just() {
  if [[ "$JUST_INSTALLED" != true ]]; then
    install_just_release
    JUST_INSTALLED=true
  fi
}

ensure_llvm_tools() {
  if command -v rustup.exe >/dev/null 2>&1; then
    rustup.exe component add llvm-tools-preview
  else
    rustup component add llvm-tools-preview
  fi
}

require_ubuntu_24_04() {
  [[ -f /etc/os-release ]] || fail "Ubuntu 24.04 package metadata is required"
  local os_id os_version
  os_id=$(sed -n 's/^ID=//p' /etc/os-release | tr -d '"')
  os_version=$(sed -n 's/^VERSION_ID=//p' /etc/os-release | tr -d '"')
  [[ "$os_id" == ubuntu && "$os_version" == 24.04 ]] \
    || fail "APT package installation supports Ubuntu 24.04, found $os_id $os_version"
}

apt_install_authenticated_candidates() {
  require_ubuntu_24_04
  command -v apt-get >/dev/null 2>&1 || fail "apt-get is required"
  command -v apt-cache >/dev/null 2>&1 || fail "apt-cache is required"
  command -v dpkg-query >/dev/null 2>&1 || fail "dpkg-query is required"

  sudo apt-get --no-allow-insecure-repositories --error-on=any update

  local -a specifications=()
  local package candidate
  for package in "$@"; do
    candidate=$(LC_ALL=C apt-cache policy "$package" | sed -n 's/^[[:space:]]*Candidate:[[:space:]]*//p')
    [[ -n "$candidate" && "$candidate" != "(none)" ]] \
      || fail "signed APT metadata has no candidate for $package"
    specifications+=("$package=$candidate")
  done

  sudo apt-get install -y --no-install-recommends \
    --no-allow-unauthenticated --no-allow-downgrades --no-remove \
    "${specifications[@]}"

  local specification expected actual
  for specification in "${specifications[@]}"; do
    package=${specification%%=*}
    expected=${specification#*=}
    actual=$(dpkg-query -W -f='${Version}' "$package") \
      || fail "APT did not install $package"
    [[ "$actual" == "$expected" ]] \
      || fail "APT installed $package $actual, expected $expected"
  done
}

install_binsec_system_packages() {
  if [[ "$(uname -s)" != Linux ]]; then
    fail "BINSEC installation is supported only on Linux"
  fi
  if [[ "${BINSEC_SYSTEM_PACKAGES_READY:-}" == 1 ]]; then
    return 0
  fi
  apt_install_authenticated_candidates "${BINSEC_APT_PACKAGES[@]}"
  BINSEC_SYSTEM_PACKAGES_READY=1
}

verify_opam_repository() {
  local repository=$1
  local actual status
  actual=$(git -C "$repository" rev-parse HEAD) \
    || fail "unable to read the OPAM repository commit"
  [[ "$actual" == "$OPAM_REPOSITORY_COMMIT" ]] \
    || fail "OPAM repository is $actual, expected $OPAM_REPOSITORY_COMMIT"
  status=$(git -C "$repository" status --short --untracked-files=all) \
    || fail "unable to verify the OPAM repository worktree"
  [[ -z "$status" ]] \
    || fail "OPAM repository differs from its pinned commit"
}

checkout_opam_repository() {
  local repository=$1
  git init --quiet "$repository"
  git -C "$repository" fetch --depth=1 --no-tags \
    "$OPAM_REPOSITORY_REMOTE" "$OPAM_REPOSITORY_COMMIT"
  git -C "$repository" checkout --quiet --detach FETCH_HEAD
  verify_opam_repository "$repository"
}

opam_package_is_installed() {
  local package=$1
  local installed
  installed=$(opam list --switch="$OPAMSWITCH" --installed --short --columns=package)
  grep -Fqx "$package" <<<"$installed"
}

verify_opam_packages() {
  local package
  for package in "$OCAML_COMPILER_PACKAGE" "$BINSEC_DECODER_PACKAGE" \
    "${BINSEC_SOLVER_PACKAGES[@]}" "$BINSEC_PACKAGE"; do
    opam_package_is_installed "$package" \
      || fail "OPAM switch is missing exact package $package"
  done
}

install_binsec() {
  install_binsec_system_packages

  export OPAMYES=1
  export OPAMROOT="$RSCRYPTO_TOOL_ROOT/opam"
  export OPAMSWITCH=rscrypto-ct

  local repository="$RSCRYPTO_TOOL_ROOT/opam-repository"
  checkout_opam_repository "$repository"
  opam init --bare --disable-sandboxing --no-setup --no-opamrc -y \
    default "$repository"
  verify_opam_repository "$repository"

  opam switch create "$OPAMSWITCH" "$OCAML_COMPILER_PACKAGE" \
    --repositories=default -y

  local -a required_packages=(
    "$BINSEC_DECODER_PACKAGE"
    "${BINSEC_SOLVER_PACKAGES[@]}"
    "$BINSEC_PACKAGE"
  )
  opam install --switch="$OPAMSWITCH" "${required_packages[@]}" -y
  verify_opam_repository "$repository"
  verify_opam_packages

  local switch_bin
  switch_bin=$(opam var bin --switch="$OPAMSWITCH")
  [[ -x "$switch_bin/binsec" ]] \
    || fail "authenticated OPAM switch is missing binsec"
  export PATH="$switch_bin:$PATH"

  local actual
  actual=$(extract_version "$("$switch_bin/binsec" -version 2>&1)") \
    || fail "unable to read BINSEC version"
  [[ "$actual" == "${BINSEC_PACKAGE#binsec.}" ]] \
    || fail "BINSEC reports $actual, expected ${BINSEC_PACKAGE#binsec.}"

  if [[ -n "${GITHUB_PATH:-}" ]]; then
    echo "$switch_bin" >>"$GITHUB_PATH"
  fi
}

install_ct_linux_packages() {
  apt_install_authenticated_candidates "${BINSEC_APT_PACKAGES[@]}" "$MUSL_APT_PACKAGE"
  BINSEC_SYSTEM_PACKAGES_READY=1
}

echo "Installing CI tools (mode: $MODE)"

case "$MODE" in
  ci)
    install_cargo_tool cargo-nextest "$CARGO_NEXTEST_VERSION"
    ensure_just
    ;;
  supply-chain)
    install_cargo_tool cargo-deny "$CARGO_DENY_VERSION"
    install_cargo_tool cargo-audit "$CARGO_AUDIT_VERSION"
    ;;
  bench)
    # Criterion is a pinned dev-dependency. Native crypto qualification also
    # disassembles the measured binary, so bench runners need LLVM tools.
    ensure_just
    ensure_llvm_tools
    ;;
  structural-bench)
    install_cargo_tool gungraun-runner "$GUNGRAUN_RUNNER_VERSION"
    ensure_just
    ;;
  profile)
    install_cargo_tool cargo-show-asm "$CARGO_SHOW_ASM_VERSION" cargo-asm
    install_cargo_tool samply "$SAMPLY_VERSION"
    install_cargo_tool cargo-llvm-lines "$CARGO_LLVM_LINES_VERSION"
    ensure_just
    ensure_llvm_tools
    ;;
  fuzz)
    install_cargo_tool cargo-fuzz "$CARGO_FUZZ_VERSION"
    ;;
  coverage)
    install_cargo_tool cargo-llvm-cov "$CARGO_LLVM_COV_VERSION"
    install_cargo_tool cargo-nextest "$CARGO_NEXTEST_VERSION"
    ensure_just
    ensure_llvm_tools
    ;;
  ct-linux)
    install_ct_linux_packages
    install_binsec
    ;;
  minimal)
    ensure_just
    ;;
  none)
    ;;
esac

if [[ "${RSCRYPTO_REQUIRE_CARGO_RAIL:-false}" == true ]]; then
  ensure_cargo_rail
  ensure_just
fi

if [[ -n "${GITHUB_PATH:-}" ]]; then
  echo "$RSCRYPTO_CARGO_BIN" >>"$GITHUB_PATH"
fi

echo "CI tool installation complete"
