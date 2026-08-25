#!/usr/bin/env bash
# Install CI tools through authenticated package-manager boundaries.
# Usage: install-tools.sh [standard|quality|release|semver|rail|ci|supply-chain|bench|structural-bench|profile|ibm|fuzz|coverage|ct-linux|minimal|none]

set -euo pipefail

MODE=${1:-standard}

CARGO_NEXTEST_VERSION=0.9.143
CARGO_DENY_VERSION=0.20.2
CARGO_AUDIT_VERSION=0.22.2
CARGO_RAIL_VERSION=0.22.2
CARGO_SEMVER_CHECKS_VERSION=0.50.0
JUST_VERSION=1.58.0
ZIZMOR_VERSION=1.29.0
CARGO_CRITERION_VERSION=1.1.0
CRITCMP_VERSION=0.1.8
GUNGRAUN_RUNNER_VERSION=0.19.4
CARGO_SHOW_ASM_VERSION=0.2.62
SAMPLY_VERSION=0.13.1
CARGO_LLVM_LINES_VERSION=0.4.48
CARGO_FUZZ_VERSION=0.13.2
CARGO_LLVM_COV_VERSION=0.9.0
ACTIONLINT_VERSION=1.7.12

OPAM_REPOSITORY_COMMIT=49f6d620cf20ae0168cfcbeb2c33932e06cb4b74
OPAM_REPOSITORY_REMOTE=https://github.com/ocaml/opam-repository.git
OCAML_COMPILER_PACKAGE=ocaml-base-compiler.5.2.1
BINSEC_PACKAGE=binsec.0.11.1
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

install_actionlint() {
  local binary="$RSCRYPTO_CARGO_BIN/actionlint"
  [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* || "$(uname -s)" == CYGWIN* ]] \
    && binary+=.exe

  echo "  actionlint: installing $ACTIONLINT_VERSION through the Go checksum database"
  GOBIN="$RSCRYPTO_CARGO_BIN" \
    GOPATH="$RSCRYPTO_TOOL_ROOT/go" \
    GOMODCACHE="$RSCRYPTO_TOOL_ROOT/go/pkg/mod" \
    GOCACHE="$RSCRYPTO_TOOL_ROOT/go-build" \
    GOPROXY=https://proxy.golang.org \
    GOSUMDB=sum.golang.org \
    GOPRIVATE='' \
    GONOSUMDB='' \
    GOINSECURE='' \
    go install "github.com/rhysd/actionlint/cmd/actionlint@v$ACTIONLINT_VERSION"

  [[ -x "$binary" ]] || fail "actionlint install did not produce an executable"
  local actual
  actual=$(extract_version "$("$binary" -version 2>&1)") \
    || fail "unable to read actionlint version"
  [[ "$actual" == "$ACTIONLINT_VERSION" ]] \
    || fail "actionlint reports $actual, expected $ACTIONLINT_VERSION"
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
  opam reinstall --switch="$OPAMSWITCH" "$BINSEC_PACKAGE" -y
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
  standard)
    install_cargo_tool cargo-nextest "$CARGO_NEXTEST_VERSION"
    install_cargo_tool cargo-deny "$CARGO_DENY_VERSION"
    install_cargo_tool cargo-audit "$CARGO_AUDIT_VERSION"
    install_cargo_tool cargo-rail "$CARGO_RAIL_VERSION"
    install_cargo_tool cargo-semver-checks "$CARGO_SEMVER_CHECKS_VERSION"
    install_cargo_tool just "$JUST_VERSION"
    ;;
  quality)
    install_cargo_tool just "$JUST_VERSION"
    install_actionlint
    install_cargo_tool zizmor "$ZIZMOR_VERSION"
    ;;
  release)
    install_cargo_tool cargo-rail "$CARGO_RAIL_VERSION"
    install_cargo_tool cargo-semver-checks "$CARGO_SEMVER_CHECKS_VERSION"
    install_cargo_tool cargo-deny "$CARGO_DENY_VERSION"
    install_cargo_tool cargo-audit "$CARGO_AUDIT_VERSION"
    ;;
  semver)
    install_cargo_tool cargo-semver-checks "$CARGO_SEMVER_CHECKS_VERSION"
    ;;
  rail)
    install_cargo_tool cargo-rail "$CARGO_RAIL_VERSION"
    ;;
  ci)
    install_cargo_tool cargo-nextest "$CARGO_NEXTEST_VERSION"
    install_cargo_tool just "$JUST_VERSION"
    ;;
  supply-chain)
    install_cargo_tool cargo-deny "$CARGO_DENY_VERSION"
    install_cargo_tool cargo-audit "$CARGO_AUDIT_VERSION"
    install_actionlint
    install_cargo_tool zizmor "$ZIZMOR_VERSION"
    ;;
  ibm)
    install_cargo_tool just "$JUST_VERSION"
    ;;
  bench)
    install_cargo_tool cargo-criterion "$CARGO_CRITERION_VERSION"
    install_cargo_tool critcmp "$CRITCMP_VERSION"
    install_cargo_tool just "$JUST_VERSION"
    ;;
  structural-bench)
    install_cargo_tool gungraun-runner "$GUNGRAUN_RUNNER_VERSION"
    install_cargo_tool just "$JUST_VERSION"
    ;;
  profile)
    install_cargo_tool cargo-show-asm "$CARGO_SHOW_ASM_VERSION" cargo-asm
    install_cargo_tool samply "$SAMPLY_VERSION"
    install_cargo_tool cargo-llvm-lines "$CARGO_LLVM_LINES_VERSION"
    install_cargo_tool just "$JUST_VERSION"
    ;;
  fuzz)
    install_cargo_tool cargo-fuzz "$CARGO_FUZZ_VERSION"
    install_cargo_tool just "$JUST_VERSION"
    ;;
  coverage)
    install_cargo_tool cargo-llvm-cov "$CARGO_LLVM_COV_VERSION"
    install_cargo_tool cargo-nextest "$CARGO_NEXTEST_VERSION"
    install_cargo_tool just "$JUST_VERSION"
    rustup component add llvm-tools-preview
    ;;
  ct-linux)
    install_cargo_tool just "$JUST_VERSION"
    install_ct_linux_packages
    install_binsec
    ;;
  minimal)
    install_cargo_tool just "$JUST_VERSION"
    ;;
  none)
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Usage: install-tools.sh [standard|quality|release|semver|rail|ci|supply-chain|bench|structural-bench|profile|ibm|fuzz|coverage|ct-linux|minimal|none]" >&2
    exit 2
    ;;
esac

if [[ -n "${GITHUB_PATH:-}" ]]; then
  echo "$RSCRYPTO_CARGO_BIN" >>"$GITHUB_PATH"
fi

echo "CI tool installation complete"
