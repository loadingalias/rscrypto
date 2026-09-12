#!/usr/bin/env bash
# Shared implementation for native Ubuntu entry points.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

platform="${1:?native platform is required}"
shift
ci=false
profile=ci
proof=false
case "${1:-}" in
  --ci-ct-full) ci=true; profile=ci-ct; proof=true; shift ;;
  --ci) ci=true; shift ;;
  --ci-compat|--ci-package|--ci-fuzz|--ci-miri|--ci-ct|--ci-bench|--ci-cross-build|--ci-cross-run) ci=true; profile="${1#--}"; shift ;;
esac
case "$profile:$platform" in
  ci-cross-run:riscv64-linux|ci-cross-run:powerpc64le-linux|ci-cross-run:s390x-linux|ci-cross-build:x86_64-linux) ;;
  ci-cross-run:*|ci-cross-build:*) echo "invalid cross-build tooling host" >&2; exit 64 ;;
  ci:*|ci-bench:*|ci-ct:*|*:x86_64-linux|ci-fuzz:aarch64-linux) ;;
  *) echo "$profile tooling is unsupported on $platform" >&2; exit 64 ;;
esac
cross_target=""
tools_archive=""
if [[ "$profile" == ci-cross-build ]]; then
  cross_target="${1:?cross-build target is required}"
  shift
elif [[ "$profile" == ci-cross-run ]]; then
  tools_archive="${1:?runner tools archive is required}"
  shift
fi
[[ "$#" -eq 0 ]] || { echo "usage: scripts/tooling/$platform.sh [--ci|--ci-compat|--ci-package|--ci-fuzz|--ci-miri|--ci-ct|--ci-ct-full|--ci-bench|--ci-cross-build TARGET|--ci-cross-run TOOLS_ARCHIVE]" >&2; exit 64; }
machine="${platform%-linux}"
[[ "$machine" != powerpc64le ]] || machine=ppc64le
case "$platform" in
  aarch64-linux|x86_64-linux|riscv64-linux|s390x-linux|powerpc64le-linux) ;;
  *) echo "unknown Linux platform: $platform" >&2; exit 64 ;;
esac
[[ "$(uname -s)" == Linux && "$(uname -m)" == "$machine" ]] || {
  echo "$platform requires a native ${platform%-linux} Linux host" >&2; exit 1;
}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
# Ubuntu Server supplies Python; install it from the selected archive if absent.
# Read only the two bootstrap strings before the full TOML reader is available.
catalog="$REPO_ROOT/.config/tooling.toml"
linux_section=linux
[[ "$ci" == false ]] || linux_section=linux-ci
bootstrap_value() { sed -n '/^\['"${2:-$linux_section}"'\]$/,/^\[/s/^'"$1"' = "\([^"]*\)"$/\1/p' "$catalog"; }
ubuntu="$(bootstrap_value ubuntu)"
snapshot="$(bootstrap_value snapshot linux)"
# shellcheck source=/dev/null
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == "$ubuntu" ]] || {
  echo "expected Ubuntu $ubuntu; found $PRETTY_NAME" >&2; exit 1;
}
sudo_cmd=()
if [[ "$(id -u)" != 0 ]]; then sudo_cmd=(sudo); fi
temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT
# Minimal Ubuntu images may omit the HTTPS trust store and Python. Bootstrap
# those through Ubuntu's signed archive, then converge them to the snapshot too.
if ! command -v python3 >/dev/null || [[ ! -f /etc/ssl/certs/ca-certificates.crt ]]; then
  "${sudo_cmd[@]}" apt-get -o APT::Update::Error-Mode=any update
  "${sudo_cmd[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y python3 ca-certificates
fi
codename="$(bootstrap_value codename)"
mkdir -p "$temporary/lists/partial"
chmod 755 "$temporary" "$temporary/lists" "$temporary/lists/partial"
# Explicit snapshot URLs work on an empty package cache.
# The archive remains signed; historical snapshots intentionally outlive Valid-Until.
for suite in "$codename" "$codename-updates" "$codename-security"; do
  printf 'deb [check-valid-until=no signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://snapshot.ubuntu.com/ubuntu/%s %s main universe\n' "$snapshot" "$suite"
done > "$temporary/sources.list"
# Dependencies must follow the snapshot even when a runner preinstalls newer packages.
cat > "$temporary/preferences" <<'PREFERENCES'
Package: *
Pin: origin snapshot.ubuntu.com
Pin-Priority: 1001
PREFERENCES
apt_options=(-o "Dir::Etc::sourcelist=$temporary/sources.list" -o Dir::Etc::sourceparts=-
  -o "Dir::Etc::preferences=$temporary/preferences" -o Dir::Etc::preferencesparts=-
  -o "Dir::State::lists=$temporary/lists" -o APT::Update::Error-Mode=any)
apt=("${sudo_cmd[@]}" env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}")
"${apt[@]}" update
catalog_get() { python3 "$SCRIPT_DIR/catalog.py" get "$@"; }
python3 "$SCRIPT_DIR/catalog.py" validate
package_section="$linux_section"
[[ "$profile" == ci || "$profile" == ci-bench || "$profile" == ci-package ]] || package_section="$profile"
mapfile -t packages < <(catalog_get "$package_section" packages)
if [[ "$profile" == ci-cross-build ]]; then
  cross_prefix="$(python3 scripts/lib/cross_build.py "$cross_target")"
  cross_arch="${cross_prefix%-linux-gnu}"
  [[ "$cross_arch" != powerpc64le ]] || cross_arch=ppc64el
  packages+=("gcc-$cross_prefix" "g++-$cross_prefix" "libc6-dev-$cross_arch-cross")
fi
if [[ "$ci" == false ]]; then
  mapfile -t native_packages < <(catalog_get "$platform" packages)
  packages+=("${native_packages[@]}")
fi
if [[ "$ci" == true && "$profile" == ci && ( "$platform" == x86_64-linux || "$platform" == aarch64-linux ) ]]; then
  mapfile -t musl_packages < <(catalog_get ci-musl packages)
  packages+=("${musl_packages[@]}")
fi
if [[ "$proof" == true && ( "$platform" == x86_64-linux || "$platform" == aarch64-linux ) ]]; then
  mapfile -t proof_packages < <(catalog_get ci-ct-proof packages)
  packages+=("${proof_packages[@]}")
fi
# Exact candidates come from the selected snapshot, including repeat installations.
pinned_packages=()
for package in "${packages[@]}"; do
  version="$(apt-cache "${apt_options[@]}" madison "$package" | awk 'NR == 1 {print $3}')"
  [[ -n "$version" && "$version" != '(none)' ]] || { echo "missing Ubuntu package: $package" >&2; exit 1; }
  pinned_packages+=("$package=$version")
done
install_options=(--allow-downgrades)
[[ "$ci" == false ]] || install_options+=(--no-install-recommends)
"${apt[@]}" install -y "${install_options[@]}" "${pinned_packages[@]}"

prefix="$HOME/.local/share/rscrypto-tooling"
mkdir -p "$prefix"
python3 "$SCRIPT_DIR/catalog.py" download "$platform" rustup "$temporary/rustup-init"
chmod +x "$temporary/rustup-init"
host="$(catalog_get "$platform" rust-host)"
channel="$(python3 "$SCRIPT_DIR/../lib/toolchain.py" --target "$host")"
"$temporary/rustup-init" -y --no-modify-path --default-host "$host" --default-toolchain none
cargo_bin="${CARGO_HOME:-$HOME/.cargo}/bin"
export PATH="$cargo_bin:$PATH"
components=()
if [[ "$ci" == false ]]; then mapfile -t components < <(catalog_get "$platform" components); fi
component_args=()
for component in "${components[@]}"; do component_args+=(--component "$component"); done
if [[ "$profile" == ci-cross-build ]]; then
  nightly="$(python3 "$SCRIPT_DIR/../lib/toolchain.py" --target "$cross_target")"
  rustup toolchain install "$channel" --profile minimal --component rustfmt
  rustup toolchain install "$nightly" --profile minimal --component clippy --component llvm-tools
  rustup target add --toolchain "$nightly" "$cross_target"
elif [[ "$profile" == ci-cross-run ]]; then
  # No compiler workloads run here; Rust supplies the pinned Nextest launcher
  # and host identity used by the existing CT orchestrator.
  stable="$(python3 "$SCRIPT_DIR/../lib/toolchain.py")"
  rustup toolchain install "$stable" --profile minimal
  rustup toolchain install "$channel" --profile minimal
elif [[ "$profile" == ci-compat ]]; then
  python3 "$REPO_ROOT/scripts/check/compat.py" --install
elif [[ "$profile" == ci-package ]]; then
  python3 "$REPO_ROOT/scripts/check/package.py" --install
elif [[ "$profile" == ci-fuzz || "$profile" == ci-miri || "$profile" == ci-ct || "$profile" == ci-bench ]]; then
  mapfile -t components < <(catalog_get "$profile" components)
  component_args=()
  for component in "${components[@]}"; do component_args+=(--component "$component"); done
  stable="$(python3 "$SCRIPT_DIR/../lib/toolchain.py")"
  if [[ "$channel" != "$stable" ]]; then rustup toolchain install "$stable" --profile minimal; fi
  rustup toolchain install "$channel" --profile minimal "${component_args[@]}"
  if [[ "$profile" == ci-fuzz || "$profile" == ci-miri ]]; then
    nightly="$(python3 "$SCRIPT_DIR/../lib/toolchain.py" --nightly)"
    mapfile -t components < <(catalog_get "$profile" nightly-components)
    component_args=()
    for component in "${components[@]}"; do component_args+=(--component "$component"); done
    rustup toolchain install "$nightly" --profile minimal "${component_args[@]}"
  fi
else
  python3 "$SCRIPT_DIR/../lib/toolchain.py" --install "$host" "${component_args[@]}"
  if [[ "$ci" == true && ( "$platform" == x86_64-linux || "$platform" == aarch64-linux ) ]]; then
    rustup target add --toolchain "$channel" "${host%-gnu}-musl"
  fi
fi
export RUSTUP_TOOLCHAIN="$channel"
binstall=false
if catalog_get "$platform" assets cargo-binstall >/dev/null 2>&1; then binstall=true; fi
# Archive tools retain their complete directory layouts, including LLVM and Zig libraries.
if [[ "$ci" == true ]]; then
  : > "$temporary/archives"
  if [[ "$binstall" == true && "$profile" != ci-cross-run ]]; then
    directory="$(python3 "$SCRIPT_DIR/catalog.py" install-archive "$platform" cargo-binstall "$prefix")"
    printf 'cargo-binstall\t%s\n' "$directory" > "$temporary/archives"
  fi
  if [[ "$profile" == ci-compat ]]; then
    mapfile -t compat_assets < <(catalog_get ci-compat assets)
    for asset in "${compat_assets[@]}"; do
      directory="$(python3 "$SCRIPT_DIR/catalog.py" install-archive "$platform" "$asset" "$prefix")"
      printf '%s\t%s\n' "$asset" "$directory" >> "$temporary/archives"
    done
  fi
else
  python3 "$SCRIPT_DIR/catalog.py" install-archives "$platform" "$prefix" > "$temporary/archives"
fi
tool_paths=()
while IFS=$'\t' read -r name directory; do
  if [[ -d "$directory/bin" ]]; then tool_paths+=("$directory/bin"); else tool_paths+=("$directory"); fi
  if [[ "$name" == llvm ]]; then export LIBCLANG_PATH="$directory/lib"; fi
done < "$temporary/archives"
if [[ "$proof" == true && ( "$platform" == x86_64-linux || "$platform" == aarch64-linux ) ]]; then
  export OPAMROOT="$prefix/opam"
  opam init --bare --no-setup --no-opamrc --yes default "$(catalog_get ci-ct-proof opam-repository)"
  if ! opam switch list --short | grep -Fxq ct; then
    opam switch create ct "$(catalog_get ci-ct-proof compiler)" --yes --no-depexts
  fi
  mapfile -t proof_tools < <(catalog_get ci-ct-proof opam)
  opam install --switch ct --yes --no-depexts "${proof_tools[@]}"
  tool_paths+=("$OPAMROOT/ct/bin")
fi
tool_paths+=("$cargo_bin")
path_prefix="$(IFS=:; echo "${tool_paths[*]}")"
export PATH="$path_prefix:$PATH"
tool_section="$platform"
[[ "$ci" == false ]] || tool_section="$profile"
mapfile -t cargo_tools < <(catalog_get "$tool_section" cargo)
if [[ "$ci" == true && "$profile" == ci && "$platform" == x86_64-linux ]]; then
  mapfile -t policy_tools < <(catalog_get ci-policy cargo)
  cargo_tools+=("${policy_tools[@]}")
fi
for tool in "${cargo_tools[@]}"; do
  version="$(catalog_get cargo "$tool")"
  # Cargo's install registry verifies exact installed package versions on reruns.
  if [[ "$binstall" == true && ( "$profile" != ci-cross-build || "$tool" != cargo-nextest ) ]]; then
    env -u RUSTC_WRAPPER -u CARGO_ENCODED_RUSTFLAGS \
      cargo +"$channel" binstall --locked --no-confirm --targets "$host" --targets "${host%-gnu}-musl" "$tool@$version"
  else
    env -u RUSTC_WRAPPER -u CARGO_ENCODED_RUSTFLAGS \
      cargo +"$channel" install --locked --target "$(catalog_get "$platform" rust-host)" --version "$version" "$tool"
  fi
done
if [[ "$ci" == false && "$platform" != aarch64-linux && "$platform" != x86_64-linux ]]; then
  env -u RUSTC_WRAPPER -u CARGO_ENCODED_RUSTFLAGS \
    cargo +"$channel" install --locked --target "$(catalog_get "$platform" rust-host)" --version "$(catalog_get versions cargo-rail)" cargo-rail
fi

if [[ "$profile" == ci-cross-build ]]; then
  python3 "$SCRIPT_DIR/transfer.py" prepare "$cross_target" "target/$cross_target-tools.tar.gz"
elif [[ "$profile" == ci-cross-run ]]; then
  tools_parent="$(mktemp -d "$prefix/runner-tools.XXXXXX")"
  tools_bin="$(python3 "$SCRIPT_DIR/transfer.py" install "$host" "$tools_archive" "$tools_parent/verified")"
  path_prefix="$tools_bin:$path_prefix"
  export PATH="$path_prefix:$PATH"
fi

# Persistent paths are shared by interactive shells and non-interactive Bash recipes.
environment="$prefix/environment.sh"
{
  if [[ "$proof" == true && ( "$platform" == x86_64-linux || "$platform" == aarch64-linux ) ]]; then
    printf 'eval "$(opam env --root=%q --switch=ct --shell=bash)"\n' "$OPAMROOT"
  fi
  printf "export PATH=%q:\"\$PATH\"\n" "$path_prefix"
  if [[ -n "${LIBCLANG_PATH:-}" ]]; then printf 'export LIBCLANG_PATH=%q\n' "$LIBCLANG_PATH"; fi
} > "$environment"
if [[ "$ci" == false ]]; then
  for startup in "$HOME/.profile" "$HOME/.bashrc"; do
    line="source \"$environment\""
    touch "$startup"
    grep -Fxq "$line" "$startup" || printf '\n%s\n' "$line" >> "$startup"
  done
fi
if [[ "$ci" == false && ( "$platform" == aarch64-linux || "$platform" == x86_64-linux ) ]]; then
# Linux profiling is available to the runner account, including non-root perf/samply.
"${sudo_cmd[@]}" tee /etc/sysctl.d/99-rscrypto-profiling.conf >/dev/null <<'CONF'
kernel.perf_event_paranoid = -1
CONF
"${sudo_cmd[@]}" sysctl -p /etc/sysctl.d/99-rscrypto-profiling.conf
# perf must match the running kernel; cloud kernels may differ from linux-generic.
if ! perf --version; then
  "${apt[@]}" install -y "linux-tools-$(uname -r)"
fi
perf --version
valgrind --version
gungraun-runner --version
samply --version
fi
if [[ "$profile" == ci || "$profile" == ci-bench ]]; then
  clang --version
  cmake --version
fi
if [[ "$ci" == false ]]; then cargo rail --version; fi
case "$profile" in
  ci-compat) wasmtime --version ;;
  ci-fuzz) cargo fuzz --version ;;
  ci-ct|ci-miri|ci-package|ci-bench) just --version ;;
  ci|ci-cross-build|ci-cross-run) cargo nextest --version ;;
esac
printf 'Installed %s tooling. Load with: source "%s"\n' "$platform" "$environment"
