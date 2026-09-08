#!/usr/bin/env bash
# Shared implementation for native Ubuntu entry points.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

platform="${1:?native platform is required}"
shift
ci=false
if [[ "${1:-}" == --ci ]]; then ci=true; shift; fi
[[ "$#" -eq 0 ]] || { echo "usage: scripts/tooling/$platform.sh [--ci]" >&2; exit 64; }
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
apt_options=(-o "Dir::Etc::sourcelist=$temporary/sources.list" -o Dir::Etc::sourceparts=-
  -o "Dir::State::lists=$temporary/lists" -o APT::Update::Error-Mode=any)
apt=("${sudo_cmd[@]}" env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}")
"${apt[@]}" update
catalog_get() { python3 "$SCRIPT_DIR/catalog.py" get "$@"; }
python3 "$SCRIPT_DIR/catalog.py" validate
mapfile -t packages < <(catalog_get "$linux_section" packages)
if [[ "$ci" == false ]]; then
  mapfile -t native_packages < <(catalog_get "$platform" packages)
  packages+=("${native_packages[@]}")
fi
# Exact candidates come from the selected snapshot, including repeat installations.
pinned_packages=()
for package in "${packages[@]}"; do
  version="$(apt-cache "${apt_options[@]}" madison "$package" | awk 'NR == 1 {print $3}')"
  [[ -n "$version" && "$version" != '(none)' ]] || { echo "missing Ubuntu package: $package" >&2; exit 1; }
  pinned_packages+=("$package=$version")
done
"${apt[@]}" install -y --allow-downgrades "${pinned_packages[@]}"

prefix="$HOME/.local/share/rscrypto-tooling"
mkdir -p "$prefix"
python3 "$SCRIPT_DIR/catalog.py" download "$platform" rustup "$temporary/rustup-init"
chmod +x "$temporary/rustup-init"
host="$(catalog_get "$platform" rust-host)"
channel="$(python3 "$SCRIPT_DIR/../lib/toolchain.py")"
"$temporary/rustup-init" -y --no-modify-path --default-host "$host" --default-toolchain none
cargo_bin="${CARGO_HOME:-$HOME/.cargo}/bin"
export PATH="$cargo_bin:$PATH"
components=()
if [[ "$ci" == false ]]; then mapfile -t components < <(catalog_get "$platform" components); fi
component_args=()
for component in "${components[@]}"; do component_args+=(--component "$component"); done
python3 "$SCRIPT_DIR/../lib/toolchain.py" --install "$host" "${component_args[@]}"
export RUSTUP_TOOLCHAIN="$channel"
# Archive tools retain their complete directory layouts, including LLVM and Zig libraries.
if [[ "$ci" == true ]]; then
  : > "$temporary/archives"
  if [[ "$platform" == aarch64-linux || "$platform" == x86_64-linux ]]; then
    directory="$(python3 "$SCRIPT_DIR/catalog.py" install-archive "$platform" cargo-binstall "$prefix")"
    printf 'cargo-binstall\t%s\n' "$directory" > "$temporary/archives"
  fi
else
  python3 "$SCRIPT_DIR/catalog.py" install-archives "$platform" "$prefix" > "$temporary/archives"
fi
tool_paths=()
while IFS=$'\t' read -r name directory; do
  if [[ -d "$directory/bin" ]]; then tool_paths+=("$directory/bin"); else tool_paths+=("$directory"); fi
  if [[ "$name" == llvm ]]; then export LIBCLANG_PATH="$directory/lib"; fi
done < "$temporary/archives"
tool_paths+=("$cargo_bin")
path_prefix="$(IFS=:; echo "${tool_paths[*]}")"
export PATH="$path_prefix:$PATH"
tool_section="$platform"
[[ "$ci" == false ]] || tool_section=ci
mapfile -t cargo_tools < <(catalog_get "$tool_section" cargo)
for tool in "${cargo_tools[@]}"; do
  version="$(catalog_get cargo "$tool")"
  # Cargo's install registry verifies exact installed package versions on reruns.
  if command -v cargo-binstall >/dev/null 2>&1 && [[ "$platform" == aarch64-linux || "$platform" == x86_64-linux ]]; then
    env -u RUSTC_WRAPPER -u CARGO_ENCODED_RUSTFLAGS \
      cargo +"$channel" binstall --locked --no-confirm --targets "$(catalog_get "$platform" rust-host)" "$tool@$version"
  else
    env -u RUSTC_WRAPPER -u CARGO_ENCODED_RUSTFLAGS \
      cargo +"$channel" install --locked --target "$(catalog_get "$platform" rust-host)" --version "$version" "$tool"
  fi
done
if [[ "$ci" == false && "$platform" != aarch64-linux && "$platform" != x86_64-linux ]]; then
  env -u RUSTC_WRAPPER -u CARGO_ENCODED_RUSTFLAGS \
    cargo +"$channel" install --locked --target "$(catalog_get "$platform" rust-host)" --version "$(catalog_get versions cargo-rail)" cargo-rail
fi

# Persistent paths are shared by interactive shells and non-interactive Bash recipes.
environment="$prefix/environment.sh"
{
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
clang --version
cmake --version
if [[ "$ci" == false ]]; then cargo rail --version; fi
cargo nextest --version
printf 'Installed %s tooling. Load with: source "%s"\n' "$platform" "$environment"
