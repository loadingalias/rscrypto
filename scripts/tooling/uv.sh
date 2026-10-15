#!/usr/bin/env bash
# Install the catalog's pinned uv for a native Linux platform and the Python that .python-version names,
# then print uv's directory. The installer calls this first; jobs that run no installer call it alone.
set -euo pipefail

platform="${1:?give the Linux platform, such as x86_64-linux}"
[[ $# -eq 1 ]] || { echo "usage: scripts/tooling/uv.sh <platform>" >&2; exit 64; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Read only the two bootstrap strings before uv provides the full TOML reader.
asset() { sed -n '/^\['"$platform"'\.assets\.uv\]$/,/^\[/s/^'"$1"' = "\([^"]*\)"$/\1/p' "$REPO_ROOT/.config/tooling.toml"; }
sha="$(asset sha256)"
[[ "$sha" =~ ^[0-9a-f]{64}$ ]] || { echo "$platform has no pinned uv archive" >&2; exit 64; }
# The layout and receipt match catalog.py install-archive.
directory="$HOME/.local/share/rscrypto-tooling/uv/${sha:0:16}"
if [[ ! -f "$directory/.rscrypto-installed" ]]; then
  temporary="$(mktemp -d)"
  trap 'rm -rf -- "$temporary"' EXIT
  curl --proto '=https' --tlsv1.2 -fsSL --retry 3 -o "$temporary/uv.tar.gz" "$(asset url)"
  sha256sum --check --quiet <<<"$sha  $temporary/uv.tar.gz"
  mkdir "$temporary/uv"
  tar -xzf "$temporary/uv.tar.gz" -C "$temporary/uv" --strip-components 1
  printf '%s\n' "$sha" > "$temporary/uv/.rscrypto-installed"
  mkdir -p "${directory%/*}"
  rm -rf -- "$directory"
  mv "$temporary/uv" "$directory"
fi
# Precompiled, the standard library imports without recompiling on every helper start.
(cd "$REPO_ROOT" && "$directory/uv" python install --compile-bytecode >&2)
printf '%s\n' "$directory"
