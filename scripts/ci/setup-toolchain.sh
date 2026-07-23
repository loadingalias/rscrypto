#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: setup-toolchain.sh <exact-toolchain> [comma-separated-components]" >&2
  exit 2
fi

toolchain=$1
components=${2:-}

if [[ ! "$toolchain" =~ ^(nightly|beta)-[0-9]{4}-[0-9]{2}-[0-9]{2}$ \
  && ! "$toolchain" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "toolchain must be an exact dated or release version: $toolchain" >&2
  exit 1
fi

if ! command -v rustup >/dev/null 2>&1; then
  echo "rustup must be present in the authenticated runner image; network bootstrap is prohibited" >&2
  exit 1
fi

install_args=(toolchain install "$toolchain" --profile minimal --no-self-update)
if [[ -n "$components" ]]; then
  IFS=',' read -r -a requested_components <<<"$components"
  for component in "${requested_components[@]}"; do
    component=${component//[[:space:]]/}
    [[ "$component" =~ ^[a-z0-9][a-z0-9-]*$ ]] || {
      echo "invalid rustup component: $component" >&2
      exit 1
    }
    install_args+=(--component "$component")
  done
fi

rustup "${install_args[@]}"
rustup default "$toolchain"
rustc "+$toolchain" --version --verbose
