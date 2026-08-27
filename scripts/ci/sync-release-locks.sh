#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
MODE=${1:-update}

case "$MODE" in
  update | --check) ;;
  *) echo "usage: sync-release-locks.sh [--check]" >&2; exit 2 ;;
esac

cd "$REPO_ROOT"

lockfiles=()
while IFS= read -r lockfile; do
  [[ "$lockfile" == Cargo.lock ]] && continue
  if grep -q '^name = "rscrypto"$' "$lockfile"; then
    lockfiles+=("$lockfile")
  fi
done < <(git ls-files -- '*Cargo.lock')

if [[ ${#lockfiles[@]} -eq 0 ]]; then
  echo "no committed auxiliary rscrypto lockfiles found" >&2
  exit 1
fi

for lockfile in "${lockfiles[@]}"; do
  manifest="${lockfile%Cargo.lock}Cargo.toml"
  if [[ ! -f "$manifest" ]]; then
    echo "missing manifest for committed lockfile: $lockfile" >&2
    exit 1
  fi

  if [[ "$MODE" == update ]]; then
    cargo update --manifest-path "$manifest" -p rscrypto
  fi
  cargo metadata --locked --no-deps --format-version 1 \
    --manifest-path "$manifest" >/dev/null
done

if [[ "$MODE" == update ]]; then
  git add -- "${lockfiles[@]}"
fi

echo "Validated ${#lockfiles[@]} auxiliary rscrypto lockfiles"
