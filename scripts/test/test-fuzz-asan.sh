#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/fuzz-packages.sh
source "$SCRIPT_DIR/../lib/fuzz-packages.sh"

maybe_disable_sccache

PACKAGE_SCOPE="full"

usage() {
  cat >&2 <<EOF
Usage: scripts/test/test-fuzz-asan.sh [--full|--scoped|--all]

Replays committed fuzz corpora under AddressSanitizer.

Environment:
  RSCRYPTO_ASAN_TARGET_DIR     Cargo target dir for ASan builds
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      PACKAGE_SCOPE="full"
      ;;
    --scoped)
      PACKAGE_SCOPE="scoped"
      ;;
    --all)
      PACKAGE_SCOPE="all"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

if ! cargo fuzz --version &>/dev/null; then
  echo "cargo-fuzz is required for ASan fuzz replay" >&2
  exit 1
fi

discover_fuzz_packages
fuzz_select_packages "$PACKAGE_SCOPE"

export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:abort_on_error=1:strict_init_order=1}"
export RUSTFLAGS="${RUSTFLAGS:-} -Zsanitizer=address -Cforce-frame-pointers=yes"
export RUSTDOCFLAGS="${RUSTDOCFLAGS:-} -Zsanitizer=address"
export CARGO_TARGET_DIR="${RSCRYPTO_ASAN_TARGET_DIR:-$FUZZ_ROOT/target-asan}"

failed=0
total=0
for package_dir in "${SELECTED_FUZZ_PACKAGES[@]:+${SELECTED_FUZZ_PACKAGES[@]}}"; do
  package_failed=0
  while IFS= read -r target; do
    [[ -z "$target" ]] && continue
    corpus_dir="$package_dir/corpus/$target"
    if [[ ! -d "$corpus_dir" ]]; then
      echo "missing ASan corpus: $(fuzz_package_label "$package_dir")/$target ($corpus_dir)" >&2
      failed=1
      package_failed=1
      continue
    fi

    total=$((total + 1))
  done < <(fuzz_list_targets "$package_dir")

  if [[ "$package_failed" -ne 0 ]]; then
    continue
  fi

  echo "ASan corpus replay package: $(fuzz_package_label "$package_dir")"
  if ! cargo test \
    -Zbuild-std \
    --target "$(fuzz_host_target)" \
    --manifest-path "$package_dir/Cargo.toml" \
    --all-features \
    --test corpus_replay \
    -- --nocapture; then
    failed=1
  fi
done

if [[ "$total" -eq 0 ]]; then
  echo "no fuzz targets selected for ASan replay" >&2
  exit 1
fi

if [[ "$failed" -ne 0 ]]; then
  echo "ASan fuzz corpus replay failed" >&2
  exit 1
fi

echo "ASan fuzz corpus replay passed for $total targets"
