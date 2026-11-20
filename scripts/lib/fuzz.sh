#!/usr/bin/env bash
# Shared helpers for the cargo-fuzz package in fuzz/. Callers set REPO_ROOT.

FUZZ_ROOT="${FUZZ_ROOT:-$REPO_ROOT/fuzz}"
FUZZ_SHARED_TARGET_DIR="${RSCRYPTO_FUZZ_TARGET_DIR:-$FUZZ_ROOT/target}"

fuzz_host_target() {
  rustc -vV | sed -n 's|host: ||p'
}

fuzz_cargo() {
  local subcommand=$1
  shift

  (
    cd "$REPO_ROOT" || exit
    if [[ "$subcommand" == build || "$subcommand" == run ]]; then
      cargo metadata --locked --no-deps --format-version 1 --manifest-path "$FUZZ_ROOT/Cargo.toml" >/dev/null
    fi
    CARGO_TARGET_DIR="$FUZZ_SHARED_TARGET_DIR" cargo fuzz "$subcommand" --fuzz-dir "$FUZZ_ROOT" "$@"
  )
}

fuzz_list_targets() {
  local targets
  targets=$(fuzz_cargo list) || return 2
  [[ -n "${targets//[[:space:]]/}" ]] || {
    echo "No fuzz targets discovered in $FUZZ_ROOT" >&2
    return 2
  }
  printf '%s\n' "$targets"
}
