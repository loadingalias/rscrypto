#!/usr/bin/env bash
# shellcheck disable=SC2034
# Package arrays are caller-visible outputs for sourced fuzz scripts.

FUZZ_ROOT="${FUZZ_ROOT:-$REPO_ROOT/fuzz}"
FUZZ_SCOPED_ROOT="${FUZZ_SCOPED_ROOT:-$REPO_ROOT/fuzz-packages}"
FUZZ_SHARED_TARGET_DIR="${RSCRYPTO_FUZZ_TARGET_DIR:-$FUZZ_ROOT/target}"

FUZZ_ALL_PACKAGES=()
FUZZ_FULL_PACKAGES=()
FUZZ_SCOPED_PACKAGES=()
SELECTED_FUZZ_PACKAGES=()

fuzz_host_target() {
  rustc -vV | sed -n 's|host: ||p'
}

fuzz_package_label() {
  local package_dir=$1
  if [[ "$package_dir" == "$FUZZ_ROOT" ]]; then
    echo "full"
    return 0
  fi

  echo "scoped/${package_dir#"$FUZZ_SCOPED_ROOT"/}"
}

discover_fuzz_packages() {
  FUZZ_ALL_PACKAGES=()
  FUZZ_FULL_PACKAGES=()
  FUZZ_SCOPED_PACKAGES=()

  local manifest package_dir
  while IFS= read -r manifest; do
    if ! grep -Eq 'cargo-fuzz[[:space:]]*=[[:space:]]*true' "$manifest"; then
      continue
    fi

    package_dir="$(dirname "$manifest")"
    FUZZ_ALL_PACKAGES+=("$package_dir")
    if [[ "$package_dir" == "$FUZZ_ROOT" ]]; then
      FUZZ_FULL_PACKAGES+=("$package_dir")
    else
      FUZZ_SCOPED_PACKAGES+=("$package_dir")
    fi
  done < <(
    find "$FUZZ_ROOT" -maxdepth 1 -type f -name Cargo.toml
    if [[ -d "$FUZZ_SCOPED_ROOT" ]]; then
      find "$FUZZ_SCOPED_ROOT" -mindepth 2 -maxdepth 2 -type f -name Cargo.toml | sort
    fi
  )
}

fuzz_select_packages() {
  local scope=$1
  SELECTED_FUZZ_PACKAGES=()

  case "$scope" in
    full)
      SELECTED_FUZZ_PACKAGES=("${FUZZ_FULL_PACKAGES[@]}")
      ;;
    scoped)
      SELECTED_FUZZ_PACKAGES=("${FUZZ_SCOPED_PACKAGES[@]}")
      ;;
    all)
      SELECTED_FUZZ_PACKAGES=("${FUZZ_ALL_PACKAGES[@]}")
      ;;
    *)
      echo "Unknown fuzz package scope: $scope" >&2
      return 1
      ;;
  esac
  [[ ${#SELECTED_FUZZ_PACKAGES[@]} -gt 0 ]] || {
    echo "No fuzz packages selected for scope: $scope" >&2
    return 2
  }
}

fuzz_in_package() {
  local package_dir=$1
  local subcommand=$2
  shift 2

  (
    cd "$REPO_ROOT" || exit
    if [[ "$subcommand" == build || "$subcommand" == run ]]; then
      cargo metadata --locked --no-deps --format-version 1 --manifest-path "$package_dir/Cargo.toml" >/dev/null
    fi
    CARGO_TARGET_DIR="$FUZZ_SHARED_TARGET_DIR" cargo fuzz "$subcommand" --fuzz-dir "$package_dir" "$@"
  )
}

fuzz_list_targets() {
  local package_dir=$1
  local targets
  targets=$(fuzz_in_package "$package_dir" list) || return 2
  [[ -n "${targets//[[:space:]]/}" ]] || {
    echo "No fuzz targets discovered in $package_dir" >&2
    return 2
  }
  printf '%s\n' "$targets"
}
