#!/usr/bin/env bash
# Shared utilities for repository scripts.

# Colors (disabled if not a terminal)
if [[ -t 1 ]]; then
  RED=$'\033[0;31m'
  GREEN=$'\033[0;32m'
  YELLOW=$'\033[0;33m'
  DIM=$'\033[0;90m'
  RESET=$'\033[0m'
else
  RED='' GREEN='' YELLOW='' DIM='' RESET=''
fi

step() {
  printf "  → %s..." "$1"
}

ok() {
  printf " ${GREEN}✓${RESET}\n"
}

fail() {
  printf " ${RED}✗${RESET}\n"
}

skip() {
  echo "  ${YELLOW}○${RESET} $1 ${DIM}($2)${RESET}"
}

show_error() {
  local log_file=$1
  echo ""
  echo "  ${RED}Error:${RESET}"
  if [[ -f "$log_file" ]]; then
    tail -40 "$log_file" | sed 's/^/    /'
  fi
  echo ""
}

# Parse args into CRATE_FLAGS and SCOPE_DESC.
# Usage: get_crate_flags "$@"
# Sets: CRATE_FLAGS, SCOPE_DESC
get_crate_flags() {
  CRATE_FLAGS=""
  SCOPE_DESC=""

  local all_flag=false
  local crates=()

  for arg in "$@"; do
    if [[ "$arg" == "--all" ]]; then
      all_flag=true
    else
      crates+=("$arg")
    fi
  done

  if [[ ${#crates[@]} -gt 0 ]]; then
    for crate in "${crates[@]}"; do
      CRATE_FLAGS="$CRATE_FLAGS -p $crate"
    done
    SCOPE_DESC="${crates[*]}"
    return 0
  fi

  if [[ "$all_flag" == true ]]; then
    CRATE_FLAGS="--workspace"
    SCOPE_DESC="workspace"
    return 0
  fi

  local since_arg=""
  if [[ -n "${RAIL_SINCE:-}" ]]; then
    since_arg="--since $RAIL_SINCE"
  fi

  # shellcheck disable=SC2086
  local affected
  affected=$(cargo rail affected $since_arg -f names-only 2>/dev/null || echo "")

  if [[ -z "$affected" ]]; then
    CRATE_FLAGS="--workspace"
    SCOPE_DESC="workspace (no changes)"
  else
    for crate in $affected; do
      CRATE_FLAGS="$CRATE_FLAGS -p $crate"
    done
    SCOPE_DESC="affected"
  fi
}

maybe_disable_sccache() {
  if [[ -n "${RUSTC_WRAPPER:-}" && "${RUSTC_WRAPPER##*/}" == "sccache" ]]; then
    if ! "$RUSTC_WRAPPER" rustc -vV >/dev/null 2>&1; then
      echo "WARNING: sccache is configured but not usable; disabling RUSTC_WRAPPER for this run."
      export RUSTC_WRAPPER=
    fi
  fi
}

ensure_target() {
  local target=$1
  if ! rustup target list --installed 2>/dev/null | grep -q "^${target}$"; then
    rustup target add "$target" >/dev/null 2>&1 || true
  fi
}
