#!/usr/bin/env bash
# Shared utilities for repository scripts.
# shellcheck disable=SC2034
# CARGO_ARGS, CARGO_SCOPE_KIND, and SCOPE_DESC are caller-visible outputs.

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=rail-plan.sh
source "$COMMON_DIR/rail-plan.sh"

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
  printf " %b✓%b\n" "$GREEN" "$RESET"
}

fail() {
  printf " %b✗%b\n" "$RED" "$RESET"
}

skip() {
  echo "  ${YELLOW}○${RESET} $1 ${DIM}($2)${RESET}"
}

show_error() {
  local log_file=$1
  echo ""
  echo "  ${RED}Error:${RESET}"
  if [[ -f "$log_file" ]]; then
    local error_pattern='error(\[[A-Z0-9]+\])?:|could not compile|panicked at|test result: FAILED|failures:|Caused by:'
    local printer=(cat)

    if command -v perl >/dev/null 2>&1; then
      # Strip ANSI escapes so grep/ripgrep can match the actual compiler text.
      printer=(perl -pe 's/\e\[[0-9;]*[[:alpha:]]//g')
    fi

    if command -v rg >/dev/null 2>&1 && "${printer[@]}" "$log_file" | rg -n -m 1 "$error_pattern" >/dev/null 2>&1; then
      echo "    Relevant excerpt:"
      "${printer[@]}" "$log_file" | rg -n -m 8 -C 3 "$error_pattern" | sed 's/^/    /'
    else
      tail -40 "$log_file" | sed 's/^/    /'
    fi
  fi
  echo ""
}

# Select the exact Cargo arguments from one Cargo Rail work decision.
# Usage: select_cargo_scope WORK_ID [true]
# Returns 1 only when Cargo Rail selected no work.
select_cargo_scope() {
  local work_id=$1
  local force_all=${2:-false}
  local arg

  CARGO_ARGS=()
  CARGO_SCOPE_KIND=""
  SCOPE_DESC=""

  if [[ "$force_all" == true ]]; then
    CARGO_ARGS=(--workspace)
    CARGO_SCOPE_KIND=workspace
    SCOPE_DESC=workspace
    return 0
  fi

  # Prime in the caller shell so subsequent command/process substitutions reuse
  # the same authenticated plan instead of replanning in isolated subshells.
  rail_prime_plan || true
  CARGO_SCOPE_KIND="$(rail_scope_mode "$work_id")"

  case "$CARGO_SCOPE_KIND" in
    empty)
      SCOPE_DESC="no changes"
      return 1
      ;;
    workspace)
      CARGO_ARGS=(--workspace)
      SCOPE_DESC="workspace (Cargo Rail)"
      ;;
    packages)
      while IFS= read -r arg; do
        [[ -n "$arg" ]] && CARGO_ARGS+=("$arg")
      done < <(rail_scope_cargo_args "$work_id")
      if [[ ${#CARGO_ARGS[@]} -eq 0 ]]; then
        echo "ERROR: Cargo Rail selected packages without Cargo arguments for $work_id" >&2
        return 2
      fi
      SCOPE_DESC="affected packages (Cargo Rail)"
      ;;
    *)
      echo "ERROR: unsupported Cargo Rail scope '$CARGO_SCOPE_KIND' for $work_id" >&2
      return 2
      ;;
  esac
}

apply_ci_resource_profile() {
  case "${RSCRYPTO_CI_RESOURCE_PROFILE:-}" in
    "" | default)
      return 0
      ;;
    constrained)
      local arch
      arch="$(uname -m 2>/dev/null || echo unknown)"

      if [[ -z "${CARGO_BUILD_JOBS:-}" ]]; then
        export CARGO_BUILD_JOBS=2
      fi

      if [[ -z "${RSCRYPTO_TEST_THREADS:-}" ]]; then
        export RSCRYPTO_TEST_THREADS="${RUST_TEST_THREADS:-1}"
      fi

      if [[ -z "${RUST_TEST_THREADS:-}" ]]; then
        export RUST_TEST_THREADS="$RSCRYPTO_TEST_THREADS"
      fi

      echo "CI resource profile: constrained (${arch}); CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}; test threads=${RSCRYPTO_TEST_THREADS}"
      ;;
    *)
      echo "WARNING: unknown RSCRYPTO_CI_RESOURCE_PROFILE=${RSCRYPTO_CI_RESOURCE_PROFILE}; ignoring."
      ;;
  esac
}

ensure_target() {
  local target=$1
  local toolchain=${2:-}
  if [[ -n "$toolchain" ]]; then
    if ! rustup target list --toolchain "$toolchain" --installed 2>/dev/null | grep -q "^${target}$"; then
      rustup target add --toolchain "$toolchain" "$target" >/dev/null 2>&1 || true
    fi
  elif ! rustup target list --installed 2>/dev/null | grep -q "^${target}$"; then
    rustup target add "$target" >/dev/null 2>&1 || true
  fi
}

activate_nightly_toolchain() {
  local toolchain_script="$COMMON_DIR/toolchain.sh"
  local toolchain_contracts="$COMMON_DIR/../../.config/toolchains.toml"
  [[ -x "$toolchain_script" ]] || {
    echo "ERROR: nightly toolchain resolver not found: $toolchain_script" >&2
    return 1
  }
  [[ -f "$toolchain_contracts" ]] || {
    echo "ERROR: nightly toolchain authority not found: $toolchain_contracts" >&2
    return 1
  }
  RUSTUP_TOOLCHAIN=$("$toolchain_script" --nightly)
  export RUSTUP_TOOLCHAIN
}
