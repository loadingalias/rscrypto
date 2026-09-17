#!/usr/bin/env bash
set -euo pipefail

remote_args=()
if [[ -n ${CARGO_RAIL_CACHE_REMOTE:-} ]]; then
  if [[ -z ${CARGO_RAIL_CACHE_MODE:-} ]]; then
    printf '%s\n' 'CARGO_RAIL_CACHE_MODE is required with CARGO_RAIL_CACHE_REMOTE' >&2
    exit 2
  fi
  remote_args+=(
    --remote "$CARGO_RAIL_CACHE_REMOTE"
    --remote-mode "$CARGO_RAIL_CACHE_MODE"
    --root-portability remap
  )
elif [[ -n ${CARGO_RAIL_CACHE_MODE:-} ]]; then
  printf '%s\n' 'CARGO_RAIL_CACHE_MODE requires CARGO_RAIL_CACHE_REMOTE' >&2
  exit 2
fi

status=0
if [[ -n ${CARGO_RAIL_CACHE_REMOTE:-} ]]; then
  cargo rail cache setup --check "${remote_args[@]}" "$@" || status=$?
else
  cargo rail cache setup --check --local-only "$@" || status=$?
fi
if ((status > 1)); then
  exit "$status"
fi
if [[ -n ${CARGO_RAIL_CACHE_REMOTE:-} ]]; then
  cargo rail cache setup --local-only "$@"
  env -u CARGO_RAIL_CACHE_REMOTE -u CARGO_RAIL_CACHE_MODE \
    -u CARGO_RAIL_CACHE_REMOTE_ENVIRONMENT cargo rail cache ready
  cargo rail cache setup "${remote_args[@]}" "$@"
  cargo rail cache probe
else
  cargo rail cache setup --local-only "$@"
  cargo rail cache ready
fi
