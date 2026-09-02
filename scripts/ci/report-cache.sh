#!/usr/bin/env bash
# Emit bounded, credential-free Cargo Rail compiler-cache telemetry.

set -euo pipefail

[[ $# -eq 0 ]] || { echo "Usage: $0" >&2; exit 2; }

status=""
if ! status=$(cargo rail cache status --scope local --format json); then
  echo "::warning::Cargo Rail cache status is unavailable"
  exit 0
fi

if ! jq -e '.result == "success" and .status.installation.healthy == true' <<<"$status" >/dev/null; then
  echo "::warning::Cargo Rail cache status is unhealthy or incompatible"
  exit 0
fi

jq '{
  healthy: .status.installation.healthy,
  usage: .status.installation.usage | {
    hits,
    misses,
    failures,
    bypasses,
    early_bypasses
  },
  local: .status.local | {
    present,
    cross_workspace,
    bytes: .cache.bytes,
    results: .cache.results,
    native_local_origins: .cache.native_local_origins,
    native_remote_origins: .cache.native_remote_origins
  },
  remote: .status.remote | {
    provider,
    mode,
    activation
  }
}' <<<"$status"
