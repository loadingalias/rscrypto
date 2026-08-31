#!/usr/bin/env bash
set -euo pipefail

operation=${RSCRYPTO_CI_OPERATION:?RSCRYPTO_CI_OPERATION is required}
runner=${RSCRYPTO_CI_RUNNER:?RSCRYPTO_CI_RUNNER is required}
platform=${RSCRYPTO_CI_PLATFORM:-}
target=${RSCRYPTO_CI_TARGET:-}
output=target/cargo-rail/cache-status.json
raw=$output.raw
projected=$output.projected

mkdir -p "$(dirname "$output")"
rm -f "$output" "$raw" "$projected"
trap 'rm -f "$raw" "$projected"' EXIT
cargo rail cache status --scope local --format json >"$raw"

jq -S '
  if .status.installation.healthy != true then
    error("Cargo Rail cache setup is unhealthy")
  elif (.status.installation.usage.failures // 0) != 0 then
    error("Cargo Rail cache recorded compiler-cache failures")
  elif (.status.local.cache.native_conflicted // 0) != 0 then
    error("Cargo Rail cache recorded conflicting results")
  else
    del(
      .status.installation.cache_base,
      .status.installation.cargo_home,
      .status.installation.config_path,
      .status.installation.wrapper_path,
      .status.local.cache.root
    )
  end
' "$raw" >"$projected"
mv "$projected" "$output"
rm -f "$raw"
trap - EXIT

identity=$operation-${target:-${platform:-$runner}}
artifact_name=$(printf '%s' "$identity" | tr -cs '[:alnum:]_.-' '-' | sed 's/^-*//; s/-*$//')
[[ -n "$artifact_name" ]] || { echo "empty cache telemetry artifact name" >&2; exit 2; }
printf 'artifact_name=cargo-rail-cache-%s\n' "$artifact_name" >>"${GITHUB_OUTPUT:-/dev/null}"
