#!/usr/bin/env bash
set -euo pipefail
unset BASH_ENV

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAPTURE="$SCRIPT_DIR/capture-cache-status.sh"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

fail() {
  echo "cache status regression failure: $*" >&2
  exit 1
}

BIN="$TMP_ROOT/bin"
mkdir -p "$BIN"
cat >"$BIN/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ "$*" == "rail cache status --scope local --format json" ]]
cat "$RSCRYPTO_MOCK_CACHE_STATUS"
EOF
chmod +x "$BIN/cargo"

cat >"$TMP_ROOT/healthy.json" <<'EOF'
{
  "command": "cache",
  "exit_code": 0,
  "mode": "status",
  "result": "success",
  "schema_version": 1,
  "scope": "local",
  "status": {
    "installation": {
      "cache_base": "/secret/cache-base",
      "cargo_home": "/secret/cargo-home",
      "config_path": "/secret/config.toml",
      "healthy": true,
      "issues": [],
      "max_bytes": 10737418240,
      "root_portability": "remap",
      "state": "installed",
      "usage": {
        "bypasses": 3,
        "failures": 0,
        "hits": 5,
        "misses": 7
      },
      "wrapper_path": "/secret/cargo-rail-native-rustc-wrapper"
    },
    "local": {
      "cache": {
        "native_conflicted": 0,
        "native_local_origins": 11,
        "native_remote_origins": 13,
        "root": "/secret/local-cas-v2"
      },
      "cross_workspace": true,
      "present": true
    },
    "remote": {
      "activation": "direct_transport_selected",
      "authority": "remote-authority-v1-sha256-test",
      "mode": "read",
      "protocol": "native-v6",
      "provider": "cloudflare-r2",
      "shared_environment_names": 3
    },
    "schema_version": 14
  }
}
EOF

run_capture() {
  local case_root=$1
  local status=$2
  mkdir -p "$case_root"
  (
    cd "$case_root"
    PATH="$BIN:$PATH" \
      GITHUB_OUTPUT="$case_root/github-output" \
      RSCRYPTO_CI_OPERATION=native \
      RSCRYPTO_CI_PLATFORM=linux \
      RSCRYPTO_CI_RUNNER=ubuntu-latest \
      RSCRYPTO_CI_TARGET=x86_64-unknown-linux-gnu \
      RSCRYPTO_MOCK_CACHE_STATUS="$status" \
      bash "$CAPTURE"
  )
}

healthy_case="$TMP_ROOT/healthy-case"
run_capture "$healthy_case" "$TMP_ROOT/healthy.json"
output="$healthy_case/target/cargo-rail/cache-status.json"
[[ -f "$output" ]] || fail "healthy status did not produce telemetry"
[[ ! -e "$output.raw" ]] || fail "healthy status retained its raw input"
[[ $(<"$healthy_case/github-output") == \
  "artifact_name=cargo-rail-cache-native-x86_64-unknown-linux-gnu" ]] \
  || fail "cache telemetry artifact identity is not deterministic"

jq -e '
  .status.installation.healthy == true and
  .status.installation.root_portability == "remap" and
  .status.installation.usage.misses == 7 and
  .status.installation.usage.bypasses == 3 and
  .status.installation.usage.failures == 0 and
  .status.local.cache.native_remote_origins == 13 and
  .status.local.cache.native_conflicted == 0 and
  .status.remote.provider == "cloudflare-r2" and
  .status.remote.mode == "read" and
  .status.remote.protocol == "native-v6"
' "$output" >/dev/null || fail "required cache telemetry was not preserved"

if rg -n '/secret/' "$output" >/dev/null; then
  fail "cache telemetry disclosed a machine-local path"
fi
for field in cache_base cargo_home config_path wrapper_path; do
  jq -e --arg field "$field" '.status.installation | has($field) | not' "$output" >/dev/null \
    || fail "cache telemetry retained installation.$field"
done
jq -e '.status.local.cache | has("root") | not' "$output" >/dev/null \
  || fail "cache telemetry retained the local CAS root"

runner_case="$TMP_ROOT/runner-case"
mkdir -p "$runner_case"
(
  cd "$runner_case"
  PATH="$BIN:$PATH" \
    GITHUB_OUTPUT="$runner_case/github-output" \
    RSCRYPTO_CI_OPERATION=msrv \
    RSCRYPTO_CI_RUNNER='runs-on=123/runner=linux x64 ci' \
    RSCRYPTO_MOCK_CACHE_STATUS="$TMP_ROOT/healthy.json" \
    bash "$CAPTURE"
)
[[ $(<"$runner_case/github-output") == \
  "artifact_name=cargo-rail-cache-msrv-runs-on-123-runner-linux-x64-ci" ]] \
  || fail "cache telemetry did not normalize a runner label portably"

expect_status_failure() {
  local name=$1
  local filter=$2
  local status="$TMP_ROOT/$name.json"
  local case_root="$TMP_ROOT/$name-case"
  jq "$filter" "$TMP_ROOT/healthy.json" >"$status"
  if run_capture "$case_root" "$status" >/dev/null 2>&1; then
    fail "$name status was accepted"
  fi
  [[ ! -e "$case_root/target/cargo-rail/cache-status.json" ]] \
    || fail "$name status retained a publishable artifact"
  [[ ! -e "$case_root/target/cargo-rail/cache-status.json.raw" ]] \
    || fail "$name status retained its raw input"
  [[ ! -e "$case_root/target/cargo-rail/cache-status.json.projected" ]] \
    || fail "$name status retained its projected input"
}

expect_status_failure unhealthy '.status.installation.healthy = false'
expect_status_failure compiler-failure '.status.installation.usage.failures = 1'
expect_status_failure conflict '.status.local.cache.native_conflicted = 1'

echo "Cache status regression tests passed"
