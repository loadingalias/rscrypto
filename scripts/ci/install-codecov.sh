#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/ci-tool-integrity.sh
source "$SCRIPT_DIR/../lib/ci-tool-integrity.sh"

runner_temp=${RUNNER_TEMP:?RUNNER_TEMP is required}
install_dir=$(mktemp -d "$runner_temp/rscrypto-codecov.XXXXXX")
ci_tool_download codecov "$install_dir"
chmod +x "$CI_TOOL_ARCHIVE_PATH"

version_output=$("$CI_TOOL_ARCHIVE_PATH" --version)
expected_version=${CI_TOOL_VERSION#v}
if [[ "$version_output" =~ ([0-9]+\.[0-9]+\.[0-9]+) ]]; then
  installed_version=${BASH_REMATCH[1]}
else
  echo "CodeCov CLI version mismatch: expected $expected_version, got $version_output" >&2
  exit 1
fi
[[ "$installed_version" == "$expected_version" ]] || {
  echo "CodeCov CLI version mismatch: expected $expected_version, got $installed_version" >&2
  exit 1
}

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  printf 'binary=%s\n' "$CI_TOOL_ARCHIVE_PATH" >>"$GITHUB_OUTPUT"
fi
