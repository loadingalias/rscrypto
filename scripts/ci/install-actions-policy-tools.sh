#!/usr/bin/env bash
# Install exact prebuilt tools not provided by the Actions tool installer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/ci-tool-integrity.sh
source "$SCRIPT_DIR/../lib/ci-tool-integrity.sh"

temp_root=${RUNNER_TEMP:?RUNNER_TEMP is required}
bin_dir="$temp_root/rscrypto-actions-policy/bin"
download_dir=$(mktemp -d "$temp_root/rscrypto-actions-policy-download.XXXXXX")
trap 'rm -rf "$download_dir"' EXIT
mkdir -p "$bin_dir" "$download_dir/actionlint" "$download_dir/ripgrep"

ci_tool_download actionlint "$download_dir/actionlint"
actionlint_version=$CI_TOOL_VERSION
tar -xzf "$CI_TOOL_ARCHIVE_PATH" -C "$download_dir/actionlint" actionlint
install -m 755 "$download_dir/actionlint/actionlint" "$bin_dir/actionlint"

ci_tool_download ripgrep "$download_dir/ripgrep"
ripgrep_version=$CI_TOOL_VERSION
ripgrep_root="ripgrep-${ripgrep_version}-x86_64-unknown-linux-musl"
tar -xzf "$CI_TOOL_ARCHIVE_PATH" -C "$download_dir/ripgrep" "$ripgrep_root/rg"
install -m 755 "$download_dir/ripgrep/$ripgrep_root/rg" "$bin_dir/rg"

installed_actionlint_version=$("$bin_dir/actionlint" -version)
installed_actionlint_version=${installed_actionlint_version%%$'\n'*}
[[ "$installed_actionlint_version" == "$actionlint_version" ]] || {
  echo "actionlint version check failed: expected $actionlint_version, got $installed_actionlint_version" >&2
  exit 1
}

installed_ripgrep_report=$("$bin_dir/rg" --version)
installed_ripgrep_report=${installed_ripgrep_report%%$'\n'*}
read -r installed_ripgrep_name installed_ripgrep_version _ <<<"$installed_ripgrep_report"
[[ "$installed_ripgrep_name" == ripgrep && "$installed_ripgrep_version" == "$ripgrep_version" ]] || {
  echo "ripgrep version check failed: expected $ripgrep_version, got $installed_ripgrep_report" >&2
  exit 1
}

printf '%s\n' "$bin_dir" >>"${GITHUB_PATH:?GITHUB_PATH is required}"
