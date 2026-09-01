#!/usr/bin/env bash
# Install the exact prebuilt actionlint release used by the Actions policy lane.

set -euo pipefail

version=1.7.12
archive="actionlint_${version}_linux_amd64.tar.gz"
url="https://github.com/rhysd/actionlint/releases/download/v${version}/${archive}"
sha256=8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8
temp_root=${RUNNER_TEMP:?RUNNER_TEMP is required}
bin_dir="$temp_root/rscrypto-actionlint/bin"
download_dir=$(mktemp -d "$temp_root/rscrypto-actionlint-download.XXXXXX")
trap 'rm -rf "$download_dir"' EXIT

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
  echo "actionlint installer requires Linux x86-64" >&2
  exit 1
}
mkdir -p "$bin_dir"
curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location \
  --retry 3 --retry-delay 2 --output "$download_dir/$archive" "$url"
printf '%s  %s\n' "$sha256" "$download_dir/$archive" | sha256sum --check --status
tar -xzf "$download_dir/$archive" -C "$download_dir" actionlint
install -m 755 "$download_dir/actionlint" "$bin_dir/actionlint"
[[ "$("$bin_dir/actionlint" -version)" == "$version" ]] || {
  echo "actionlint version check failed" >&2
  exit 1
}
printf '%s\n' "$bin_dir" >>"${GITHUB_PATH:?GITHUB_PATH is required}"
