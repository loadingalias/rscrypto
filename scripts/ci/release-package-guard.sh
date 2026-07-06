#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/ci/release-package-guard.sh [options]

Build and validate the crates.io package artifact for a release.

Options:
  --crate NAME              Crate to package (default: rscrypto)
  --expected-version VER    Require Cargo.toml package version to match VER
  --expected-git-sha SHA    Require .cargo_vcs_info.json git.sha1 to match SHA
  --package-path PATH       Validate an existing .crate instead of running cargo package
  -h, --help                Show this help
EOF
}

crate="rscrypto"
expected_version=""
expected_git_sha=""
package_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --crate)
      crate="${2:?missing value for --crate}"
      shift 2
      ;;
    --expected-version)
      expected_version="${2:?missing value for --expected-version}"
      shift 2
      ;;
    --expected-git-sha)
      expected_git_sha="${2:?missing value for --expected-git-sha}"
      shift 2
      ;;
    --package-path)
      package_path="${2:?missing value for --package-path}"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if command -v python3 >/dev/null 2>&1; then
  python_bin="python3"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
else
  echo "python3 or python is required" >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  sha256_file() {
    sha256sum "$1" | awk '{print $1}'
  }
else
  sha256_file() {
    shasum -a 256 "$1" | awk '{print $1}'
  }
fi

metadata="$(cargo metadata --no-deps --format-version 1)"
crate_version="$(
  METADATA="$metadata" "$python_bin" - "$crate" <<'PY'
import json
import os
import sys

metadata = json.loads(os.environ["METADATA"])
crate = sys.argv[1]
for package in metadata["packages"]:
    if package["name"] == crate:
        print(package["version"])
        break
else:
    raise SystemExit(f"crate not found in cargo metadata: {crate}")
PY
)"

if [[ -n "$expected_version" && "$crate_version" != "$expected_version" ]]; then
  echo "crate version $crate_version does not match expected $expected_version" >&2
  exit 1
fi

if [[ -z "$expected_git_sha" ]]; then
  expected_git_sha="$(git rev-parse HEAD)"
fi

if [[ -z "$package_path" ]]; then
  package_path="target/package/${crate}-${crate_version}.crate"
  rm -f "$package_path"
  cargo package --locked -p "$crate"
fi

if [[ ! -f "$package_path" ]]; then
  echo "package artifact missing: $package_path" >&2
  exit 1
fi

package_root="${crate}-${crate_version}"
vcs_json="$(tar -xOf "$package_path" "${package_root}/.cargo_vcs_info.json" 2>/dev/null || true)"
if [[ -z "$vcs_json" ]]; then
  echo "package is missing ${package_root}/.cargo_vcs_info.json" >&2
  exit 1
fi

VCS_JSON="$vcs_json" "$python_bin" - "$expected_git_sha" <<'PY'
import json
import os
import sys

expected_git_sha = sys.argv[1]
data = json.loads(os.environ["VCS_JSON"])
git = data.get("git") or {}
actual = git.get("sha1")
if actual != expected_git_sha:
    raise SystemExit(f"package git sha {actual!r} does not match expected {expected_git_sha!r}")
if git.get("dirty") is True:
    raise SystemExit("package was built from a dirty working tree")
path_in_vcs = data.get("path_in_vcs")
if path_in_vcs not in ("", None):
    raise SystemExit(f"unexpected package path_in_vcs: {path_in_vcs!r}")
PY

contents="$(mktemp)"
trap 'rm -f "$contents"' EXIT
tar -tzf "$package_path" > "$contents"

forbidden=(
  '(^|/)\.DS_Store$'
  "^${package_root}/AGENTS\\.md$"
  "^${package_root}/\\.agents(/|$)"
  "^${package_root}/\\.claude(/|$)"
  "^${package_root}/\\.codex(/|$)"
  "^${package_root}/\\.zed(/|$)"
  "^${package_root}/assets/distribution(/|$)"
  "^${package_root}/docs/.*/notes(/|$)"
  "^${package_root}/docs/funding(/|$)"
  "^${package_root}/docs/tasks(/|$)"
  "^${package_root}/docs/llvm-ppc64le-readvolatile-issue\\.md$"
  "^${package_root}/.*\\.(pem|key)$"
  '(~|\.swp)$'
)

violations="$(mktemp)"
trap 'rm -f "$contents" "$violations"' EXIT
: > "$violations"

for pattern in "${forbidden[@]}"; do
  grep -E "$pattern" "$contents" >> "$violations" || true
done

if [[ -s "$violations" ]]; then
  echo "package contains forbidden release files:" >&2
  sort -u "$violations" >&2
  exit 1
fi

crate_sha256="$(sha256_file "$package_path")"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "crate_path=$package_path"
    echo "crate_name=$(basename "$package_path")"
    echo "crate_sha256=$crate_sha256"
    echo "crate_version=$crate_version"
  } >> "$GITHUB_OUTPUT"
fi

echo "package: $package_path"
echo "sha256:  $crate_sha256"
