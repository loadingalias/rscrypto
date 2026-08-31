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
  jq -er --arg crate "$crate" '
    [.packages[] | select(.name == $crate) | .version]
    | if length == 1 then .[0] else error("crate not found exactly once in cargo metadata: \($crate)") end
  ' <<<"$metadata"
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

jq -e --arg expected "$expected_git_sha" '
  if (.git.sha1 // null) != $expected then
    error("package git sha \(.git.sha1 // null) does not match expected \($expected)")
  elif (.git.dirty // false) == true then
    error("package was built from a dirty working tree")
  elif (.path_in_vcs // "") != "" then
    error("unexpected package path_in_vcs: \(.path_in_vcs)")
  else
    true
  end
' <<<"$vcs_json" >/dev/null

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
