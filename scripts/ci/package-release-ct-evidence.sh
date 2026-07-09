#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/ci/package-release-ct-evidence.sh --version VERSION [--input DIR] [--out DIR]

Packages CT lane artifacts downloaded by actions/download-artifact into a
single release bundle and emits GitHub Actions outputs for the bundle path,
name, and SHA-256.
EOF
}

version=""
input_dir="ct-release-artifacts"
out_dir="release-artifacts"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      version="${2:?missing value for --version}"
      shift 2
      ;;
    --input)
      input_dir="${2:?missing value for --input}"
      shift 2
      ;;
    --out)
      out_dir="${2:?missing value for --out}"
      shift 2
      ;;
    -h|--help)
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

if [[ -z "$version" ]]; then
  echo "--version is required" >&2
  exit 2
fi

if [[ ! -d "$input_dir" ]]; then
  echo "CT evidence artifact directory missing: $input_dir" >&2
  exit 1
fi

mapfile -t reports < <(find "$input_dir" -type f \( -name 'ct-report-*.json' -o -name 'ct-report-*.md' \) | sort)
if [[ "${#reports[@]}" -eq 0 ]]; then
  echo "no CT reports found under $input_dir" >&2
  exit 1
fi

mkdir -p "$out_dir"
staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT

bundle_name="rscrypto-${version}-ct-evidence.tar.gz"
bundle_path="$out_dir/$bundle_name"
manifest="$staging/CT-EVIDENCE-MANIFEST.txt"

mkdir -p "$staging/ct-evidence"
cp -R "$input_dir"/. "$staging/ct-evidence/"

(
  cd "$staging"
  find ct-evidence -type f -print0 | sort -z | xargs -0 sha256sum
) > "$manifest"

tar -czf "$bundle_path" -C "$staging" CT-EVIDENCE-MANIFEST.txt ct-evidence
bundle_sha256="$(sha256sum "$bundle_path" | awk '{print $1}')"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "bundle_path=$bundle_path"
    echo "bundle_name=$bundle_name"
    echo "bundle_sha256=$bundle_sha256"
  } >> "$GITHUB_OUTPUT"
fi

echo "CT evidence bundle: $bundle_path"
echo "sha256: $bundle_sha256"
