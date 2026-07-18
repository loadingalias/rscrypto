#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/ci/package-release-source.sh --version VERSION --tag TAG --commit SHA [options]

Build a deterministic source archive from an annotated release tag.

Options:
  --version VERSION   Unprefixed crate version
  --tag TAG           Release tag (must equal vVERSION)
  --commit SHA        Full commit expected beneath TAG
  --out DIR           Output directory (default: target/package)
  --root PATH         Repository root (default: current repository)
  -h, --help          Show this help
EOF
}

version=""
tag=""
commit=""
out_dir="target/package"
root="$(git rev-parse --show-toplevel)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      version=${2:?missing value for --version}
      shift 2
      ;;
    --tag)
      tag=${2:?missing value for --tag}
      shift 2
      ;;
    --commit)
      commit=${2:?missing value for --commit}
      shift 2
      ;;
    --out)
      out_dir=${2:?missing value for --out}
      shift 2
      ;;
    --root)
      root=${2:?missing value for --root}
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

[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?$ ]] || {
  echo "source archive error: --version must be an unprefixed SemVer version" >&2
  exit 2
}
[[ "$tag" == "v$version" ]] || {
  echo "source archive error: tag $tag does not match version $version" >&2
  exit 1
}
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || {
  echo "source archive error: --commit must be a full lowercase Git commit" >&2
  exit 2
}

git -C "$root" cat-file -e "$commit^{commit}" 2>/dev/null || {
  echo "source archive error: commit is not present: $commit" >&2
  exit 1
}
git -C "$root" rev-parse -q --verify "$tag^{tag}" >/dev/null || {
  echo "source archive error: release ref must be an annotated tag: $tag" >&2
  exit 1
}
tag_commit=$(git -C "$root" rev-parse "$tag^{commit}")
[[ "$tag_commit" == "$commit" ]] || {
  echo "source archive error: tag $tag resolves to $tag_commit, not $commit" >&2
  exit 1
}

mkdir -p "$out_dir"
archive_name="rscrypto-${version}-source.tar.gz"
archive_path="$out_dir/$archive_name"
archive_tmp="${archive_path}.tmp.$$"
contents=$(mktemp)
trap 'rm -f "$archive_tmp" "$contents"' EXIT

git -C "$root" archive --format=tar --prefix="rscrypto-${version}/" "$commit" | gzip -n -9 > "$archive_tmp"
tar -tzf "$archive_tmp" > "$contents"

if awk -v prefix="rscrypto-${version}/" 'index($0, prefix) != 1 { exit 1 }' "$contents"; then
  :
else
  echo "source archive error: archive contains a path outside rscrypto-${version}/" >&2
  exit 1
fi

for required in Cargo.toml Cargo.lock rust-toolchain.toml .github/workflows/release.yaml; do
  grep -Fxq "rscrypto-${version}/$required" "$contents" || {
    echo "source archive error: archive is missing $required" >&2
    exit 1
  }
done

mv "$archive_tmp" "$archive_path"
trap 'rm -f "$contents"' EXIT

if command -v sha256sum >/dev/null 2>&1; then
  source_sha256=$(sha256sum "$archive_path" | awk '{print $1}')
else
  source_sha256=$(shasum -a 256 "$archive_path" | awk '{print $1}')
fi

if [[ -n ${GITHUB_OUTPUT:-} ]]; then
  {
    echo "source_path=$archive_path"
    echo "source_name=$archive_name"
    echo "source_sha256=$source_sha256"
  } >> "$GITHUB_OUTPUT"
fi

echo "source archive: $archive_path"
echo "sha256:        $source_sha256"
