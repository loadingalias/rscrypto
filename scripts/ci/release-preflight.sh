#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/ci/release-preflight.sh [options]

Validate a tag-triggered release before any publish step.

Options:
  --crate NAME   Crate to release (default: rscrypto)
  --tag TAG      Git tag to validate (default: GITHUB_REF_NAME)
  -h, --help     Show this help
EOF
}

crate="rscrypto"
tag="${GITHUB_REF_NAME:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --crate)
      crate="${2:?missing value for --crate}"
      shift 2
      ;;
    --tag)
      tag="${2:?missing value for --tag}"
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

if [[ -z "$tag" ]]; then
  echo "release tag is required" >&2
  exit 1
fi

case "$tag" in
  v[0-9]*.[0-9]*.[0-9]*)
    ;;
  *)
    echo "release tag must look like vMAJOR.MINOR.PATCH: $tag" >&2
    exit 1
    ;;
esac

if ! git rev-parse -q --verify "${tag}^{tag}" >/dev/null; then
  echo "release ref must be an annotated signed tag, not a lightweight tag: $tag" >&2
  exit 1
fi

git config gpg.ssh.allowedSignersFile "$PWD/.github/allowed-signers"
git tag -v "$tag"

tag_commit="$(git rev-list -n 1 "$tag")"
head_commit="$(git rev-parse HEAD)"
if [[ "$tag_commit" != "$head_commit" ]]; then
  echo "checked-out commit $head_commit does not match tag commit $tag_commit" >&2
  exit 1
fi

tag_version="${tag#v}"
if command -v python3 >/dev/null 2>&1; then
  python_bin="python3"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
else
  echo "python3 or python is required" >&2
  exit 1
fi

cargo_metadata="$(cargo metadata --no-deps --format-version 1)"
crate_version="$(
  METADATA="$cargo_metadata" "$python_bin" - "$crate" <<'PY'
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

if [[ "$crate_version" != "$tag_version" ]]; then
  echo "tag $tag does not match Cargo.toml version $crate_version" >&2
  exit 1
fi

if ! grep -qE "^## \\[$tag_version\\]" CHANGELOG.md; then
  echo "CHANGELOG.md is missing a release heading for $tag_version" >&2
  exit 1
fi

cargo rail config validate --strict
cargo rail config sync --check
cargo rail unify --check --explain
cargo deny check all
# RustCrypto `rsa` is used only as a dev/test/bench oracle. Production RSA
# verification is implemented in `src/auth/rsa.rs`; keep this scoped to the
# known Marvin advisory until the oracle dependency is removed or fixed.
cargo audit --ignore RUSTSEC-2023-0071
cargo semver-checks --package "$crate" --all-features

# `cargo rail release check` is a pre-tag gate. The release run consumes
# `.changes` files before creating the signed tag, so tag preflight
# validates the already-materialized release commit directly.

scripts/ci/release-package-guard.sh \
  --crate "$crate" \
  --expected-version "$tag_version" \
  --expected-git-sha "$tag_commit"
