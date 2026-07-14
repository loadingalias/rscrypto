#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

prepare_recipe=$(cd "$REPO_ROOT" && just --show release-prepare)
tag_recipe=$(cd "$REPO_ROOT" && just --show release-tag)

grep -Fq "cargo rail release check rscrypto --extended" <<<"$prepare_recipe"
grep -Fq "RSCRYPTO_RELEASE_PUSH=1 cargo rail release run rscrypto --bump auto --yes --skip-publish --skip-tag" \
  <<<"$prepare_recipe"
grep -Fq "cargo rail release finalize rscrypto --yes --skip-publish" <<<"$tag_recipe"

if grep -Fq "cargo rail release check" <<<"$tag_recipe"; then
  echo "release-tag must not require consumed change files" >&2
  exit 1
fi

echo "Release recipe regression tests passed"
