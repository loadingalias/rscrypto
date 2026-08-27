#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

prepare_recipe=$(cd "$REPO_ROOT" && just --show release-prepare)
tag_recipe=$(cd "$REPO_ROOT" && just --show release-tag)
push_recipe=$(cd "$REPO_ROOT" && just --dry-run push 2>&1)

grep -Fq "cargo rail release check rscrypto --extended" <<<"$prepare_recipe"
grep -Fq "cargo rail release run rscrypto --bump auto --yes --pr" <<<"$prepare_recipe"
grep -Fq "git push" <<<"$prepare_recipe"
grep -Fq "cargo rail release finalize rscrypto --yes --skip-publish" <<<"$tag_recipe"
# shellcheck disable=SC2016 # Match the literal command rendered by just.
grep -Fq 'scripts/ci/release-evidence-check.sh --commit "$(git rev-parse HEAD)"' <<<"$tag_recipe"
grep -Fq 'scripts/ci/repository-controls-evidence.sh' <<<"$tag_recipe"
if grep -Fq -- '--allow-redacted-bypass' <<<"$tag_recipe"; then
  echo "the pre-tag repository controls gate must require full bypass visibility" >&2
  exit 1
fi

controls_line=$(grep -nF 'scripts/ci/repository-controls-evidence.sh' <<<"$tag_recipe" | cut -d: -f1)
evidence_line=$(grep -nF 'scripts/ci/release-evidence-check.sh' <<<"$tag_recipe" | cut -d: -f1)
finalize_line=$(grep -nF 'cargo rail release finalize' <<<"$tag_recipe" | cut -d: -f1)
if (( controls_line >= evidence_line || evidence_line >= finalize_line )); then
  echo "release-tag must validate repository controls and exact-commit evidence before creating the tag" >&2
  exit 1
fi

if grep -Fq "cargo rail release check" <<<"$tag_recipe" || grep -Fq "cargo rail unify" <<<"$tag_recipe"; then
  echo "release-tag must consume exact-commit evidence instead of repeating release preparation" >&2
  exit 1
fi

grep -Fq 'scripts/ci/pre-push.sh' <<<"$push_recipe"
grep -Fq 'git push --set-upstream origin HEAD' <<<"$push_recipe"
if (cd "$REPO_ROOT" && just --show push-full >/dev/null 2>&1); then
  echo "push-full must not exist; just push is the single supported push command" >&2
  exit 1
fi
if grep -Fq -- '--no-verify' <<<"$push_recipe"; then
  echo "supported push recipes must not bypass Git hooks" >&2
  exit 1
fi
if grep -Fq -- '--light' <<<"$push_recipe" || grep -Fq -- '--full' <<<"$push_recipe"; then
  echo "the supported push recipe must not expose validation profiles" >&2
  exit 1
fi

echo "Release recipe regression tests passed"
