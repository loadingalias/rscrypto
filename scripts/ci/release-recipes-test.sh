#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

prepare_recipe=$(cd "$REPO_ROOT" && just --show release-prepare)
tag_recipe=$(cd "$REPO_ROOT" && just --show release-tag)
controls_recipe=$(cd "$REPO_ROOT" && just --show check-repository-controls)
push_recipe=$(cd "$REPO_ROOT" && just --dry-run push 2>&1)
push_full_recipe=$(cd "$REPO_ROOT" && just --dry-run push-full 2>&1)

grep -Fq "cargo rail release check rscrypto --extended" <<<"$prepare_recipe"
grep -Fq "cargo rail release run rscrypto --bump auto --yes --pr" <<<"$prepare_recipe"
grep -Fq "git push" <<<"$prepare_recipe"
grep -Fq "cargo rail release finalize rscrypto --yes --skip-publish" <<<"$tag_recipe"
# shellcheck disable=SC2016 # Match the literal command rendered by just.
grep -Fq 'scripts/ci/release-evidence-check.sh --commit "$(git rev-parse HEAD)"' <<<"$tag_recipe"
grep -Fq 'just check-repository-controls' <<<"$tag_recipe"
grep -Fq 'scripts/ci/repository-controls-evidence.sh' <<<"$controls_recipe"
if grep -Fq -- '--allow-redacted-bypass' <<<"$controls_recipe"; then
  echo "the pre-tag repository controls gate must require full bypass visibility" >&2
  exit 1
fi

controls_line=$(grep -nF 'just check-repository-controls' <<<"$tag_recipe" | cut -d: -f1)
evidence_line=$(grep -nF 'scripts/ci/release-evidence-check.sh' <<<"$tag_recipe" | cut -d: -f1)
finalize_line=$(grep -nF 'cargo rail release finalize' <<<"$tag_recipe" | cut -d: -f1)
if (( controls_line >= evidence_line || evidence_line >= finalize_line )); then
  echo "release-tag must validate repository controls and exact-commit evidence before creating the tag" >&2
  exit 1
fi

if grep -Fq "cargo rail release check" <<<"$tag_recipe"; then
  echo "release-tag must not require consumed change files" >&2
  exit 1
fi

grep -Fq 'scripts/ci/pre-push.sh --light' <<<"$push_recipe"
grep -Fq 'git push --set-upstream "origin" HEAD' <<<"$push_recipe"
grep -Fq 'scripts/ci/pre-push.sh --full' <<<"$push_full_recipe"
grep -Fq 'git push --set-upstream "origin" HEAD' <<<"$push_full_recipe"
if grep -Fq -- '--no-verify' <<<"$push_recipe$push_full_recipe"; then
  echo "supported push recipes must not bypass Git hooks" >&2
  exit 1
fi
if grep -Fq 'RSCRYPTO_PRE_PUSH_' <<<"$push_recipe$push_full_recipe"; then
  echo "supported push recipes must not depend on hook coordination state" >&2
  exit 1
fi

echo "Release recipe regression tests passed"
