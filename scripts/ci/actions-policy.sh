#!/usr/bin/env bash
# Fast, dependency-light policy for the checked-in GitHub Actions surface.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

cargo rail config validate --strict
scripts/ci/check-action-pins.sh

while IFS= read -r -d '' script; do
  bash -n "$script"
done < <(find scripts -type f -name '*.sh' -print0)

ruby -e '
  require "yaml"
  Dir[".github/**/*.{yaml,yml}"].sort.each do |path|
    YAML.safe_load(File.read(path), aliases: true, filename: path)
  end
'

scripts/ci/check-action-pins-test.sh
scripts/ci/check-locked-cargo.sh
scripts/ci/check-locked-cargo-test.sh
scripts/ci/install-tools-mode-test.sh
scripts/lib/python-resolver-test.sh
bash -c 'source scripts/lib/ci-tool-integrity.sh; ci_tool_validate_manifest'
scripts/ci/remote-cache-recipes-test.sh
scripts/bench/bench-wrapper-test.sh
scripts/ci/seal-remote-evidence-test.sh
scripts/ci/feature-contracts-test.sh
scripts/ci/feature-planning-test.sh
scripts/ci/assurance-planning-test.sh
scripts/test/test-fuzz-scheduler-test.sh
scripts/ci/emit-manual-matrix-test.sh
scripts/ci/changed-test-planning-test.sh
scripts/ci/check-worktree-test.sh
scripts/ci/pre-push-test.sh
scripts/ci/release-identity-test.sh
scripts/ci/publish-immutable-release-test.sh
actionlint
zizmor .github/workflows .github/actions
