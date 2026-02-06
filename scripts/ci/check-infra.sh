#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧱 Infrastructure Correctness Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! command -v actionlint >/dev/null 2>&1; then
  echo "error: actionlint is required but not installed" >&2
  exit 1
fi

echo ""
echo "1) actionlint"
actionlint

echo ""
echo "2) Shell syntax lint (bash -n)"
while IFS= read -r script; do
  bash -n "$script"
done < <(git ls-files | rg '^scripts/.*\.sh$')

echo ""
echo "3) Workflow pin/schema policy"
scripts/ci/pin-actions.sh --verify-only

echo ""
echo "4) No masked failures in required workflows"
required_workflows=(
  ".github/workflows/fast-pr.yaml"
  ".github/workflows/commit.yaml"
)
for wf in "${required_workflows[@]}"; do
  if rg -n '\|\|[[:space:]]*true' "$wf" >/dev/null 2>&1; then
    echo "error: forbidden '|| true' in required workflow: $wf" >&2
    rg -n '\|\|[[:space:]]*true' "$wf" >&2
    exit 1
  fi
done

echo ""
echo "5) Dead script detection (.sh)"
while IFS= read -r script; do
  refs=$(rg -n --fixed-strings "$script" justfile .github/workflows scripts/README.md scripts 2>/dev/null \
    | grep -F -v "${script}:" || true)
  if [[ -z "$refs" ]]; then
    echo "error: unreferenced script: $script" >&2
    exit 1
  fi
done < <(git ls-files | rg '^scripts/.*\.sh$')

echo ""
echo "✅ Infrastructure checks passed"
