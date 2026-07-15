#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/release-ci-check.sh"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

mkdir -p "$TMP_ROOT/bin"
cat >"$TMP_ROOT/bin/gh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ "$1 $2" == "run list" ]]; then
  case "${FAKE_GH_MODE:-success}" in
    missing)
      echo '[]'
      ;;
    pending)
      echo '[{"databaseId":4242,"status":"in_progress","conclusion":"","url":"https://example.invalid/runs/4242","createdAt":"2026-07-15T00:00:00Z"}]'
      ;;
    failed)
      echo '[{"databaseId":4242,"status":"completed","conclusion":"failure","url":"https://example.invalid/runs/4242","createdAt":"2026-07-15T00:00:00Z"}]'
      ;;
    prior-success)
      echo '[
        {"databaseId":4242,"status":"completed","conclusion":"success","url":"https://example.invalid/runs/4242","createdAt":"2026-07-15T00:00:00Z"},
        {"databaseId":4243,"status":"completed","conclusion":"cancelled","url":"https://example.invalid/runs/4243","createdAt":"2026-07-15T01:00:00Z"}
      ]'
      ;;
    *)
      echo '[{"databaseId":4242,"status":"completed","conclusion":"success","url":"https://example.invalid/runs/4242","createdAt":"2026-07-15T00:00:00Z"}]'
      ;;
  esac
  exit 0
fi

if [[ "$1 $2" == "run view" ]]; then
  graph_conclusion=success
  if [[ ${FAKE_GH_MODE:-success} == graph-failed ]]; then
    graph_conclusion=failure
  fi
  cat <<JSON
{"jobs":[{"name":"CI Suite / Cargo Graph Assurance / run","conclusion":"${graph_conclusion}"}]}
JSON
  exit 0
fi

echo "unexpected gh invocation: $*" >&2
exit 2
EOF
chmod +x "$TMP_ROOT/bin/gh"

export PATH="$TMP_ROOT/bin:$PATH"
commit=0123456789abcdef0123456789abcdef01234567

output=$($CHECKER --commit "$commit" --repo loadingalias/rscrypto)
grep -Fq 'Exact-commit CI passed: https://example.invalid/runs/4242' <<<"$output"

output=$(FAKE_GH_MODE=prior-success "$CHECKER" --commit "$commit" --repo loadingalias/rscrypto)
grep -Fq 'Exact-commit CI passed: https://example.invalid/runs/4242' <<<"$output"

for mode in missing pending failed graph-failed; do
  if FAKE_GH_MODE=$mode "$CHECKER" --commit "$commit" --repo loadingalias/rscrypto >/dev/null 2>&1; then
    echo "release CI check accepted invalid mode: $mode" >&2
    exit 1
  fi
done

echo "Release CI regression tests passed"
