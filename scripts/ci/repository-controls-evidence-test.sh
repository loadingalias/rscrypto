#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKER="$SCRIPT_DIR/repository-controls-evidence.sh"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

mkdir -p "$TMP_ROOT/bin"
cat >"$TMP_ROOT/bin/gh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

endpoint=${!#}
case "$endpoint" in
  repos/loadingalias/rscrypto)
    echo '{"id":1115910108,"full_name":"loadingalias/rscrypto","visibility":"public","default_branch":"main"}'
    ;;
  'repos/loadingalias/rscrypto/rulesets?includes_parents=true&per_page=100')
    echo '[{"id":19077982,"name":"protect-main","target":"branch","source_type":"Repository","source":"loadingalias/rscrypto","enforcement":"active"}]'
    ;;
  repos/loadingalias/rscrypto/rulesets/19077982)
    jq --arg mode "${FAKE_GH_MODE:-success}" '
      . + {
        id: 19077982,
        source_type: "Repository",
        source: "loadingalias/rscrypto",
        current_user_can_bypass: "never",
        created_at: "2026-07-16T21:26:51-04:00",
        updated_at: "2026-07-16T21:26:51-04:00"
      }
      | if $mode == "bypass" then
          .bypass_actors = [{actor_id: 5, actor_type: "RepositoryRole", bypass_mode: "always"}]
          | .current_user_can_bypass = "always"
        elif $mode == "inactive" then
          .enforcement = "disabled"
        elif $mode == "missing-check" then
          .rules |= map(select(.type != "required_status_checks"))
        elif $mode == "wrong-app" then
          (.rules[] | select(.type == "required_status_checks") | .parameters.required_status_checks[0].integration_id) = 42
        elif $mode == "redacted" then
          del(.bypass_actors, .current_user_can_bypass)
        elif $mode == "redacted-self" then
          del(.bypass_actors)
        else
          .
        end
    ' "$EXPECTED_POLICY"
    ;;
  'repos/loadingalias/rscrypto/rules/branches/main?per_page=100')
    jq --arg mode "${FAKE_GH_MODE:-success}" '
      [.rules[] | . + {
        ruleset_source_type: "Repository",
        ruleset_source: "loadingalias/rscrypto",
        ruleset_id: 19077982
      }]
      | if $mode == "wrong-effective" then map(select(.type != "deletion")) else . end
    ' "$EXPECTED_POLICY"
    ;;
  repos/loadingalias/rscrypto/commits/main)
    jq -n --arg sha "$EXPECTED_SHA" '{sha: $sha}'
    ;;
  *)
    echo "unexpected gh api endpoint: $endpoint" >&2
    exit 2
    ;;
esac
EOF
chmod +x "$TMP_ROOT/bin/gh"

export PATH="$TMP_ROOT/bin:$PATH"
export EXPECTED_POLICY="$REPO_ROOT/.github/rulesets/protect-main.json"
export EXPECTED_SHA
EXPECTED_SHA=$(git -C "$REPO_ROOT" rev-parse HEAD)

output="$TMP_ROOT/repository-controls.json"
github_output="$TMP_ROOT/github-output"
GITHUB_OUTPUT="$github_output" "$CHECKER" \
  --root "$REPO_ROOT" \
  --repo loadingalias/rscrypto \
  --commit "$EXPECTED_SHA" \
  --output "$output" >/dev/null

jq -e --arg commit "$EXPECTED_SHA" '
  .schema_version == 1
  and .kind == "rscrypto.repository-controls"
  and .release_commit == $commit
  and .repository.full_name == "loadingalias/rscrypto"
  and .repository.default_branch == "main"
  and .repository.default_branch_sha == $commit
  and .live.ruleset.current_user_can_bypass == "never"
  and ([.live.effective_rules[] | select(.type == "required_status_checks")] | length == 1)
' "$output" >/dev/null
grep -Fxq "evidence_name=$(basename "$output")" "$github_output"
grep -Fxq "evidence_path=$output" "$github_output"
grep -Eq '^evidence_sha256=[0-9a-f]{64}$' "$github_output"

for mode in bypass inactive missing-check wrong-app wrong-effective redacted; do
  if FAKE_GH_MODE=$mode "$CHECKER" \
    --root "$REPO_ROOT" \
    --repo loadingalias/rscrypto \
    --commit "$EXPECTED_SHA" \
    --output "$TMP_ROOT/$mode.json" >/dev/null 2>&1; then
    echo "repository controls check accepted invalid mode: $mode" >&2
    exit 1
  fi
done

redacted_output="$TMP_ROOT/redacted.json"
FAKE_GH_MODE=redacted "$CHECKER" \
  --root "$REPO_ROOT" \
  --repo loadingalias/rscrypto \
  --commit "$EXPECTED_SHA" \
  --output "$redacted_output" \
  --allow-redacted-bypass >/dev/null
jq -e '.validation.bypass_actors.status == "redacted_by_github_api"' "$redacted_output" >/dev/null

FAKE_GH_MODE=redacted-self "$CHECKER" \
  --root "$REPO_ROOT" \
  --repo loadingalias/rscrypto \
  --commit "$EXPECTED_SHA" \
  --output "$TMP_ROOT/redacted-self.json" \
  --allow-redacted-bypass >/dev/null

if FAKE_GH_MODE=bypass "$CHECKER" \
  --root "$REPO_ROOT" \
  --repo loadingalias/rscrypto \
  --commit "$EXPECTED_SHA" \
  --output "$TMP_ROOT/bypass-allowed.json" \
  --allow-redacted-bypass >/dev/null 2>&1; then
  echo "repository controls check treated a visible bypass as redacted" >&2
  exit 1
fi

echo "Repository controls evidence regression tests passed"
