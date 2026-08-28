#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/release-ct-recovery-check.sh"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

mkdir -p "$TMP_ROOT/bin"
cat >"$TMP_ROOT/bin/gh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

mode=${FAKE_GH_MODE:-success}
if [[ "$1" == "api" && "$2" == "repos/loadingalias/rscrypto/actions/runs/4242" ]]; then
  branch=main
  conclusion=success
  event=workflow_dispatch
  path=.github/workflows/ct.yaml
  sha=$EXPECTED_WORKFLOW_SHA
  status=completed
  repo=loadingalias/rscrypto
  [[ "$mode" == wrong-branch ]] && branch=recovery
  [[ "$mode" == failed-run ]] && conclusion=failure
  [[ "$mode" == wrong-event ]] && event=push
  [[ "$mode" == incomplete-run ]] && status=in_progress
  [[ "$mode" == wrong-workflow ]] && path=.github/workflows/weekly.yaml
  [[ "$mode" == wrong-sha ]] && sha=0000000000000000000000000000000000000000
  [[ "$mode" == fork ]] && repo=attacker/rscrypto
  cat <<JSON
{"event":"$event","status":"$status","conclusion":"$conclusion","head_branch":"$branch","head_sha":"$sha","path":"$path","html_url":"https://example.invalid/runs/4242","head_repository":{"full_name":"$repo"}}
JSON
  exit 0
fi

if [[ "$1 $2 ${3:-}" == "run view 4242" ]]; then
  lane=success
  complete=success
  [[ "$mode" == failed-lane ]] && lane=failure
  [[ "$mode" == failed-complete ]] && complete=failure
  if [[ ${FAKE_GH_GROUP:-s390x} == x86_64 ]]; then
    cat <<JSON
{"jobs":[
  {"name":"Resolve CT Matrix","conclusion":"success"},
  {"name":"CT Full (AMD Zen4) / run","conclusion":"$lane"},
  {"name":"CT Full (AMD Zen5) / run","conclusion":"success"},
  {"name":"CT Full (Intel Ice Lake) / run","conclusion":"success"},
  {"name":"CT Full (Intel Sapphire Rapids) / run","conclusion":"success"},
  {"name":"Complete (CT)","conclusion":"$complete"}
]}
JSON
  else
    cat <<JSON
{"jobs":[
  {"name":"Resolve CT Matrix","conclusion":"success"},
  {"name":"CT Full (IBM Z s390x) / run","conclusion":"$lane"},
  {"name":"Complete (CT)","conclusion":"$complete"}
]}
JSON
  fi
  exit 0
fi

if [[ "$1" == "api" && "$2" == "repos/loadingalias/rscrypto/actions/runs/4242/artifacts?per_page=100" ]]; then
  if [[ ${FAKE_GH_GROUP:-s390x} == x86_64 ]]; then
    cat <<'JSON'
{"total_count":4,"artifacts":[
  {"name":"ct-raw-amd-zen4","expired":false,"size_in_bytes":4096},
  {"name":"ct-raw-amd-zen5","expired":false,"size_in_bytes":4096},
  {"name":"ct-raw-intel-icl","expired":false,"size_in_bytes":4096},
  {"name":"ct-raw-intel-spr","expired":false,"size_in_bytes":4096}
]}
JSON
    exit 0
  fi
  case "$mode" in
    missing-artifact) echo '{"total_count":0,"artifacts":[]}' ;;
    expired-artifact) echo '{"total_count":1,"artifacts":[{"name":"ct-raw-ibm-s390x","expired":true,"size_in_bytes":4096}]}' ;;
    empty-artifact) echo '{"total_count":1,"artifacts":[{"name":"ct-raw-ibm-s390x","expired":false,"size_in_bytes":0}]}' ;;
    wrong-artifact) echo '{"total_count":1,"artifacts":[{"name":"ct-ibm-s390x","expired":false,"size_in_bytes":4096}]}' ;;
    extra-artifact) echo '{"total_count":2,"artifacts":[{"name":"ct-raw-ibm-s390x","expired":false,"size_in_bytes":4096},{"name":"other","expired":false,"size_in_bytes":1}]}' ;;
    *) echo '{"total_count":1,"artifacts":[{"name":"ct-raw-ibm-s390x","expired":false,"size_in_bytes":4096}]}' ;;
  esac
  exit 0
fi

echo "unexpected gh invocation: $*" >&2
exit 2
EOF
chmod +x "$TMP_ROOT/bin/gh"

export EXPECTED_WORKFLOW_SHA=1234567890abcdef1234567890abcdef12345678
export PATH="$TMP_ROOT/bin:$PATH"
unset BASH_ENV

output="$TMP_ROOT/github-output"
GITHUB_OUTPUT="$output" "$CHECKER" \
  --run-id 4242 \
  --platform-group s390x \
  --workflow-commit "$EXPECTED_WORKFLOW_SHA" \
  --repo loadingalias/rscrypto >/dev/null
grep -Fxq 'recovery_run_id=4242' "$output"
grep -Fxq 'recovery_run_url=https://example.invalid/runs/4242' "$output"

x86_output="$TMP_ROOT/github-output-x86_64"
FAKE_GH_GROUP=x86_64 GITHUB_OUTPUT="$x86_output" "$CHECKER" \
  --run-id 4242 \
  --platform-group x86_64 \
  --workflow-commit "$EXPECTED_WORKFLOW_SHA" \
  --repo loadingalias/rscrypto >/dev/null
grep -Fxq 'recovery_run_id=4242' "$x86_output"
grep -Fxq 'recovery_run_url=https://example.invalid/runs/4242' "$x86_output"

for mode in \
  wrong-branch failed-run wrong-event incomplete-run wrong-workflow wrong-sha fork \
  failed-lane failed-complete missing-artifact expired-artifact empty-artifact \
  wrong-artifact extra-artifact; do
  if FAKE_GH_MODE="$mode" "$CHECKER" \
    --run-id 4242 \
    --platform-group s390x \
    --workflow-commit "$EXPECTED_WORKFLOW_SHA" \
    --repo loadingalias/rscrypto >/dev/null 2>&1; then
    echo "CT recovery check accepted $mode" >&2
    exit 1
  fi
done

echo "Release CT recovery regression tests passed"
