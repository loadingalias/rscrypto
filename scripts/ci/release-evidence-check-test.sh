#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/release-evidence-check.sh"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

mkdir -p "$TMP_ROOT/bin"
cat >"$TMP_ROOT/bin/gh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ "$1 $2" == "run list" ]]; then
  if [[ ${FAKE_GH_MODE:-success} == missing-run ]]; then
    echo '[]'
  else
    cat <<JSON
[{"databaseId":4242,"headSha":"${EXPECTED_SHA}","status":"completed","conclusion":"success","url":"https://example.invalid/runs/4242","createdAt":"2026-07-14T00:00:00Z"}]
JSON
  fi
  exit 0
fi

if [[ "$1 $2" == "run view" ]]; then
  rsa_conclusion=success
  if [[ ${FAKE_GH_MODE:-success} == failed-rsa ]]; then
    rsa_conclusion=failure
  fi
  cat <<JSON
{"jobs":[
  {"name":"Constant-Time Evidence (weekly) / CT Full (RISE RISC-V riscv64) / run","conclusion":"success"},
  {"name":"Constant-Time Evidence (weekly) / Complete (CT)","conclusion":"success"},
  {"name":"RSA Evidence (weekly) / Complete (RSA)","conclusion":"${rsa_conclusion}"},
  {"name":"Complete (weekly)","conclusion":"success"}
]}
JSON
  exit 0
fi

echo "unexpected gh invocation: $*" >&2
exit 2
EOF
chmod +x "$TMP_ROOT/bin/gh"

sha=0123456789abcdef0123456789abcdef01234567
export EXPECTED_SHA=$sha
export PATH="$TMP_ROOT/bin:$PATH"

output="$TMP_ROOT/github-output"
GITHUB_OUTPUT="$output" "$CHECKER" --commit "$sha" --repo loadingalias/rscrypto >/dev/null
grep -Fxq 'weekly_run_id=4242' "$output"
grep -Fxq 'weekly_run_url=https://example.invalid/runs/4242' "$output"

if FAKE_GH_MODE=missing-run "$CHECKER" --commit "$sha" --repo loadingalias/rscrypto >/dev/null 2>&1; then
  echo "release evidence check accepted a missing exact-commit Weekly run" >&2
  exit 1
fi

if FAKE_GH_MODE=failed-rsa "$CHECKER" --commit "$sha" --repo loadingalias/rscrypto >/dev/null 2>&1; then
  echo "release evidence check accepted failed RSA evidence" >&2
  exit 1
fi

echo "Release evidence regression tests passed"
