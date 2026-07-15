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

export PATH="$TMP_ROOT/bin:$PATH"

fixture="$TMP_ROOT/repo"
mkdir -p "$fixture/src" "$fixture/scripts/ci" \
  "$fixture/tools/ct-binsec-harness" "$fixture/tools/ct-dudect" "$fixture/tools/ct-harness"
cat >"$fixture/Cargo.toml" <<'EOF'
[package]
name = "rscrypto"
version = "0.7.3"
edition = "2024"
EOF
for lock in \
  Cargo.lock \
  tools/ct-binsec-harness/Cargo.lock \
  tools/ct-dudect/Cargo.lock \
  tools/ct-harness/Cargo.lock; do
  cat >"$fixture/$lock" <<'EOF'
version = 4

[[package]]
name = "rscrypto"
version = "0.7.3"
EOF
done
printf 'pub fn marker() {}\n' >"$fixture/src/lib.rs"
printf 'release helper v1\n' >"$fixture/scripts/ci/release-helper.sh"
printf '# Changelog\n' >"$fixture/CHANGELOG.md"

git -C "$fixture" init -q
git -C "$fixture" config user.email test@example.invalid
git -C "$fixture" config user.name "Release Evidence Test"
git -C "$fixture" add .
git -C "$fixture" commit -qm "evidence"
evidence_sha=$(git -C "$fixture" rev-parse HEAD)

export EXPECTED_SHA=$evidence_sha
output="$TMP_ROOT/github-output"
GITHUB_OUTPUT="$output" "$CHECKER" --root "$fixture" --commit "$evidence_sha" --repo loadingalias/rscrypto >/dev/null
grep -Fxq 'weekly_run_id=4242' "$output"
grep -Fxq 'weekly_run_url=https://example.invalid/runs/4242' "$output"
grep -Fxq "weekly_commit=$evidence_sha" "$output"
grep -Fxq 'weekly_version=0.7.3' "$output"
grep -Fxq 'weekly_evidence_mode=exact_commit' "$output"

if FAKE_GH_MODE=missing-run "$CHECKER" --root "$fixture" --commit "$evidence_sha" --repo loadingalias/rscrypto >/dev/null 2>&1; then
  echo "release evidence check accepted a missing exact-commit Weekly run" >&2
  exit 1
fi

if FAKE_GH_MODE=failed-rsa "$CHECKER" --root "$fixture" --commit "$evidence_sha" --repo loadingalias/rscrypto >/dev/null 2>&1; then
  echo "release evidence check accepted failed RSA evidence" >&2
  exit 1
fi

sed -i.bak 's/0\.7\.3/0.7.4/g' "$fixture/Cargo.toml" "$fixture"/Cargo.lock "$fixture"/tools/*/Cargo.lock
rm -f "$fixture/Cargo.toml.bak" "$fixture/Cargo.lock.bak" "$fixture"/tools/*/Cargo.lock.bak
printf 'release helper v2\n' >"$fixture/scripts/ci/release-helper.sh"
printf '\n## 0.7.4\n' >>"$fixture/CHANGELOG.md"
git -C "$fixture" add .
git -C "$fixture" commit -qm "release-only delta"
release_sha=$(git -C "$fixture" rev-parse HEAD)

: >"$output"
GITHUB_OUTPUT="$output" "$CHECKER" --root "$fixture" --commit "$release_sha" --repo loadingalias/rscrypto >/dev/null
grep -Fxq "weekly_commit=$evidence_sha" "$output"
grep -Fxq 'weekly_version=0.7.3' "$output"
grep -Fxq 'weekly_evidence_mode=release_only_delta' "$output"

printf 'pub fn marker() { unreachable!() }\n' >"$fixture/src/lib.rs"
git -C "$fixture" add src/lib.rs
git -C "$fixture" commit -qm "runtime delta"
runtime_sha=$(git -C "$fixture" rev-parse HEAD)
if "$CHECKER" --root "$fixture" --commit "$runtime_sha" --repo loadingalias/rscrypto >/dev/null 2>&1; then
  echo "release evidence check promoted evidence across a runtime source change" >&2
  exit 1
fi

echo "Release evidence regression tests passed"
