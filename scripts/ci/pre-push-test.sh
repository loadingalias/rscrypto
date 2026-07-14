#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

fixture="$TMP_ROOT/repository"
fake_bin="$TMP_ROOT/bin"
mock_log="$TMP_ROOT/commands.log"
mkdir -p "$fixture/scripts/ci" "$fixture/scripts/lib" "$fake_bin" "$TMP_ROOT/home/.cargo/bin"
cp "$REPO_ROOT/scripts/ci/pre-push.sh" "$fixture/scripts/ci/pre-push.sh"
cp "$REPO_ROOT/scripts/lib/common.sh" "$REPO_ROOT/scripts/lib/rail-plan.sh" "$fixture/scripts/lib/"

git -C "$fixture" init --quiet

cat >"$fake_bin/cargo" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'cargo %s\n' "$*" >>"$MOCK_LOG"
if [[ "$*" == "rail change check --merge-base --required" ]]; then
  exit 42
fi
SH
chmod +x "$fake_bin/cargo"

cat >"$fake_bin/just" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'just %s\n' "$*" >>"$MOCK_LOG"
SH
chmod +x "$fake_bin/just"

plan='{"result":"success","files":[{"path":"Cargo.toml"}],"scope":{"mode":"workspace","surfaces":{"build":true,"test":true}}}'

normal_output="$TMP_ROOT/normal.out"
if (
  cd "$fixture"
  HOME="$TMP_ROOT/home" \
    PATH="$fake_bin:$PATH" \
    MOCK_LOG="$mock_log" \
    RAIL_PLAN_JSON_CACHE="$plan" \
    scripts/ci/pre-push.sh --light
) >"$normal_output" 2>&1; then
  echo "ordinary pushes must fail when release intent coverage fails" >&2
  exit 1
fi
grep -Fq "cargo rail change check --merge-base --required" "$mock_log"

: >"$mock_log"
release_output="$TMP_ROOT/release.out"
(
  cd "$fixture"
  HOME="$TMP_ROOT/home" \
    PATH="$fake_bin:$PATH" \
    MOCK_LOG="$mock_log" \
    RAIL_PLAN_JSON_CACHE="$plan" \
    RSCRYPTO_RELEASE_PUSH=1 \
    scripts/ci/pre-push.sh --light
) >"$release_output" 2>&1

if grep -Fq "cargo rail change check --merge-base --required" "$mock_log"; then
  echo "cargo-rail release pushes must not recheck consumed change files" >&2
  exit 1
fi
grep -Fq "cargo rail unify --check --explain" "$mock_log"
grep -Fq "just check" "$mock_log"
grep -Fq "cargo-rail validated and consumed the release change files" "$release_output"

echo "Pre-push release regression tests passed"
