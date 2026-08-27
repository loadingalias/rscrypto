#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

fixture="$TMP_ROOT/repository"
fake_bin="$TMP_ROOT/bin"
mock_log="$TMP_ROOT/commands.log"
output="$TMP_ROOT/pre-push.out"
mkdir -p "$fixture/scripts/ci" "$fixture/scripts/lib" "$fixture/src" "$fake_bin" "$TMP_ROOT/home/.cargo/bin"
cp "$REPO_ROOT/scripts/ci/pre-push.sh" "$fixture/scripts/ci/pre-push.sh"
cp "$REPO_ROOT/scripts/lib/common.sh" "$REPO_ROOT/scripts/lib/rail-plan.sh" "$fixture/scripts/lib/"

printf '%s\n' '#!/usr/bin/env bash' 'exit 0' >"$fixture/scripts/example.sh"
printf '%s\n' '#![no_std]' >"$fixture/src/lib.rs"

git -C "$fixture" init --quiet --initial-branch=main
git -C "$fixture" config user.email "ci@example.invalid"
git -C "$fixture" config user.name "CI"
git -C "$fixture" add .
git -C "$fixture" commit --quiet -m "baseline"
base_commit="$(git -C "$fixture" rev-parse HEAD)"
git -C "$fixture" update-ref refs/remotes/origin/main "$base_commit"

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'printf '\''cargo %s\n'\'' "$*" >>"$MOCK_LOG"' \
  'case "$*" in' \
  '  "rail change check --merge-base --required") exit "${MOCK_CHANGE_STATUS:-0}" ;;' \
  '  "rail unify --check") exit "${MOCK_UNIFY_STATUS:-0}" ;;' \
  '  "rail config validate --strict"|"rail config migrate --check") exit "${MOCK_CONFIG_STATUS:-0}" ;;' \
  '  *) echo "unexpected cargo command: $*" >&2; exit 97 ;;' \
  'esac' \
  >"$fake_bin/cargo"
chmod +x "$fake_bin/cargo"

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'printf '\''just %s\n'\'' "$*" >>"$MOCK_LOG"' \
  '[[ "$*" == "--list" ]]' \
  >"$fake_bin/just"
chmod +x "$fake_bin/just"

run_pre_push() {
  : >"$mock_log"
  (
    cd "$fixture"
    HOME="$TMP_ROOT/home" \
      PATH="$fake_bin:$PATH" \
      MOCK_LOG="$mock_log" \
      MOCK_CHANGE_STATUS="${MOCK_CHANGE_STATUS:-0}" \
      MOCK_UNIFY_STATUS="${MOCK_UNIFY_STATUS:-0}" \
      MOCK_CONFIG_STATUS="${MOCK_CONFIG_STATUS:-0}" \
      scripts/ci/pre-push.sh
  ) >"$output" 2>&1
}

git -C "$fixture" switch --quiet -c topic
printf '%s\n' '#![no_std]' '// topic change' >"$fixture/src/lib.rs"
git -C "$fixture" add src/lib.rs
git -C "$fixture" commit --quiet -m "change source"

if ! run_pre_push; then
  cat "$output" >&2
  exit 1
fi
grep -Fq 'Fast pre-push checks passed.' "$output"
grep -Fq 'cargo rail change check --merge-base --required' "$mock_log"
if grep -Fq 'cargo rail unify' "$mock_log"; then
  echo "ordinary source changes must not run Cargo graph unification" >&2
  exit 1
fi
if grep -Fq 'just ' "$mock_log"; then
  echo "ordinary source changes must not run Just validation or host checks" >&2
  exit 1
fi

if (
  cd "$fixture"
  HOME="$TMP_ROOT/home" PATH="$fake_bin:$PATH" MOCK_LOG="$mock_log" scripts/ci/pre-push.sh --light
) >"$output" 2>&1; then
  echo "pre-push profiles must not be accepted" >&2
  exit 1
fi
grep -Fq 'does not accept arguments' "$output"

git -C "$fixture" switch --quiet main
if run_pre_push; then
  echo "pre-push must reject main" >&2
  exit 1
fi
grep -Fq 'refusing to push directly from main' "$output"

git -C "$fixture" switch --quiet --detach topic
if run_pre_push; then
  echo "pre-push must reject detached HEAD" >&2
  exit 1
fi
grep -Fq 'refusing to push from a detached HEAD' "$output"
git -C "$fixture" switch --quiet topic

git -C "$fixture" update-ref -d refs/remotes/origin/main
if run_pre_push; then
  echo "pre-push must reject an unavailable immutable base" >&2
  exit 1
fi
grep -Fq 'refs/remotes/origin/main is unavailable' "$output"
git -C "$fixture" update-ref refs/remotes/origin/main "$base_commit"

printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' 'exit 0' >"$fixture/scripts/check.sh"
git -C "$fixture" add scripts/check.sh
git -C "$fixture" commit --quiet -m "add shell check"
run_pre_push
grep -Fq 'Shell syntax' "$output"

printf '%s\n' 'default:' '    @true' >"$fixture/justfile"
git -C "$fixture" add justfile
git -C "$fixture" commit --quiet -m "add justfile"
run_pre_push
grep -Fq 'just --list' "$mock_log"

mkdir -p "$fixture/tools/example"
printf '%s\n' '[package]' 'name = "example"' 'version = "0.1.0"' >"$fixture/tools/example/Cargo.toml"
git -C "$fixture" add tools/example/Cargo.toml
git -C "$fixture" commit --quiet -m "add cargo manifest"
run_pre_push
grep -Fq 'cargo rail unify --check' "$mock_log"

mkdir -p "$fixture/.config"
printf '%s\n' '[release]' 'source = "changes"' >"$fixture/.config/rail.toml"
git -C "$fixture" add .config/rail.toml
git -C "$fixture" commit --quiet -m "add rail config"
run_pre_push
grep -Fq 'cargo rail config validate --strict' "$mock_log"
grep -Fq 'cargo rail config migrate --check' "$mock_log"

MOCK_CHANGE_STATUS=42
if run_pre_push; then
  echo "pre-push must fail when release intent coverage fails" >&2
  exit 1
fi
unset MOCK_CHANGE_STATUS

git -C "$fixture" switch --quiet main
git -C "$fixture" switch --quiet -c rail/release-test
printf '%s\n' 'release preparation' >"$fixture/release.txt"
git -C "$fixture" add release.txt
git -C "$fixture" commit --quiet \
  -m "prepare release" \
  -m "Rail-Release-Mode: prepare"
MOCK_CHANGE_STATUS=42 run_pre_push
if grep -Fq 'cargo rail change check --merge-base --required' "$mock_log"; then
  echo "Cargo Rail release branches must not repeat consumed release intent" >&2
  exit 1
fi
grep -Fq 'Cargo Rail release branch has consumed its reviewed intent' "$output"

git -C "$fixture" switch --quiet main
git -C "$fixture" switch --quiet -c whitespace-test
printf 'trailing whitespace \n' >"$fixture/bad.txt"
git -C "$fixture" add bad.txt
git -C "$fixture" commit --quiet -m "add bad whitespace"
if run_pre_push; then
  echo "pre-push must reject outgoing diff whitespace errors" >&2
  exit 1
fi
grep -Fq 'trailing whitespace' "$output"

if rg -q 'just (check|test)|check-actions' "$mock_log"; then
  echo "pre-push must never run builds, tests, or the action suite" >&2
  exit 1
fi

echo "Pre-push regression tests passed"
