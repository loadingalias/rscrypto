#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEALER="$ROOT/scripts/ci/seal-remote-evidence.sh"
FIXTURE="$(mktemp -d "${TMPDIR:-/tmp}/rscrypto-seal-evidence-test.XXXXXX")"
trap 'rm -rf "$FIXTURE"' EXIT

git -C "$FIXTURE" init --quiet
mkdir -p "$FIXTURE/target/ct/example" "$FIXTURE/a"
printf '{"status":"pass"}\n' > "$FIXTURE/target/ct/example/report.json"
printf 'left\n' > "$FIXTURE/a/b"
printf 'right\n' > "$FIXTURE/a__b"

RSCRYPTO_REPO_ROOT="$FIXTURE" \
DEV_MACHINE_TARGET="test-target" \
DEV_MACHINE_INSTANCE_TYPE="test-shape" \
  "$SEALER" ct valid-run target/ct/example >/dev/null

RUN_DIR="$FIXTURE/benchmark_results/ct/valid-run"
[[ -f "$RUN_DIR/artifacts/target__ct__example/report.json" ]] || {
  echo "sealed evidence omitted the requested artifact" >&2
  exit 1
}
grep -Fqx 'target=test-target' "$RUN_DIR/remote-run.txt" || {
  echo "sealed evidence omitted the remote target" >&2
  exit 1
}
grep -Fqx 'instance_type=test-shape' "$RUN_DIR/remote-run.txt" || {
  echo "sealed evidence omitted the instance shape" >&2
  exit 1
}

ARCHIVE="$FIXTURE/benchmark_results/.transfers/valid-run.tar"
MANIFEST="$ARCHIVE.sha256"
read -r EXPECTED_DIGEST EXPECTED_NAME < "$MANIFEST"
[[ "$EXPECTED_NAME" == "valid-run.tar" ]] || {
  echo "sealed evidence manifest named the wrong archive" >&2
  exit 1
}
if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL_DIGEST="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
else
  ACTUAL_DIGEST="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
fi
[[ "$ACTUAL_DIGEST" == "$EXPECTED_DIGEST" ]] || {
  echo "sealed evidence archive digest does not match its manifest" >&2
  exit 1
}
if tar -tf "$ARCHIVE" | grep -Ev '^ct/valid-run(/|$)' | grep -q .; then
  echo "sealed evidence archive escaped its exact run root" >&2
  exit 1
fi

if RSCRYPTO_REPO_ROOT="$FIXTURE" "$SEALER" ct traversal ../escape >/dev/null 2>&1; then
  echo "sealed evidence accepted a traversal path" >&2
  exit 1
fi
if RSCRYPTO_REPO_ROOT="$FIXTURE" "$SEALER" ct collision a/b a__b >/dev/null 2>&1; then
  echo "sealed evidence accepted colliding artifact labels" >&2
  exit 1
fi

echo "Remote evidence sealing tests passed"
