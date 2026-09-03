#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_ID="bench-wrapper-contract-$$"
ARTIFACT_DIR="$REPO_ROOT/benchmark_results/criterion/$RUN_ID"
TRANSFER_DIR="$REPO_ROOT/benchmark_results/.transfers"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT" "$ARTIFACT_DIR" "$TRANSFER_DIR/$RUN_ID.tar" "$TRANSFER_DIR/$RUN_ID.tar.sha256"' EXIT

fail() {
  echo "benchmark wrapper test failure: $*" >&2
  exit 1
}

DEV_MACHINE_TARGET=aws-linux-arm64-bench \
DEV_MACHINE_INSTANCE_TYPE=c8g.xlarge \
RSCRYPTO_BENCH_RUN_ID="$RUN_ID" \
BENCH_DRY_RUN_PLAN=true \
  "$SCRIPT_DIR/bench.sh" p256-ecdh output_dir="$TMP_ROOT/output" > "$TMP_ROOT/bench.log"

[[ -f "$ARTIFACT_DIR/results.txt" ]] || fail "remote run omitted structured results"
[[ -f "$ARTIFACT_DIR/source-files.sha256" ]] || fail "remote run omitted source provenance"
[[ -f "$TRANSFER_DIR/$RUN_ID.tar" ]] || fail "remote run omitted the collection archive"
[[ -f "$TRANSFER_DIR/$RUN_ID.tar.sha256" ]] || fail "remote run omitted the collection manifest"
grep -Fqx "run_id=$RUN_ID" "$ARTIFACT_DIR/remote-run.txt" || fail "run ID metadata is missing"
grep -Fqx 'target=aws-linux-arm64-bench' "$ARTIFACT_DIR/remote-run.txt" || fail "target metadata is missing"
grep -Fqx 'instance_type=c8g.xlarge' "$ARTIFACT_DIR/remote-run.txt" || fail "instance metadata is missing"
grep -Fqx 'mode=remote' "$ARTIFACT_DIR/remote-run.txt" || fail "development-machine run was not remote"
grep -Eq '^source_identity=sha256:[0-9a-f]{64}$' "$ARTIFACT_DIR/remote-run.txt" \
  || fail "source identity is missing or malformed"
grep -Fq "Remote run ID: $RUN_ID" "$TMP_ROOT/bench.log" || fail "collection ID was not reported"

expected_digest=$(awk '{print $1}' "$TRANSFER_DIR/$RUN_ID.tar.sha256")
actual_digest=$(sha256sum "$TRANSFER_DIR/$RUN_ID.tar" | awk '{print $1}')
[[ "$actual_digest" == "$expected_digest" ]] || fail "collection archive digest is invalid"
[[ $(tar -tf "$TRANSFER_DIR/$RUN_ID.tar" | sed -n '1p') == "criterion/$RUN_ID/" ]] \
  || fail "collection archive is not rooted at the exact Criterion run"

echo "Benchmark wrapper tests passed"
