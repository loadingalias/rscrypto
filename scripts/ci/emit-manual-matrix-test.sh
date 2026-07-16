#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

matrix="$({
  cd "$REPO_ROOT"
  GH_RUN_ID=1 CT_PLATFORMS=rise-riscv scripts/ci/emit-manual-matrix.sh ct
})"

jq -e '
  length == 1
  and .[0].platform == "rise-riscv"
  and .[0].target == "riscv64gc-unknown-linux-gnu"
  and .[0].tools_mode == "none"
  and .[0].enable_rust_cache == true
' <<<"$matrix" >/dev/null

ct_default="$({
  cd "$REPO_ROOT"
  GH_RUN_ID=1 CT_PLATFORMS=all scripts/ci/emit-manual-matrix.sh ct
})"
jq -e 'length == 8 and all(.platform != "rise-riscv")' <<<"$ct_default" >/dev/null

bench_default="$({
  cd "$REPO_ROOT"
  GH_RUN_ID=1 BENCH_PLATFORMS=all scripts/ci/emit-manual-matrix.sh bench
})"
jq -e 'length == 8 and all(.platform != "rise-riscv")' <<<"$bench_default" >/dev/null

if GH_RUN_ID=1 BENCH_PLATFORMS=riscv scripts/ci/emit-manual-matrix.sh bench >/dev/null 2>&1; then
  echo "generic benchmark matrix accepted the RISC-V lane" >&2
  exit 1
fi

scripts/ct/python.sh - "$REPO_ROOT/ct.toml" <<'PY'
import pathlib
import sys
import tomllib

manifest = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
cases = {case["name"]: case for case in manifest["dudect_case"]}
p384_sign = cases["ecdsa_p384_sign_fixed_vs_random_secret"]
assert p384_sign["samples"] == 20_000, "P-384 signing must retain the full evidence sample count"
assert p384_sign.get("gate", "required") == "required", "P-384 signing must remain release-blocking"
assert p384_sign["timeout_seconds"] == 7_200, "P-384 signing must fit physical RISC-V runtime"

for name in (
  "ecdsa_p256_keypair_sign_fixed_vs_random_secret",
  "ecdsa_p384_keypair_sign_fixed_vs_random_secret",
):
  assert name not in cases, f"{name} redundantly precomputes public keys without measuring them"
PY

github_output="$(mktemp)"
trap 'rm -f "$github_output"' EXIT
GITHUB_OUTPUT="$github_output" scripts/ct/python.sh - "$REPO_ROOT" <<'PY'
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / "scripts" / "ct"))

from validate_release_evidence import expected_lanes

lanes = expected_lanes(root, root / "scripts" / "ci" / "emit-manual-matrix.sh")
assert len(lanes) == 9, "release validation must resolve every CT lane under GitHub Actions"
assert pathlib.Path(os.environ["GITHUB_OUTPUT"]).stat().st_size == 0, "matrix lookup must not write step outputs"
PY

echo "Manual matrix CT regression tests passed"
