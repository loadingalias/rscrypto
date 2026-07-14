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

scripts/ct/python.sh - "$REPO_ROOT/ct.toml" <<'PY'
import pathlib
import sys
import tomllib

manifest = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
cases = {case["name"]: case for case in manifest["dudect_case"]}
for name in (
  "ecdsa_p384_sign_fixed_vs_random_secret",
  "ecdsa_p384_keypair_sign_fixed_vs_random_secret",
):
  case = cases[name]
  assert case["samples"] == 20_000, f"{name} must retain the full evidence sample count"
  assert case.get("gate", "required") == "required", f"{name} must remain release-blocking"
  assert case["timeout_seconds"] == 7_200, f"{name} must fit physical RISC-V runtime"
PY

echo "Manual matrix CT regression tests passed"
