#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --platform NAME --target TRIPLE [--dudect-timeout N] [--binsec-timeout N] [--dudect-filter CSV] [--dudect-gate required|diagnostic|all] [--raw]" >&2
  exit 2
}

platform=""
target=""
dudect_timeout=1800
binsec_timeout=900
dudect_filter=""
dudect_gate=required
raw=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) platform=${2:-}; shift 2 ;;
    --target) target=${2:-}; shift 2 ;;
    --dudect-timeout) dudect_timeout=${2:-}; shift 2 ;;
    --binsec-timeout) binsec_timeout=${2:-}; shift 2 ;;
    --dudect-filter) dudect_filter=${2:-}; shift 2 ;;
    --dudect-gate) dudect_gate=${2:-}; shift 2 ;;
    --raw) raw=true; shift ;;
    *) usage ;;
  esac
done

case "$platform:$target" in
  amd-zen4:x86_64-unknown-linux-gnu | intel-spr:x86_64-unknown-linux-gnu | \
    intel-icl:x86_64-unknown-linux-gnu | amd-zen5:x86_64-unknown-linux-gnu | \
    graviton3:aarch64-unknown-linux-gnu | graviton4:aarch64-unknown-linux-gnu | \
    ibm-s390x:s390x-unknown-linux-gnu | \
    ibm-power10:powerpc64le-unknown-linux-gnu | \
  rise-riscv:riscv64gc-unknown-linux-gnu) ;;
  *) usage ;;
esac
[[ "$dudect_timeout" =~ ^[1-9][0-9]*$ && "$binsec_timeout" =~ ^[1-9][0-9]*$ ]] || usage
case "$dudect_gate" in required | diagnostic | all) ;; *) usage ;; esac

evidence_dir=target/ct-evidence-package
mkdir -p "$evidence_dir"
{
  echo "CT platform: $platform"
  echo "CT target: $target"
  uname -a
  rustc -vV
  cargo -V
  command -v lscpu >/dev/null 2>&1 && lscpu || true
} 2>&1 | tee "$evidence_dir/host-$platform.log"

args=(
  --target "$target"
  --dudect-timeout "$dudect_timeout"
  --binsec-timeout "$binsec_timeout"
  --dudect-gate "$dudect_gate"
)
[[ -z "$dudect_filter" ]] || args+=(--dudect-filter "$dudect_filter")

status=0
scripts/lib/python.sh scripts/ct/full.py "${args[@]}" \
  2>&1 | tee "$evidence_dir/ct-full-$platform.log" || status=$?

package_args=(--target "$target" --suffix "$platform" --out-dir "$evidence_dir")
[[ "$raw" == false ]] || package_args+=(--raw)
scripts/lib/python.sh scripts/ct/package_evidence.py "${package_args[@]}"
exit "$status"
