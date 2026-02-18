# Scripts Map

This file is the caller map for `scripts/`.

## Entry Points

- `scripts/check/check.sh`
  - Called by: `just check`
- `scripts/check/check-all.sh`
  - Called by: `just check-all`
- `scripts/check/check-win.sh`
  - Called by: `just check-win`, `scripts/check/check-all.sh`
- `scripts/check/check-linux.sh`
  - Called by: `just check-linux`, `scripts/check/check-all.sh`
- `scripts/check/check-ibm.sh`
  - Called by: `scripts/check/check-all.sh`
- `scripts/test/test.sh`
  - Called by: `just test`
- `scripts/test/test-miri.sh`
  - Called by: `just test-miri`
- `scripts/test/test-fuzz.sh`
  - Called by: `just test-fuzz`
- `scripts/bench/bench.sh`
  - Called by: `just bench`
- `scripts/ci/ci-check.sh`
  - Called by: `just ci-check`, CI workflows
- `scripts/ci/check-infra.sh`
  - Called by: `scripts/ci/ci-check.sh`
- `scripts/ci/install-tools.sh`
  - Called by: `.github/actions/setup/action.yaml`, `.github/actions/setup-runson/action.yaml`, `.github/actions/setup-namespace/action.yaml`
- `scripts/ci/pin-actions.sh`
  - Called by: `just pin-actions`, `just verify-actions`
- `scripts/ci/run-tune.sh`
  - Called by: `.github/workflows/tune.yaml`, `scripts/tune/apply.sh`
- `scripts/tune/apply.sh`
  - Called by: `just tune-apply`
- `scripts/ci/pre-push.sh`
  - Called by: optional local Git hook (`.git/hooks/pre-push`)

## Shared Libraries

- `scripts/lib/common.sh`
  - Sourced by: `scripts/check/*.sh`, `scripts/test/*.sh`, `scripts/ci/ci-check.sh`
- `scripts/lib/targets.sh`
  - Sourced by: `scripts/check/check-all.sh`, `scripts/check/check-win.sh`, `scripts/check/check-linux.sh`
- `scripts/lib/target-matrix.py`
  - Called by: `scripts/lib/targets.sh`, `.github/workflows/commit.yaml`, `.github/workflows/weekly.yaml`, `just target-matrix-shell`, `just target-matrix-json`
- `scripts/lib/toolchain.sh`
  - Called by: `.github/actions/setup-toolchain/action.yaml`, `.github/actions/setup-runson/action.yaml`

## Utilities

- `scripts/check/zig-cc.sh`
  - Called by: `scripts/check/check-linux.sh`
- `scripts/bench/criterion-summary.py`
  - Called by: `just bench-summary`, `just bench-compare`, `just bench-blake3-compare`
- `scripts/bench/blake3-codegen-audit.sh`
  - Called by: `just blake3-codegen-audit`
- `scripts/bench/comp-check.py`
  - Called by: `just comp-check`
- `scripts/gen_blake3_x86_asm_ports.py`
  - Called by: `just gen-blake3-x86-asm-ports`
- `scripts/gen/kernel_tables.py`
  - Called by: `just gen-kernel-tables`; also documented in `crates/checksum/src/dispatch.rs` and `crates/checksum/src/generated/README.md`
- `scripts/gen_hashes_testdata.py`
  - Called by: `just gen-hashes-testdata` (manual developer utility)
