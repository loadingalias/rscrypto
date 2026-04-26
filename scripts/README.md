# Scripts Map

Caller map for `scripts/`. Every `.sh` under `scripts/` must appear here with
its caller.

## Entry Points (called from `justfile` or CI)

| Script | Callers |
|--------|---------|
| `check/check.sh`               | `just check` |
| `check/check-all.sh`           | `just check-all` |
| `ci/ci-check.sh`               | `just ci-check`, `_ci-suite.yaml` (IBM/RISC-V pre-run) |
| `test/test.sh`                 | `just test`, `just test-all`, `_ci-suite.yaml` |
| `test/test-feature-matrix.sh`  | `just test-feature-matrix`, `scripts/check/check.sh`, `scripts/ci/ci-check.sh`, `weekly.yaml` |
| `test/test-miri.sh`            | `just test-miri`, `weekly.yaml` |
| `test/test-fuzz.sh`            | `just test-fuzz`, `weekly.yaml` |
| `test/test-coverage.sh`        | `just test-coverage`, `just test-fuzz-coverage`, `just test-all-coverage`, `weekly.yaml` |
| `bench/bench.sh`               | `just bench`, `just bench-quick` |
| `ci/pin-actions.sh`            | `just pin-actions`, `just check-actions`, `scripts/update/update-all.sh` |
| `ci/pre-push.sh`               | `just ci-pre-push`, optional `.git/hooks/pre-push` |
| `update/update-all.sh`         | `just update`, `just update-check` |

## Cross-platform Check Helpers

| Script | Callers |
|--------|---------|
| `check/check-win.sh`    | `scripts/check/check-all.sh` |
| `check/check-linux.sh`  | `scripts/check/check-all.sh` |
| `check/check-ibm.sh`    | `scripts/check/check-all.sh` |
| `check/zig-cc.sh`       | `scripts/check/check-linux.sh`, `scripts/check/check-ibm.sh` |

## Bench Internals

| Script | Callers |
|--------|---------|
| `ci/run-bench.sh`            | `scripts/bench/bench.sh`, `bench.yaml` |
| `bench/blake3-gap-gate.sh`   | `scripts/ci/run-bench.sh` |

## CI-only (not surfaced via `just`)

| Script | Callers |
|--------|---------|
| `ci/install-tools.sh`          | `.github/actions/setup/action.yaml` |
| `ci/emit-manual-matrix.sh`     | `bench.yaml` |
| `ci/nostd-wasm-suite.sh`       | `_ci-suite.yaml` (no_std, wasm matrices) |

## Shared Libraries (sourced, not invoked)

| Script | Sourced by |
|--------|------------|
| `lib/common.sh`         | `scripts/check/*.sh`, `scripts/test/*.sh`, `scripts/ci/ci-check.sh` |
| `lib/rail-plan.sh`      | `scripts/lib/common.sh` |
| `lib/fuzz-packages.sh`  | `scripts/test/test-fuzz.sh`, `scripts/test/test-coverage.sh` |
| `lib/targets.sh`        | `scripts/check/check-all.sh`, `scripts/check/check-linux.sh`, `scripts/check/check-ibm.sh` |
| `lib/target-matrix.sh`  | `scripts/lib/targets.sh`, `_ci-suite.yaml` (target-matrix job) |
| `lib/toolchain.sh`      | `.github/actions/setup-toolchain/action.yaml` |

## Results layout

Bench results — both local (`just bench*`) and CI (`/extract-bench` skill pulls
GitHub Actions artifacts) — land under:

```
benchmark_results/<YYYY-MM-DD>/<os>/<arch>/results.txt
```

Local runs use the host calendar date and `linux|macos|windows` +
`x86-64|aarch64`. Same layout in CI; the extractor writes into the same tree
so local and CI runs interleave by date without collision.
