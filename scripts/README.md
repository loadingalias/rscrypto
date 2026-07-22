# Scripts Map

Caller map for `scripts/`. Every `.sh` under `scripts/` must appear here with
its caller.

## Entry Points (called from `justfile` or CI)

| Script | Callers |
|--------|---------|
| `check/check.sh`               | `just check` |
| `check/check-all.sh`           | `just check-all` |
| `check/check-feature-matrix.sh`| `just check-feature-matrix`, `scripts/check/check.sh`, `ci/run-rust-job.sh` |
| `check/asm-ledger.sh`          | `scripts/check/check.sh` |
| `check/zeroize-evidence.sh`    | `just check-zeroize-evidence`, `scripts/check/check-all.sh` |
| `ci/ci-check.sh`               | `just ci-check`, `ci/run-rust-job.sh` |
| `ci/native-check.sh`           | `ci/run-rust-job.sh` |
| `test/test.sh`                 | `just test`, `just test-all`, `ci/run-rust-job.sh` |
| `test/test-feature-matrix.sh`  | `just test-feature-matrix`, `scripts/check/check.sh`, `ci/run-rust-job.sh` |
| `test/test-miri.sh`            | `just test-miri`, `ci/run-rust-job.sh` |
| `test/test-fuzz.sh`            | `just test-fuzz`, `ci/run-rust-job.sh` |
| `test/test-fuzz-asan.sh`       | `just test-fuzz-asan`, `ci/run-rust-job.sh` |
| `test/test-rsa-leakage.sh`     | `just test-rsa-leakage`, `ci/run-rust-job.sh` |
| `test/test-coverage.sh`        | `just test-coverage`, `just test-fuzz-coverage`, `weekly.yaml` |
| `bench/bench.sh`               | `just bench`, `just bench-quick` |
| `ci/check-action-pins.sh`      | `just check-actions`, `ci/ci-check.sh`, `ci/dependabot-smoke.sh` |
| `ci/check-action-pins-test.sh` | `just check-actions`, `ci/dependabot-smoke.sh` |
| `ci/dependabot-smoke-test.sh`  | `just check-actions` |
| `ci/check-ci-ownership.sh`     | `just check-actions`, `ci/check-ci-ownership-test.sh` |
| `ci/check-ci-ownership-test.sh`| `just check-actions` |
| `ci/run-rust-job-test.sh`      | `just check-actions` |
| `ci/emit-manual-matrix-test.sh`| `just check-actions` |
| `ci/pre-push-test.sh`          | `just check-actions` |
| `ci/release-plan-check.sh`     | `just release-check` |
| `ci/release-evidence-check.sh` | `just release-tag`, `release.yaml`, `ci/release-evidence-check-test.sh` |
| `ci/release-evidence-check-test.sh` | `just check-actions` |
| `ci/repository-controls-evidence.sh` | `just check-repository-controls`, `release.yaml`, `ci/repository-controls-evidence-test.sh` |
| `ci/repository-controls-evidence-test.sh` | `just check-actions` |
| `ci/package-release-source.sh` | `release.yaml`, `ci/release-identity-test.sh` |
| `ci/package-release-ct-evidence.sh` | `release.yaml` |
| `ci/release-package-guard.sh` | `ci/release-preflight.sh` |
| `ci/release-preflight.sh` | `release.yaml` |
| `ci/write-release-manifest.sh` | `release.yaml`, `ci/release-identity-test.sh` |
| `ci/release-identity-test.sh` | `just check-actions` |
| `ci/publish-immutable-release.sh` | `release.yaml`, `ci/publish-immutable-release-test.sh` |
| `ci/publish-immutable-release-test.sh` | `just check-actions` |
| `ci/release-recipes-test.sh`   | `just check-actions` |
| `ci/pre-push.sh`               | `just push`, `just push-full` |
| `ct/artifacts.sh`              | `just ct`, `just ct-artifacts`, `scripts/ct/full.py` |
| `ct/dudect.sh`                 | `just ct-dudect`, `scripts/ct/full.py` |
| `ct/python.sh`                 | CT recipes, `ci/run-rust-job.sh`, and Python-backed CT, check, and release scripts |
| `update/update-all.sh`         | `just update`, `just update-check` |

The optimized secret-lifecycle inspection performed by
`check/zeroize-evidence.sh` is mapped to its source ownership and host-binary
claim in [`docs/secret-lifecycle.md`](../docs/secret-lifecycle.md).

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
| `ci/run-bench.sh`            | `scripts/bench/bench.sh`, `ci/mlkem-aarch64-gate.sh`, `ci/run-rust-job.sh` |
| `bench/blake3-gap-gate.sh`   | `scripts/ci/run-bench.sh` |

## CI-only (not surfaced via `just`)

| Script | Callers |
|--------|---------|
| `ci/install-tools.sh`          | `.github/actions/setup/action.yaml` |
| `ci/run-rust-job.sh`           | `.github/workflows/_rust-job.yaml` |
| `ci/dependabot-smoke.sh`       | `ci/run-rust-job.sh` |
| `ci/emit-manual-matrix.sh`     | `bench.yaml`, `ct.yaml` |
| `ci/mlkem-aarch64-gate.sh`     | `ci/run-rust-job.sh` |
| `ci/nostd-wasm-suite.sh`       | `ci/cross-targets.sh` |
| `ci/cross-targets.sh`          | `ci/run-rust-job.sh` |

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

Bench results from local runs (`just bench*`) and CI (`/extract-bench` skill
pulls GitHub Actions artifacts) land under:

```
benchmark_results/<YYYY-MM-DD>/<os>/<arch>/results.txt
```

Local runs use the host calendar date and `linux|macos|windows` +
`x86-64|aarch64`. Same layout in CI; the extractor writes into the same tree
so local and CI runs interleave by date without collision.
