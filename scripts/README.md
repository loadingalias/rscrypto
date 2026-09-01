# Scripts Map

Maintainer-owned caller and ownership map for `scripts/`. Keep every `.sh`
listed with its direct entry point or sourcing site. User-facing commands remain
the recipes reported by `just --list`.

## Entry Points (called from `justfile` or CI)

| Script                                    | Callers                                                                                                                           |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `check/affected.sh`                       | `just check`, `just validate`; creates one saved plan for the selected local composition                                         |
| `check/policy.sh`                         | `check/affected.sh`, `ci.yaml`, `qualification.yaml`                                                                              |
| `check/check.sh`                          | `check/affected.sh`, `check/check-all.sh`, `ci.yaml`, `qualification.yaml`                                                        |
| `check/check-all.sh`                      | `just check-all`                                                                                                                  |
| `check/feature-contracts.sh`              | `just feature-contracts`, `scripts/check/affected.sh`, `scripts/check/check-all.sh`, `ci.yaml`, `qualification.yaml`                |
| `check/asm-ledger.sh`                     | `scripts/check/policy.sh`                                                                                                         |
| `check/rsa-asm-provenance.sh`             | `check/asm-ledger.sh`; direct `--archive PATH` reconstructs the three pinned RSA snapshots offline                                |
| `check/signature-asm-provenance.py`       | `check/asm-ledger.sh`; direct `--upstream-repo PATH [--clang PATH]` reproduces the 36 pinned ECDSA, Ed25519, and X25519 snapshots |
| `check/hash-vector-provenance.py`         | `scripts/check/policy.sh`; optional exact upstream checkouts reproduce hash-vector corpora                                        |
| `check/auth-vector-provenance.py`         | `scripts/check/policy.sh`; optional `--upstream-root PATH` reproduces the pinned C2SP/Wycheproof corpus                           |
| `check/feature-boundaries.py`             | `scripts/check/policy.sh`                                                                                                         |
| `check/zeroize-evidence.sh`               | `just check-zeroize-evidence`, `scripts/check/check-all.sh`, `qualification.yaml`                                                 |
| `ci/check-locked-cargo.sh`                | `scripts/ci/actions-policy.sh`                                                                                                    |
| `ci/check-locked-cargo-test.sh`           | `scripts/ci/actions-policy.sh`                                                                                                    |
| `ci/target-contracts.sh`                  | `just target-contract`, `scripts/check/check-all.sh`, `ci.yaml`, `qualification.yaml`                                             |
| `test/test.sh`                            | `just test` (`--all` for the full workspace), `ci.yaml`, `qualification.yaml`                                                     |
| `test/test-examples.sh`                   | `just test-examples`                                                                                                              |
| `test/miri-contracts.sh`                  | `just miri-contract`, `scripts/check/affected.sh`, `ci.yaml`, `qualification.yaml`                                                |
| `test/test-miri.sh`                       | `just test-miri`, `test/miri-contracts.sh`                                                                                        |
| `test/fuzz-contracts.sh`                  | `just fuzz-contract`, `scripts/check/affected.sh`, `ci.yaml`                                                                      |
| `test/test-fuzz.sh`                       | `just test-fuzz`, `test/fuzz-contracts.sh`, `qualification.yaml`                                                                  |
| `test/test-fuzz-scheduler-test.sh`        | `just check-actions`; proves bounded concurrency, exact selection, corpus retention, and aggregated failure                       |
| `test/test-fuzz-asan.sh`                  | `just test-fuzz-asan`, `qualification.yaml`                                                                                       |
| `test/test-rsa-leakage.sh`                | `just test-rsa-leakage`, `qualification.yaml`                                                                                     |
| `test/test-rsa-macos-asm.sh`              | `just test-rsa-macos-asm` on a physical local Apple Silicon Mac                                                                   |
| `test/test-rsa-linux-asm.sh`              | `just test-rsa-linux-asm`, `qualification.yaml` on physical Linux x86-64                                                          |
| `test/test-coverage.sh`                   | `just test-coverage` (`--nextest` or `--fuzz` for one source), `qualification.yaml`                                               |
| `bench/bench.sh`                          | `just bench` (`--quick` for reduced measurement time)                                                                             |
| `bench/profile.sh`                        | `just profile`                                                                                                                    |
| `ci/check-action-pins.sh`                 | `just check-actions`, `ci.yaml`                                                                                                   |
| `ci/actions-policy.sh`                    | `scripts/check/policy.sh`                                                                                                         |
| `ci/check-action-pins-test.sh`            | `just check-actions`                                                                                                              |
| `ci/remote-cache-recipes-test.sh`         | `just check-actions`                                                                                                              |
| `ci/feature-contracts-test.sh`            | `just check-actions`; proves unique compile graphs, focused runtime scopes, and disjoint deterministic shards                      |
| `ci/feature-planning-test.sh`             | `just check-actions`; proves exact algorithm groups, full feature-policy selection, and fail-closed unattributed inputs            |
| `ci/activate-plan.sh`                     | `.github/actions/plan/action.yaml`; validates and exports one transported plan                                                    |
| `ci/require-work.sh`                      | Direct CI/Qualification executors with repository-scoped work                                                                    |
| `ci/emit-manual-matrix-test.sh`           | `just check-actions`                                                                                                              |
| `ci/changed-test-planning-test.sh`        | `just check-actions`                                                                                                              |
| `ci/check-worktree-test.sh`               | `just check-actions`                                                                                                              |
| `ci/pre-push-test.sh`                     | `just check-actions`                                                                                                              |
| `ci/package-release-source.sh`            | `release.yaml`, `ci/release-identity-test.sh`                                                                                     |
| `ci/package-release-ct-evidence.sh`       | `release.yaml`                                                                                                                    |
| `ci/release-package-guard.sh`             | `ci/release-preflight.sh`                                                                                                         |
| `ci/release-preflight.sh`                 | `release.yaml`                                                                                                                    |
| `ci/release-identity-test.sh`             | `just check-actions`; verifies deterministic source packaging and immutable tag identity                                          |
| `ci/publish-immutable-release.sh`         | `release.yaml`, `ci/publish-immutable-release-test.sh`                                                                            |
| `ci/publish-immutable-release-test.sh`    | `just check-actions`                                                                                                              |
| `ci/pre-push.sh`                          | `just push`                                                                                                                       |
| `ct/artifacts.sh`                         | `just ct-artifacts`, `scripts/ct/full.py`                                                                                         |
| `ct/dudect.sh`                            | `just ct-dudect`, `scripts/ct/full.py`                                                                                            |
| `ct/dudect_report_test.py`                | `scripts/check/policy.sh`                                                                                                         |
| `lib/python.sh`                           | Resolves Python 3.11+ for Cargo Rail readers and Python-backed CT, check, benchmark, and release tooling                          |
| `update/update-all.sh`                    | `just update` (`--check` for a non-mutating preview)                                                                              |
| `render_perf_chart.rs`                    | `just chart` compiles and executes this source directly                                                                           |

The optimized secret-lifecycle inspection performed by
`check/zeroize-evidence.sh` is mapped to its source ownership and host-binary
claim in [`docs/secret-lifecycle.md`](../docs/secret-lifecycle.md).

## Cross-platform Check Helpers

| Script                                 | Callers                                                      |
| -------------------------------------- | ------------------------------------------------------------ |
| `check/lint-independent-workspaces.sh` | `scripts/check/check.sh --all`           |
| `check/zig-cc.sh`                      | `scripts/ct/artifacts.sh`, `scripts/ct/binsec.py` |

## Bench Internals

| Script                            | Callers                                                                    |
| --------------------------------- | -------------------------------------------------------------------------- |
| `ci/run-bench.sh`                 | `scripts/bench/bench.sh`, `ci/mlkem-aarch64-gate.sh`, `bench.yaml` |
| `bench/blake3-gap-gate.sh`        | `scripts/ci/run-bench.sh`                                                  |
| `bench/benchmark_catalog.py`      | `ci/run-bench.sh`, `bench/profile.sh`, `benchmark_catalog_test.py`         |
| `bench/benchmark_catalog_test.py` | `scripts/check/policy.sh`                                                  |

## Constant-Time Internals

| Script                              | Callers and validation                                                                                                  |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `ct/full.py`                        | `just ct-full`, `ct/ci.sh`                                                                                             |
| `ct/binsec.py`                      | `just ct-binsec`, `ct/full.py`                                                                                           |
| `ct/validate.py`                    | `just ct-validate`, `ct/full.py`                                                                                         |
| `ct/asm_heuristics.py`              | `ct/artifacts.sh`; parsed hazards are covered by `ct/asm_heuristics_test.py`                                             |
| `ct/asm_heuristics_test.py`         | `scripts/check/policy.sh`                                                                                                |
| `ct/dudect_report.py`               | `ct/dudect.sh`; parsing and gate behavior are covered by `ct/dudect_report_test.py`                                      |
| `ct/evidence_validation_test.py`    | `scripts/check/policy.sh`; covers symbol reconstruction, artifact packaging, heuristics, and release-evidence validation |
| `ct/package_evidence.py`            | `ct/ci.sh`; compact-package behavior is covered by `ct/evidence_validation_test.py`                                     |
| `ct/provenance.py`                  | `ct/artifacts.sh`; emitted provenance is consumed and checked by `ct/validate.py` and the full pipeline                 |
| `ct/symbolize_linked_binary.py`     | `ct/artifacts.sh`; parsers and reconstruction are covered by `ct/evidence_validation_test.py`                           |
| `ct/validate_release_evidence.py`   | `ci/package-release-ct-evidence.sh`; covered by CT evidence validation tests                                             |

## CI-only (not surfaced via `just`)

| Script                     | Callers                                       |
| -------------------------- | --------------------------------------------- |
| `ci/install-tools.sh`      | `qualification.yaml` supply-chain/fuzz lanes and `ct.yaml` formal-analysis lanes |
| `ci/setup-toolchain.sh`    | `.github/actions/rust/action.yaml`            |
| `ci/native-platform.sh`    | `ci/target-contracts.sh`                      |
| `ci/emit-manual-matrix.sh` | `bench.yaml`, `ct.yaml`                       |
| `ci/mlkem-aarch64-gate.sh` | `qualification.yaml`                          |
| `ci/nostd-wasm-suite.sh`   | `ci/cross-targets.sh`                         |
| `ci/cross-targets.sh`      | `ci/target-contracts.sh`                      |

## Shared Libraries (sourced, not invoked)

| Script                     | Sourced by                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------ |
| `lib/common.sh`            | Check/test entry points plus native, cross-target, benchmark-gate, and pre-push CI scripts |
| `lib/rail-plan.sh`         | `scripts/lib/common.sh`                                                                    |
| `lib/fuzz-packages.sh`     | `scripts/test/test-fuzz.sh`, `scripts/test/test-coverage.sh`                               |
| `lib/feature-profiles.sh`  | `check/feature-contracts.sh`, `scripts/ci/nostd-wasm-suite.sh`                         |
| `lib/targets.sh`           | `scripts/ci/cross-targets.sh`                                                           |
| `lib/target-matrix.sh`     | `scripts/lib/targets.sh`, `scripts/ci/target-contracts.sh`                              |
| `lib/toolchain.sh`         | Toolchain setup, Miri/fuzz helpers, and cross-target check scripts                          |
| `lib/ci-tool-integrity.sh` | `ci/nostd-wasm-suite.sh`, `just check-actions`                                             |

## Python boundary

Repository tooling requires Python 3.11 or newer and uses only the standard
library. Python remains where structured TOML/JSON, binary and archive
inspection, statistical evidence, or CT orchestration makes shell materially
less safe or maintainable. Cargo Rail's bundled plan reader is also Python and
runs through the same resolver. Simple JSON selection and redaction stays in
`jq`; no Python compatibility package, virtual environment, or package-manager
bootstrap remains.

## CI tool integrity

Direct executable downloads are declared in
`.config/ci-tool-archives.tsv`. Each supported host has one exact version,
filename, HTTPS URL, and repository-owned SHA-256; the shared verifier checks
the digest before extraction, installation, or execution. OCI tools use an
image digest in their local action definition.

Package-manager tools install into a fresh runner-temporary root. The installer
has only three modes: `supply-chain`, `fuzz`, and `ct-linux`. Cargo installs are
exact-versioned. Ubuntu packages come from signed metadata, and the OPAM
repository is pinned to one commit before BINSEC is built. Rustup installs only
the exact repository toolchain contracts. Actions policy does not require
`yq`; it uses the already-required Ruby and Python standard libraries.
Qualification installs its two exact coverage tools from supported prebuilt
releases with fallback disabled.

## CI architecture

The workflow split follows proof domain and frequency, not algorithms or CPU
architectures:

| Workflow | Responsibility |
| --- | --- |
| `ci.yaml` | Fast affected pull-request and main-branch gate |
| `qualification.yaml` | Weekly and release-grade cross-platform assurance |
| `ct.yaml` | Reusable/manual constant-time evidence matrix |
| `bench.yaml` | Manual performance measurements on named hardware |
| `release.yaml` | Exact-commit package and publication transaction |
| `scorecard.yaml` | GitHub supply-chain scorecard |

`release.yaml` calls `qualification.yaml`, which calls `ct.yaml`; both calls
stay on the exact release commit. IBM, RISC-V, Windows, macOS, x86-64, and
AArch64 are matrix rows. RSA and ML-KEM are assurance lanes inside
Qualification; they are not workflow boundaries.

Cargo Rail plans exactly once in `ci.yaml`. The planner also runs selected cheap
repository policy, so a workflow-only change does not start a second runner.
Built-in Cargo work starts the single warm host-Rust job; `contracts.features`
uses `.config/feature-matrix.json` to select algorithm or capability groups,
resolves every affected compile profile from Cargo's feature graph, and packs
only those profiles into at most two compile and three runtime shards.
Manifest, catalog, shared-surface, or unattributed inputs widen to the complete
59-compile/9-runtime contract. `targets.platforms` independently materializes
only affected platform proof rows; built-in `dependency-policy` starts the
dependency audit. The core job already owns ordinary Linux x86-64 and is not
duplicated. Affected `assurance.ct` work runs one cold x86-64 structural gate:
release harness construction, generated-code inspection, and strict manifest
and artifact validation. Affected `assurance.rsa` work runs the x86-64 assembly
differential and symbol contract, with eligible compiler work using the same R2
policy as the other trusted native lanes. Neither decision starts physical
timing, formal analysis, or a cross-platform assurance sweep in pull requests.

Affected `assurance.miri` work selects only the portable unsafe-boundary row,
the focused RSA row, or both. Affected `assurance.fuzz` work selects algorithm-
sized rows and unions their exact target names into one executor, so tool setup
and compatible builds are shared without converting each target into a job.
Both lanes are deliberately cache-cold. Manifest, catalog, shared harness, and
unattributed inputs widen fail-closed; the root lockfile alone selects neither
nightly lane. Qualification still runs both Miri rows, the portable row again
under Tree Borrows, every fuzz target, and every committed corpus under ASan.
Miri remains on x86-64 because it forces portable execution; deep Linux and
macOS AArch64 rows own native runtime and backend-differential proof instead of
duplicating interpreter and fuzz hosts.

The selected Actions policy lane downloads the exact checksum-verified
actionlint release and the exact prebuilt Zizmor release, then runs the same
`scripts/ci/actions-policy.sh` entry point as `just check-actions`. No CI linter
is compiled from source.

Qualification captures one `--all` plan, restores the complete feature
contract, and materializes every platform catalog row as an independent retry
unit. Its reusable CT workflow verifies that same plan once, checks out the
planned commit on every evidence host, and retains the complete physical/formal
matrix. Core test jobs and coverage install the exact prebuilt Nextest release
with source fallback disabled. Coverage runs deterministic nextest and
committed-corpus replay in its own lane. Optimized zeroization runs cold and
retains no compiler cache. RSA
leakage and cross-architecture evidence likewise remain qualification work.
Generic cross compilation stays on Linux x86-64;
hosted and donated machines run only irreducibly native evidence. Linux x86-64
rows restore and verify the plan. Other rows check out the exact planned commit;
the local Cargo Rail issue records the missing lean verifier path for donated
architectures and macOS x86-64.

The Cargo Rail compiler cache is acceleration, never selection or correctness
authority. `just rail-cache-setup` previews, installs, and probes the same
remapped policy used by development machines. CI reads R2 for trusted PRs and
writes only from protected `main`; missing fork secrets disable the cache.
Qualification enables reuse for host, feature-contract, and supported native
platform rows; release preflight is read-only. Cross targets, Clippy, rustdoc,
doctests, Miri, fuzzing, CT, benchmarks, macOS x86-64, and donated hosts stay
cold because the released cache deliberately bypasses or cannot install on
those classes.

Local and remote development use the same affected commands: `just plan`,
`just check`, `just test`, and `just validate`. `just check` runs selected
compile feature contracts; `just validate` adds selected runtime contracts and
shares one saved plan across policy, checks, feature contracts, and tests.
`just feature-contracts [compile|runtime] [N/M]` reproduces any CI shard.
`just target-contract ROW [shallow|deep]` reproduces any independently
executable platform row, locally or through `ssh-just`.
`just miri-contract ROW` and `just fuzz-contract ROW` reproduce the same
algorithm-sized proof rows used by affected CI. `just validate` consumes the
saved local plan and adds only its selected Miri and fuzz rows after ordinary
tests.
`just ct-structural` reproduces the affected CI constant-time structure gate;
`just ct-full` and `just test-rsa-leakage` remain deliberate assurance commands.
`ssh-just TARGET validate` creates the plan after the development machine's
exact repository sync; provider lifecycle and short-lived R2 credentials remain
outside this repository.

## Results layout

Bench results from local runs (`just bench`) and CI (`/extract-bench` skill
pulls GitHub Actions artifacts) land under:

```
benchmark_results/<YYYY-MM-DD>/<os>/<arch>/results.txt
```

Local runs use the host calendar date and `linux|macos|windows` +
`x86-64|aarch64`. Same layout in CI; the extractor writes into the same tree
so local and CI runs interleave by date without collision.
