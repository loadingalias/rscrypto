# Scripts Map

Maintainer-owned caller and ownership map for `scripts/`. Keep every `.sh`
listed with its direct entry point or sourcing site. User-facing commands remain
the recipes reported by `just --list`.

## Entry Points (called from `justfile` or CI)

| Script                                    | Callers                                                                                                                           |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `check/check.sh`                          | `just check`                                                                                                                      |
| `check/check-all.sh`                      | `just check-all`                                                                                                                  |
| `check/check-feature-matrix.sh`           | `just check-feature-matrix`, `scripts/check/check.sh`, `ci/run-rust-job.sh`                                                       |
| `check/asm-ledger.sh`                     | `scripts/check/check.sh`                                                                                                          |
| `check/rsa-asm-provenance.sh`             | `check/asm-ledger.sh`; direct `--archive PATH` reconstructs the three pinned RSA snapshots offline                                |
| `check/signature-asm-provenance.py`       | `check/asm-ledger.sh`; direct `--upstream-repo PATH [--clang PATH]` reproduces the 36 pinned ECDSA, Ed25519, and X25519 snapshots |
| `check/hash-vector-provenance.py`         | `scripts/check/check.sh`; optional exact upstream checkouts reproduce hash-vector corpora                                         |
| `check/auth-vector-provenance.py`         | `scripts/check/check.sh`; optional `--upstream-root PATH` reproduces the pinned C2SP/Wycheproof corpus                            |
| `check/feature-boundaries.py`             | `scripts/check/check.sh`                                                                                                          |
| `check/zeroize-evidence.sh`               | `just check-zeroize-evidence`, `scripts/check/check-all.sh`                                                                       |
| `ci/ci-check.sh`                          | `just ci-check`, `ci/run-rust-job.sh`                                                                                             |
| `ci/check-locked-cargo.sh`                | `ci/ci-check.sh`                                                                                                                  |
| `ci/check-locked-cargo-test.sh`           | `ci/ci-check.sh`                                                                                                                  |
| `ci/capture-cache-status.sh`              | `_rust-job.yaml` cache telemetry                                                                                                  |
| `ci/capture-cache-status-test.sh`         | `just check-actions`                                                                                                              |
| `ci/native-check.sh`                      | `ci/run-rust-job.sh`                                                                                                              |
| `test/test.sh`                            | `just test` (`--all` for the full workspace), `ci/run-rust-job.sh`                                                                |
| `test/test-examples.sh`                   | `just test-examples`; executed in CI through `ci/run-rust-job.sh`                                                                 |
| `test/test-feature-matrix.sh`             | `just test-feature-matrix`, `scripts/check/check.sh`, `ci/run-rust-job.sh`                                                        |
| `test/test-miri.sh`                       | `just test-miri`, `ci/run-rust-job.sh`                                                                                            |
| `test/test-fuzz.sh`                       | `just test-fuzz`, `ci/run-rust-job.sh`                                                                                            |
| `test/test-fuzz-scheduler-test.sh`        | `just check-actions`                                                                                                              |
| `test/test-fuzz-asan.sh`                  | `just test-fuzz-asan`, `ci/run-rust-job.sh`                                                                                       |
| `test/test-rsa-leakage.sh`                | `just test-rsa-leakage`, `ci/run-rust-job.sh`                                                                                     |
| `test/test-rsa-macos-asm.sh`              | `just test-rsa-macos-asm` on a physical local Apple Silicon Mac                                                                   |
| `test/test-coverage.sh`                   | `just test-coverage` (`--nextest` or `--fuzz` for one source), `weekly.yaml`                                                       |
| `bench/bench.sh`                          | `just bench` (`--quick` for reduced measurement time)                                                                             |
| `bench/profile.sh`                        | `just profile`                                                                                                                    |
| `ci/check-action-pins.sh`                 | `just check-actions`, `ci/ci-check.sh`, `ci/dependabot-smoke.sh`                                                                  |
| `ci/check-action-pins-test.sh`            | `just check-actions`, `ci/dependabot-smoke.sh`                                                                                    |
| `ci/tool-integrity-test.sh`               | `just check-actions`                                                                                                              |
| `ci/remote-cache-recipes-test.sh`         | `just check-actions`                                                                                                              |
| `ci/dependabot-smoke-test.sh`             | `just check-actions`                                                                                                              |
| `ci/check-ci-ownership.sh`                | `just check-actions`, `ci/check-ci-ownership-test.sh`                                                                             |
| `ci/check-ci-ownership-test.sh`           | `just check-actions`                                                                                                              |
| `ci/run-rust-job-test.sh`                 | `just check-actions`                                                                                                              |
| `ci/emit-manual-matrix-test.sh`           | `just check-actions`                                                                                                              |
| `ci/materialize-rail-plan.sh`             | `ci.yaml`, `weekly.yaml`, and its regression test                                                                                 |
| `ci/materialize-rail-plan-test.sh`        | `just check-actions`                                                                                                              |
| `ci/changed-test-planning-test.sh`        | `just check-actions`                                                                                                              |
| `ci/check-worktree-test.sh`               | `just check-actions`                                                                                                              |
| `ci/pre-push-test.sh`                     | `just check-actions`                                                                                                              |
| `ci/release-evidence-check.sh`            | `just release-tag`, `release.yaml`, `ci/release-evidence-check-test.sh`                                                           |
| `ci/release-evidence-check-test.sh`       | `just check-actions`                                                                                                              |
| `ci/release-ct-recovery-check.sh`         | `release.yaml`, `ci/release-ct-recovery-check-test.sh`                                                                            |
| `ci/release-ct-recovery-check-test.sh`    | `just check-actions`                                                                                                              |
| `ci/repository-controls-evidence.sh`      | `just release-tag`, `release.yaml`, `ci/repository-controls-evidence-test.sh`                                                     |
| `ci/repository-controls-evidence-test.sh` | `just check-actions`                                                                                                              |
| `ci/package-release-source.sh`            | `release.yaml`, `ci/release-identity-test.sh`                                                                                     |
| `ci/package-release-ct-evidence.sh`       | `release.yaml`                                                                                                                    |
| `ci/release-package-guard.sh`             | `ci/release-preflight.sh`                                                                                                         |
| `ci/release-preflight.sh`                 | `release.yaml`                                                                                                                    |
| `ci/write-release-manifest.sh`            | `release.yaml`, `ci/release-identity-test.sh`                                                                                     |
| `ci/release-identity-test.sh`             | `just check-actions`                                                                                                              |
| `ci/publish-immutable-release.sh`         | `release.yaml`, `ci/publish-immutable-release-test.sh`                                                                            |
| `ci/publish-immutable-release-test.sh`    | `just check-actions`                                                                                                              |
| `ci/release-recipes-test.sh`              | `just check-actions`                                                                                                              |
| `ci/pre-push.sh`                          | `just push`                                                                                                                       |
| `ct/artifacts.sh`                         | `just ct-artifacts`, `scripts/ct/full.py`                                                                                         |
| `ct/dudect.sh`                            | `just ct-dudect`, `scripts/ct/full.py`                                                                                            |
| `ct/dudect_report_test.py`                | `scripts/check/check.sh`                                                                                                          |
| `lib/python.sh`                           | Resolves Python 3.11+ for Cargo Rail readers and Python-backed CT, check, benchmark, and release tooling                          |
| `update/update-all.sh`                    | `just update` (`--check` for a non-mutating preview)                                                                              |
| `render_perf_chart.rs`                    | `just chart` compiles and executes this source directly                                                                           |

The optimized secret-lifecycle inspection performed by
`check/zeroize-evidence.sh` is mapped to its source ownership and host-binary
claim in [`docs/secret-lifecycle.md`](../docs/secret-lifecycle.md).

## Cross-platform Check Helpers

| Script                                 | Callers                                                      |
| -------------------------------------- | ------------------------------------------------------------ |
| `check/check-win.sh`                   | `scripts/check/check-all.sh`             |
| `check/check-zig.sh`                  | `scripts/check/check-all.sh`             |
| `check/lint-independent-workspaces.sh` | `scripts/check/check.sh --all`           |
| `check/zig-cc.sh`                      | `scripts/check/check-zig.sh`             |

## Bench Internals

| Script                            | Callers                                                                    |
| --------------------------------- | -------------------------------------------------------------------------- |
| `ci/run-bench.sh`                 | `scripts/bench/bench.sh`, `ci/mlkem-aarch64-gate.sh`, `ci/run-rust-job.sh` |
| `bench/blake3-gap-gate.sh`        | `scripts/ci/run-bench.sh`                                                  |
| `bench/benchmark_catalog.py`      | `ci/run-bench.sh`, `bench/profile.sh`, `benchmark_catalog_test.py`         |
| `bench/benchmark_catalog_test.py` | `scripts/check/check.sh`                                                   |

## Constant-Time Internals

| Script                              | Callers and validation                                                                                                  |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `ct/full.py`                        | `just ct-full`, `ci/run-rust-job.sh`; its dispatch contract is covered by `ci/run-rust-job-test.sh`                    |
| `ct/binsec.py`                      | `just ct-binsec`, `ct/full.py`                                                                                           |
| `ct/validate.py`                    | `just ct-validate`, `ct/full.py`                                                                                         |
| `ct/asm_heuristics.py`              | `ct/artifacts.sh`; parsed hazards are covered by `ct/asm_heuristics_test.py`                                             |
| `ct/asm_heuristics_test.py`         | `scripts/check/check.sh`                                                                                                 |
| `ct/dudect_report.py`               | `ct/dudect.sh`; parsing and gate behavior are covered by `ct/dudect_report_test.py`                                      |
| `ct/evidence_validation_test.py`    | `scripts/check/check.sh`; covers symbol reconstruction, artifact packaging, heuristics, and release-evidence validation |
| `ct/package_evidence.py`            | `ci/run-rust-job.sh`; compact-package behavior is covered by `ct/evidence_validation_test.py`                           |
| `ct/provenance.py`                  | `ct/artifacts.sh`; emitted provenance is consumed and checked by `ct/validate.py` and the full pipeline                 |
| `ct/symbolize_linked_binary.py`     | `ct/artifacts.sh`; parsers and reconstruction are covered by `ct/evidence_validation_test.py`                           |
| `ct/validate_release_evidence.py`   | `ci/package-release-ct-evidence.sh`; covered by CT evidence validation and release-evidence regression tests            |

## CI-only (not surfaced via `just`)

| Script                     | Callers                                       |
| -------------------------- | --------------------------------------------- |
| `ci/install-tools.sh`      | `.github/actions/setup/action.yaml`           |
| `ci/install-codecov.sh`    | `weekly.yaml`                                 |
| `ci/setup-toolchain.sh`    | `.github/actions/setup-toolchain/action.yaml` |
| `ci/run-rust-job.sh`       | `.github/workflows/_rust-job.yaml`            |
| `ci/dependabot-smoke.sh`   | `ci/run-rust-job.sh`                          |
| `ci/emit-manual-matrix.sh` | `bench.yaml`, `ct.yaml`                       |
| `ci/mlkem-aarch64-gate.sh` | `ci/run-rust-job.sh`                          |
| `ci/nostd-wasm-suite.sh`   | `ci/cross-targets.sh`                         |
| `ci/cross-targets.sh`      | `ci/run-rust-job.sh`                          |

## Shared Libraries (sourced, not invoked)

| Script                     | Sourced by                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------ |
| `lib/common.sh`            | Check/test entry points plus native, cross-target, benchmark-gate, and pre-push CI scripts |
| `lib/rail-plan.sh`         | `scripts/lib/common.sh`                                                                    |
| `lib/fuzz-packages.sh`     | `scripts/test/test-fuzz.sh`, `scripts/test/test-coverage.sh`                               |
| `lib/feature-profiles.sh`  | Feature-matrix scripts, `scripts/check/check-all.sh`, `scripts/ci/nostd-wasm-suite.sh`      |
| `lib/targets.sh`           | `scripts/check/check-all.sh`, `scripts/check/check-zig.sh`, `scripts/ci/cross-targets.sh` |
| `lib/target-matrix.sh`     | `scripts/lib/targets.sh`, `scripts/ci/ci-check.sh`                                         |
| `lib/toolchain.sh`         | Toolchain setup, Miri/fuzz helpers, and cross-target check scripts                          |
| `lib/ci-tool-integrity.sh` | `ci/install-codecov.sh`, `ci/nostd-wasm-suite.sh`                                          |

## Python boundary

Repository tooling requires Python 3.11 or newer and uses only the standard
library. Python remains where structured TOML/JSON, binary and archive
inspection, statistical evidence, or CT orchestration makes shell materially
less safe or maintainable. Cargo Rail's bundled plan reader is also Python and
runs through the same resolver. Simple JSON selection and redaction stays in
`jq`; no Python compatibility package, virtual environment, or package-manager
bootstrap remains.

## Script contracts

This table is the ownership audit for inputs, side effects, failure policy, and
tests. A script belongs to exactly one row; its concrete caller remains in the
maps above.

| Owner | Inputs | Side effects | Failure policy and evidence |
| --- | --- | --- | --- |
| Shared libraries | Sourcing script arguments, repository paths, typed target and feature catalogs | Define functions and readonly data in the caller; no independent entry-point effects | Reject malformed catalogs, unknown selectors, and missing tools; exercised through every caller and the CI planning regression suite |
| Local checks and tests | Recipe arguments, Cargo metadata, manifests, target/feature catalogs, vectors, and explicit environment selectors | Cargo build output plus bounded logs or evidence under `target/` | Fail on the first violated contract or aggregate named failures without weakening assertions; focused Python tests, shell regression tests, and `just check` own coverage |
| CI planning and execution | GitHub event fields, immutable plan artifacts, typed matrix rows, operation selectors, and repository variables | Materialize plan-bound matrices, run repository commands, and write bounded artifacts under `target/` or `ci-evidence/` | Reject missing or mismatched plan identity, commit, operation, target, tool mode, and trust mode before execution; `just check-actions` runs every shell regression fixture plus `actionlint` and `zizmor` |
| Tool installation | Exact tool mode, toolchain contract, integrity catalog, runner OS, and architecture | Install exact tools into runner-temporary roots and emit environment paths | Reject absent checksums, version drift, unsupported hosts, mutable downloads, or unauthenticated Cargo Rail; integrity and installer fixtures run under `just check-actions` |
| Constant-time evidence | `ct.toml`, target/profile/gate selectors, exact toolchain and linker state, harness output, and release-bound artifacts | Write target-scoped assembly, disassembly, timing, formal-analysis, provenance, report, and package files under `target/` | Distinguish pass, diagnostic, unsupported, timeout, and blocking failure exactly as `ct.toml` declares; parser/unit fixtures run in `just check`, while target execution uses the focused CT recipes |
| Benchmarks and profiling | Catalog selectors, platform facts, filters, sample mode, and profiler arguments | Write Criterion, structural, profile, code-generation, or chart artifacts in their documented result roots | Reject unknown catalog rows and unsupported platform/tool combinations; catalog tests run in `just check`, while measurements remain non-correctness evidence |
| Release and update | Exact commit/tag, downloaded qualification artifacts, Cargo Rail release state, lockfiles, and `--check` preview mode | Prepare locks and release manifests, verify or publish immutable artifacts, or update coordinated manifests | Fail closed on identity, evidence, worktree, package, signature, or publication mismatch; release and update adapters are covered by `just check-actions` and the release recipes |
| Remote support | Target plus arguments passed by the `ssh-*` recipes | No repository script owns provider state; the external `dev-machine` front door owns creation, sync, bootstrap, and teardown | Provider and lease validation live in `dev-machine`; repository recipes preserve the `rscrypto` project scope and propagate failure |
| Performance chart | `benchmark_results/OVERVIEW.md` | Rewrites `assets/readme/perf.svg` through `just chart` | Rejects missing or malformed benchmark rows during compilation or execution; the generated SVG is reviewed with its source data |

## CI tool integrity

Direct executable downloads are declared in
`.config/ci-tool-archives.tsv`. Each supported host has one exact version,
filename, HTTPS URL, and repository-owned SHA-256; the shared verifier checks
the digest before extraction, installation, or execution. OCI tools use an
image digest in their local action definition.

Package-manager tools install into a fresh runner-temporary root; CI never
restores Cargo binaries, Cargo install metadata, Go module state, or OPAM
switches from a cache. Cargo installs exact crates from crates.io and
authenticates crate contents against registry checksums. Go installs an exact
module through the public checksum database. Ubuntu 24.04 APT dependencies
resolve from signed repository metadata; installation pins each signed
candidate selected after the metadata refresh, verifies the installed version,
and refuses downgrades. OPAM uses exact packages from a repository pinned to a
full Git commit and verifies package source hashes from that immutable metadata.
CT formal reports bind the resulting BINSEC executable by SHA-256. Rustup
receives only the exact stable or nightly contract declared in
`rust-toolchain.toml` and `.config/toolchains.toml`; runner images must provide
rustup, which verifies component downloads against the exact distribution
manifest, because network bootstrap installers are rejected.

## Planned CI

Cargo Rail creates one validated named-work plan per CI or Qualification run.
`.config/ci-plan-variants.json` owns the selectable suite rows and their typed
execution dimensions; repository scripts own command implementations, while
workflows enforce trust. The planner uploads the exact plan and bundled strict
reader together. Every selected job checks out the plan-bound commit, verifies
the plan identity and complete checkout, and then executes its catalog
operation. No consumer replans or infers work from changed paths.

The repository-scoped `ci-policy` work item widens the matrix to the complete
catalog when shared workflows or dispatch infrastructure change. This keeps
source-only changes narrow without pretending that a shared executor edit can
be validated by one arbitrarily selected row.

Pull requests select affected Cargo work and CI rows. Manual CI and Qualification use
Cargo Rail's typed all-work override. The planner installs the authenticated
Surface component, but `.config/rail.toml` temporarily disables Surface while
Cargo Rail cannot distinguish expected `compile_fail` doctest invocations from
compiler failures. Planning and every selected CI row remain fail-closed.

Cargo Rail Action v8.2.0 does not publish verified native cache components for
the IBM Z and POWER hosts. Those two native rows still install Cargo Rail core,
verify the saved plan, and run in full; only compiler-result reuse is skipped.

## Compiler-result reuse

`.github/actions/setup/action.yaml` is the only CI compiler-cache owner. It
uses the same immutable Cargo Rail action revision and authenticated Cargo Rail
version as pull-request planning, installs the cache before repository Cargo
tools, and leaves tool executables and package-manager state uncached.

The repository variable `CARGO_RAIL_CACHE_URL` selects rscrypto's canonical
Cloudflare R2 L2 authority. The bucket-scoped
`CARGO_RAIL_R2_READ_ACCESS_KEY_ID` / `CARGO_RAIL_R2_READ_SECRET_ACCESS_KEY`
secret pair can only read it; the corresponding `WRITE` pair can read and
write it. The URL contains no credentials and does not belong in
`.config/rail.toml`. A missing secret pair skips L2 cleanly, which keeps fork
and Dependabot jobs correct without disclosing repository credentials.

CI applies qualified root remapping and requires Cargo Rail's authenticated
provider/protocol probe whenever the URL is configured. Configure
the `just ssh-*` machines with the same normalized URL and remap policy when
their provider identities may share compiler results. Repository code selects
trust (`read` for pull requests, `read-write` for trusted jobs);
`~/dev-machines` owns the corresponding remote machine setup and credentials.
Different URLs or physical-root mode produce isolated caches by design.

Ordinary, pull-request, qualification, and release jobs select `read`; their
provider credential is also read-only. The affected main-branch seeder alone
selects `read-write` and receives the distinct writer credential. Setup fails
if R2 authentication or the `native-v6` protocol marker is unavailable; later
per-compilation transport failures take Cargo Rail's verified fallback path.
Representative compiler jobs preserve a redacted cache-status artifact so
local and remote origins, misses, bypasses, conflicts, failures, capacity,
mode, provider, and setup health remain visible.

Local development uses the same machine-owned setup documented in
[`CONTRIBUTING.md`](../CONTRIBUTING.md). Miri and optimized zeroization evidence
set `CARGO_RAIL_CACHE=off` because those checks require a deliberately cold
compiler path. Cross-target and otherwise unsupported compiler operations rely
on Cargo Rail's typed bypass instead of clearing a global wrapper.

## Runner and container audit

The 2026-08-28 audit used CI run `33140323747` and Qualification run
`33094159266`. GitHub step timestamps produced this cold-run breakdown:

| Representative job | Checkout | Setup | Repository operation | Artifact/upload |
| --- | ---: | ---: | ---: | ---: |
| Linux x86-64 native | 4 s | 545 s | 1,608 s | none |
| Linux AArch64 native | 4 s | 443 s | 1,444 s | none |
| Windows x86-64 native | 9 s | 13 s | 281 s | none |
| Linux feature contracts | 5 s | 104 s | 3,544 s | none |
| Linux coverage | 13 s | 565 s | 2,100 s | 7 s |

The x86-64 native log further split setup into approximately 11 seconds for
the exact Rust toolchain, 426 seconds to compile `cargo-nextest`, and 108
seconds to compile `just`. RunsOn reported 22.33 seconds from job creation to a
ready runner. The repository-operation column contains compilation and command
execution because each repository script deliberately remains one policy
boundary; Cargo and Nextest logs retain the finer per-command timing.

`.github/runs-on.yml` uses the maintained Ubuntu 24 full x86-64 and AArch64
images, provider labels for physical evidence, non-spot runners, bounded gp3
volumes, and no RunsOn extras or MagicCache. The ordinary CI lanes allow the
documented family fallback; benchmark and evidence lanes retain explicit
families. The existing Linux setup cost is material, but this change installs
Cargo Rail before Cargo tools so the authenticated L2 can reuse those exact
compilations. Keep the maintained images until cache-status artifacts from
seeded runs show the residual setup cost. Introduce versioned custom Linux
images only if that evidence still shows repeated tool compilation; the
13-second Windows setup does not justify a custom image.

The only repository Dockerfile is `oss-fuzz/Dockerfile`. It pins the OSS-Fuzz
Rust builder by digest, accepts an explicit source ref, crosses no provider
credential boundary, and contains no compiler cache or `cargo-chef` layer.
Cargo Rail is therefore the sole compiler-result cache, and `cargo-chef` has no
retained role.

## Results layout

Bench results from local runs (`just bench`) and CI (`/extract-bench` skill
pulls GitHub Actions artifacts) land under:

```
benchmark_results/<YYYY-MM-DD>/<os>/<arch>/results.txt
```

Local runs use the host calendar date and `linux|macos|windows` +
`x86-64|aarch64`. Same layout in CI; the extractor writes into the same tree
so local and CI runs interleave by date without collision.
