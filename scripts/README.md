# Scripts

The scripts in this directory implement the local development, test, evidence,
and benchmark commands.
The entry points for users are the recipes that `just --list` shows.
Those entry points call the supporting modules.

## Check entry points

| Script                                 | Caller |
| -------------------------------------- | ------ |
| `check/check.sh`                       | `just check`, `just ci-check`, `just ci-check-target` |
| `check/compat.py`                      | `just ci-compat` |
| `test/test-musl.sh`                    | `just test-musl` |
| `check/macos.sh`                       | `just check-macos`, background Mac qualification |
| `check/macos_async.py`                 | local pre-push hook, `just qualify-macos`, `just macos-status` |
| `check/qualified.py`                   | local qualification reuse and release status identity |
| `check/dependencies.sh`                | `just ci-policy`, and the dependency checks in `just check` |
| `check/lint-independent-workspaces.sh` | `check/check.sh` |
| `asm/p384.py check`                    | `check/check.sh` |
| `asm/provenance.py check`              | `check/check.sh` |

`check/check_runner_test.py` tests command selection, repair behavior, and failure propagation with substitute executors.
Run it with `scripts/lib/python.sh scripts/check/check_runner_test.py`.

`check/lint-independent-workspaces.sh` lints every independent Cargo workspace that Git tracks or would add.
Ignored manifests, such as local evidence crates, are not repository policy.

## Generated P-384 assembly

`asm/p384_x86_64.py` and `asm/p384_aarch64.py` generate two things in `src/auth/p384_x86_64.rs` and `src/auth/p384_aarch64.rs`: the P-384 field kernels, and the x86-64 in-place point doubling.
The other assembly in those files is written by hand.
`asm/p384.py` is the entry point:

- `check` (run by `just check`) fails when a committed block differs from its generator.
- `write` generates the blocks again after a generator change.
- `simulate [--cases N]` runs every generated kernel in bit-exact x86-64 and AArch64 models against Python integers.
  It includes edge values and carry chains that random operands almost never reach.

`asm/p384_test.py` (part of `just test-scripts`) checks that edits and single dropped carries are detected.
Native differential tests against the portable implementation are the runtime evidence.

## Derived assembly provenance

Each `src/**/*_assembly_provenance.tsv` manifest pins the upstream archive, the upstream members, and the SHA-256 of
every committed output derived from them.
`asm/provenance.py check` fails when an output differs from its manifest, or when assembly under `src/` without an
`rscrypto contributors` copyright header has no manifest output.
It does not download upstream archives.

## Test entry points

| Script                   | Caller |
| ------------------------ | ------ |
| `test/test.sh`           | `just test` |
| `test/cross.py`          | `just test-cross prepare TARGET ARCHIVE`, `just test-cross run TARGET ARCHIVE` |
| `test/doctest_bundle.py` | RISC-V doctest compilation and execution on the target |
| `test/test-examples.sh`  | `just test-examples` |
| `test/test-miri.sh`      | `just test-miri` |
| `test/test-fuzz.sh`      | `just test-fuzz` |
| `test/test-fuzz-asan.sh` | `just test-fuzz-asan` |
| `test/test-coverage.py`  | `just test-coverage` |
| `test/coverage_run.py`   | Coverage-only nextest target runner, invoked by `test-coverage.py` |
| `test/test-rsa-asm.sh`   | `just test-rsa-linux-asm`, `just test-rsa-macos-asm` |

- `just test-scripts` runs the argument-forwarding, toolchain, test, check, and fuzz regressions
  with substitute executors.
- Check recipes run `test/fixtures.py` before compilation.
  It verifies Git-visible payloads under `testdata/` and `tests/vectors/`.
  Each directory's `SHA256SUMS` must cover its payloads, including new untracked inputs.
  Missing, extra, duplicate, changed, or symlinked inputs fail.
  `README.md` and `NIST-NOTICE.txt` are provenance metadata, not hashed payloads.
  Git-ignored scratch files stay outside this inventory.
- Each ChaCha20 differential test must report execution of every backend marked runnable.
  All three tests must report the same inventory and target architecture.
  The inventory includes required features and availability.
  Portable-only and unavailable results do not claim accelerated execution.
- `just ct-test` runs the CT tooling regressions, the harness self-tests,
  and the raw timing exporter tests, without timing cases.
- Coverage publishes raw LCOV, its merged profile, the exact executable inventory, and `provenance.json`.
  `provenance.json` binds the report to the effective source, tool versions, suite arguments,
  the environment that affects execution, and the core artifact hashes.
  Schema 2 records discovered tests, ignored/filter status, executed tests, executable hashes,
  and profile counts per test and suite.
  The coverage-only target runner separates discovery profiles from test profiles.
  Collection requires every selected test to execute successfully and produce a nonempty profile.
  Executable replacement and incomplete publication fail; `provenance.json` is published last.
  Per-suite LCOV files remain in `coverage/suites/` and are hashed in provenance.
  The replay helper records consumed paths only during coverage collection.
  The collector reconciles those receipts with corpus hashes taken before and after each suite,
  including ignored local discoveries when requested.
  `test/coverage_test.py` exercises attribution and deliberate publication failures through `just test-scripts`.
  The bounded qualification and mutation results are in [validation evidence](../docs/validation-evidence.md).
- `just test-transfer` checks source binding, artifact integrity, safe extraction,
  and the pinned rustdoc compile-and-run contract, including deliberate failures.
  It needs the repository-pinned nightly and runs a small Rust fixture.

### Cross-built tests

RISC-V, POWER, and IBM Z CI build on Ubuntu x86-64 with the `--ci-cross-build TARGET` tooling.

CI, CT, Bench, and Profile share two preparation actions:

- [cross-build/setup](../.github/actions/cross-build/setup/action.yml) restores the target's APT state, runs the pinned cross-build installer,
  and saves verified state after the last package installer.
  CI selects compiler-cache authority afterward.
  The other workflows keep compiler caching disabled.
- [cross-build/upload](../.github/actions/cross-build/upload/action.yml) requires both the prepared suite and its runner-tools archive
  before uploading the target's existing artifact name for two days.
  Diagnostic logs retain their separate failure handling and retention periods.

Each workflow still owns its preparation command, runner selection, timeout, native execution,
and fail-fast matrix.
The actions do not select tests or qualify targets.
`test/cross_workflow_test.py` exercises the action shell steps, including installation failure and either missing archive,
and checks their caller/consumer wiring through `just test-scripts`.
The transfer regressions in `just test-transfer` remain responsible for source, compiler, target, suite,
and file-integrity checks.

The 2026-10-05 consolidation passed `just test-scripts`, all 12 `just test-transfer` tests, the seven CT planner tests, `actionlint`,
and ShellCheck on both action shell steps.
Four existing installer tests needed Linux/APT/sudo or PowerShell and were skipped on macOS.
Comparison with the initial worktree confirmed unchanged native jobs, planners, runner matrices,
timeouts, and workflow-level settings.
No hosted run of the uncommitted actions exists yet;
these local results do not claim native qualification.

Test preparation and native execution retain these contracts:

- The build runs the same target-specific checks.
  It builds all release tests in both dispatch modes, including all doctest compilation checks.
- Nextest archives and persisted doctest programs move to the matching native runner.
  Its `--ci-cross-run TOOLS_ARCHIVE` tooling only runs them.
- The test bundle uses XZ compression to reuse repeated code across standalone doctest programs.
  Its executable bytes, modes, and sealed hashes stay unchanged; existing gzip bundles remain readable.
- Preparation is not a runtime pass.
- The Rust release profile, target compiler, feature sets, and test assertions do not change.
- Profile CI uses the same exact-artifact transfer boundary for x86-64, AArch64, RISC-V, POWER,
  and IBM Z.

Each preparation job also cross-builds the pinned `just`
and Nextest tools into a separate archive bound to the source.

- The native bootstrap verifies that archive, and the ELF architecture of each tool,
  before it adds the tool directory to `PATH`.
- No Cargo tools compile on the execution runner.
- Both Nextest builds use the same locked crate release.
  The producer and consumer identities must match, except for the host architecture.
- GNU cross-compilers and target libc development packages
  use the same Ubuntu CI snapshot as native provisioning.
- The x86-64 preparation jobs use the shared Cargo Rail compiler cache.
  Trusted `main` pushes can populate it.
  Other jobs with credentials are read-only, and pull requests from forks build cold.
- The native execution jobs enable no compiler cache,
  because they run the sealed programs without compiling the crate.

Cross-builds run dependency build scripts and procedural macros on x86-64.
They keep the target runtime evidence, but they do not qualify those tools as native POWER, IBM Z,
or RISC-V host programs.

The archive records the Git revision, the effective source digest, the compiler, Nextest,
the release settings, and the digest and executable bit of every file.
Execution rejects different sources, missing or changed files, inherited selection overrides,
and the wrong host architecture.
The source checkout supplies fixtures, and it must match the build checkout,
including untracked source files.
CI downloads only the named artifact from the current workflow run.
Artifacts are not shared caches.
The build runner and the artifact service stay trusted.
Digests detect corruption and mismatches.
They do not detect a compromised producer that forges its own metadata.

Doctests use rustdoc's extraction inventory and compilation checks, and keep `compile_fail`, error-code checks, `no_run`,
and `should_panic`.
Transfer preparation disables merging,
because the merged runner of the pinned rustdoc runs even with global `--no-run`.
Each runnable standalone program must then run on the target hardware.
Ordinary `just test` doctests keep rustdoc's default merging behavior.

### Examples, Miri, and fuzzing

- Example names and feature requirements come from Cargo metadata.
- Use `just test-miri --rsa` for the focused RSA scope.
- Use `just test-fuzz --targets A,B` for an explicit group of fuzz targets.
- Package scope is selected before targets.
  `--all` includes matching targets from the full and scoped packages.
  `--full` and `--scoped` limit the scope.
  The default is full, including named targets.
- `just test-fuzz --build --all` builds every fuzz package without starting fuzzing.

## Constant-time evidence

| Script                 | Caller |
| ---------------------- | ------ |
| `ct/zig-cc.sh`         | `ct/artifacts.sh`, `ct/binsec.py` |
| `ct/test.sh`           | `just ct-test` |
| `ct/artifacts.sh`      | `just ct-artifacts`, `ct/full.py` |
| `ct/dudect.sh`         | `just ct-dudect`, `ct/full.py` (preparation only) |
| `ct/dudect_execute.py` | `ct/dudect.sh`, `ct/full.py` |

- `ct/full.py`, `ct/binsec.py`, and `ct/validate.py` implement `just ct-full`, `just ct-binsec`,
  and `just ct-validate`.
- `ct/manifest.py` owns the shared target and measurement selection.
- `ct/provenance.py` owns the shared file hashing and build identity.
- The other Python files under `ct/` implement local artifact provenance, disassembly analysis,
  report parsing, and their focused regression tests.

`just ct-dudect --smoke` uses each selected case's `smoke_samples` from `ct.toml`.
`--samples` overrides the environment sample setting,
and the environment setting overrides the manifest smoke budgets.
Each smoke case keeps its own measurements.
The latest report summarizes the selected cases and their requested budgets.

### Transferred CT

RISC-V, POWER, and IBM Z use `just ct-full --target TARGET --prepare-archive ARCHIVE` on the x86-64 build host, and `--run-archive ARCHIVE` on the matching native hardware.

- Preparation keeps strict API and artifact validation, the generated-code checks,
  and the cleanup sentinel.
  It also compiles and disassembles the exact DudeCT executable that will be timed.
- The consumer verifies the source and the artifacts.
  It then runs the full manifest campaign, with the same sampling, threshold, and per-case timeouts.
- No target code is rebuilt during measurement.
- Reports separate the build host from the measurement host,
  and keep the original preparation bundle.
- Only the three targets in `lib/cross_build.py` can use this transfer mode.
  It cannot bypass native BINSEC on targets that need it.

### DudeCT runs

DudeCT prepares one binary, disassembly, symbol map, linker log,
and provenance snapshot for each invocation, under `target/ct/<target>/<profile>/dudect/runs/<run>/shared/`.

- The bundle is read-only after preparation.
- Each `ct-full` case runs that binary and writes its own CSV, stdout, and report under `cases/<manifest-name>-<unique>/`.
  Reports reference the shared files and their hashes; they do not copy them.
- `ct-full` confirms a required case before it decides it when the screening |t| exceeds the threshold
  or reaches `review_fraction` of it. The confirmation reruns the same binary with `sample_factor`
  times the samples, in its own case directory. `[dudect_confirmation]` in `ct.toml` sets both values,
  and [the constant-time policy](../docs/constant-time.md#dudect-decision) explains the decision.
- Standalone selection evidence is under the same run's `selection/` directory,
  with a small latest report at `dudect/dudect-report.json`.
- A failed preparation or execution cannot reuse the measurements of an earlier run.
- Historical runs stay on disk until you remove them.
  Full reports list only their current run.

### Replay

`just ct-replay --target TARGET --source-root SOURCE --archive ARCHIVE --out OUTPUT --case CASE` repeats prepared POWER, IBM Z, or RISC-V cases three times on one allowed CPU.

- `--case` takes an exact manifest case, `mldsa` for every required ML-DSA kernel case,
  or `mldsa-probe` for the ML-DSA diagnostic probes.
- The suite lists all planned cases, keeps each case's original budget and every result,
  and fails if any required case fails.
  A tooling failure leaves the inventory explicitly incomplete.
- `--repetitions 1` measures each case once, at the same sample count.
- The target defaults to RISC-V for existing callers.
  It must match both the archive and the physical runner.
- Replay validates the original source and the transferred binary,
  keeps the manifest sample count and timeout, and keeps all results at threshold 10.
- Timing failures do not shorten the planned campaign.
  Execution failures do.
- Host snapshots record affinity, frequency settings where the host exposes them, load,
  and processes.
  They do not guarantee an otherwise idle machine.

Replay is diagnostic evidence, not full qualification.

### Diagnostic CT runs

The CT workflow's `diagnostic_case` input takes exact case names, a comma-separated list,
or the `mldsa` and `mldsa-probe` groups, on any requested Linux architecture.
Native rows run `ct-full --dudect-case`.
Cross rows (POWER, IBM Z, RISC-V) verify the complete prepared archive and replay the cases once.
These runs provide scoped evidence and do not qualify a release.

For a P-384 public-key diagnostic on RISC-V, set `architectures` to `riscv64-linux`
and `diagnostic_case` to `ecdsa_p384_public_key_fixed_vs_random_secret`.

The planner rejects unknown cases and unsupported targets before it schedules builds.
Release callers, and manual runs with no diagnostic input, keep the full required lane.

## Secret stack evidence

| Script             | Caller |
| ------------------ | ------ |
| `stack/frames.py`  | `just stack-frames [--target TARGET]... [--portable] [--rustflags FLAGS] [--json PATH]` |
| `stack/residue.py` | `just stack-residue [--board BOARD]... [--backend native\|portable]... [--output DIR]` |

### Frame review

`stack/frames.py` reviews each scrubbed secret-worker boundary in a linked release binary.

- It builds `tools/frame-review` with `-Z emit-stack-sizes` and the shipping release profile.
  The binary runs every ML-DSA and ML-KEM public operation once.
- Linux targets other than the host link through `ct/zig-cc.sh`.
- Frame sizes come from the compiler's `.stack_sizes` records.
  Calls come from the linked disassembly, resolved by address.

The review fails when:

- a worker's depth below its caller is larger than the scrub buffer;
- a worker reaches capability detection;
- a caller of a worker does not also call the scrub;
- a path cannot be bounded: a function without a frame record, an indirect transfer,
  an unresolved import, or recursion.
  These paths are reported, never dropped.

Assumptions and limits:

- libc `memcpy`, `memmove`, `memset`, `memcmp`, and `bcmp` count as frameless leaves plus the target red zone.
  Each report lists this assumption.
- Panic exits are listed but not counted, because the artifact uses `panic = "abort"`.
- The tool does not check where the buffer sits inside the scrub frame.
  It checks only that the scrub's extent can hold the buffer.
- The review covers x86-64, AArch64, POWER, IBM Z, and RISC-V 64 Linux ELF binaries.
- It is static build evidence for the reviewed compiler, target, and features.
  It does not run the binary.

`stack/frames_test.py` (part of `just test-scripts`) checks the parsing and failure rules on synthetic disassembly for each architecture.

### Residue measurement

`stack/residue.py` measures the moved-copy residue of secret owners.

1. It builds `tools/residue-harness` for RV32 (`riscv32imac-unknown-none-elf`)
   and Cortex-M3 (`thumbv6m-none-eabi`), with native and `portable-only` backends.
1. It boots each build in QEMU on the `virt` and `mps2-an385` boards.
1. Each scenario runs once, on a painted stack and painted allocator arenas.
1. When the scenario returns, the harness copies the dead stack and both arenas to a snapshot,
   with loops that make no calls.
   Only then does it derive the scenario's secret byte strings again.
1. The host reports how many bytes of each secret occur in 16-byte windows in each region,
   and how many whole copies remain.

Pass and fail rules:

- A scenario marked `none` fails on any secret byte found.
- Two controls must find a planted secret in full:
  one in a returned frame, and one in a freed allocation.
- A third control clears its copy before it returns, and must find nothing.
- A panic, a missing scenario, or a truncated log also fails.

`--output DIR` keeps the raw UART logs and `report.json`.
QEMU runs show what the compiled code leaves in memory under emulation.
They are not device timing evidence or device stack evidence.
`stack/residue_test.py` (part of `just test-scripts`) checks the parsing and the judgement on synthetic logs.

## Benchmarks and updates

| Script                | Caller |
| --------------------- | ------ |
| `bench/runner.py`     | `just bench`, `just profile`, `just bench-export`, code inspection recipes |
| `bench/execution.py`  | Shared Cargo build, discovery, and provenance for measurement and profiling |
| `bench/measure.py`    | Runner: measurement and completion verification |
| `bench/file_cache.py` | Optional BLAKE3 file harness: file-scoped Linux cache conditioning and residency evidence |
| `bench/profile.py`    | Runner: exact-case local Samply capture, or transferred native `perf` capture |
| `bench/profile_ci.py` | Validation and dispatch of manual CI profile requests |
| `bench/evidence.py`   | Shared build and runtime environment collector |
| `bench/settings.py`   | Measurement, profiling, and watchdog settings |
| `bench/transfer.py`   | Compile-only preparation and verified native consumption for RISC-V, POWER, and IBM Z |
| `bench/bounded.py`    | `just bench`, `just profile`: process-tree deadline |
| `update-all.sh`       | `just update` |

- `bench/benchmark_catalog.py` owns algorithm, target, and curated profile-preset selection.
- `.config/criterion.json` owns the shared Criterion defaults and the maximum run budget.
- `benches/common/criterion.rs` applies them to every Criterion harness.
  `bench/settings.py` resolves overrides for one invocation.
- `bench/bounded.py` stops the whole benchmark or profile process tree within the budget.
- `bench/runner.py` resolves filters to unique cases.
- `bench/measure.py` runs one process for each configuration,
  and verifies the statistical artifacts before the runner marks a run complete.
- Export is a separate runner command.
- `just bench --list` uses the same resolver to list the actual cases with their catalog work classes,
  without starting a measurement run.
  `--diag` enables diagnostic cases.

`bench/benchmark_catalog_test.py` includes the benchmark runner and profiling regression tests.
Run that focused suite with `scripts/lib/python.sh scripts/bench/benchmark_catalog_test.py`.
It tests the Python runner, Just argument forwarding, and export in temporary directories,
with substitute Cargo, benchmark, and profiler executors.
It does not run cryptographic benchmarks.

Local and development-machine benchmarks share unique run directories under `benchmark_results/criterion/<run-id>/`, with logs, the plan,
provenance, raw data, and completion status.
Explicit exports and checksums are in `benchmark_results/.transfers/`.
See [benchmarking](../docs/benchmarking.md) for explicit baseline comparisons and remote collection.

Cargo Rail planning supplies the affected scope for `just test`.
Check, Miri, and fuzz commands run independently of that plan.

## Shared libraries

| Script                                 | Sourced or called by |
| -------------------------------------- | -------------------- |
| `lib/rail-plan.sh`                     | `test/test.sh`       |
| `lib/fuzz-packages.sh`                 | Fuzz scripts         |
| `lib/python.sh`                        | Python-based check, test, CT, and benchmark scripts |
| `lib/toolchain.py`, `lib/toolchain.sh` | Shared toolchain selection for installers, builds, checks, tests, and benchmarks |
| `tooling/transfer.py`                  | Cross-build and verify the pinned native runner tools |
| `tooling/apt_state.py`                 | Verify restored APT indexes against the pinned Ubuntu snapshot before `tooling/linux.sh` uses them |
| `lib/evidence_bundle.py`               | Source binding, sealing, and transfer integrity for cross-compiled tests, tools, and CT |
| `lib/cross_build.py`                   | Explicit target identities and the cross-compiler environment for test, tool, and CT preparation |

Python tooling needs Python 3.11 or newer.
The updater installs its catalog-pinned Python libraries into a temporary virtual environment.
Checks and benchmarks use only the standard library.

## Native tooling

### Runner profiles

[`.github/runs-on.yml`](../.github/runs-on.yml) defines the AWS runner profiles.
Change CPU, instance families, images, storage, and Spot policy there.
Workflows refer to profile names, and do not override their shapes.

- CI, CT, and benchmark cross-builds have separate profiles.
  So do native CI, fuzzing, and CT measurement.
- Ordinary jobs use price-capacity-optimized Spot.
  CT and benchmark measurement use separate fixed On-Demand profiles.
- Cargo Rail cache setup and reporting are enabled only for native x86-64
  and Arm64 Linux compilation, and for the three x86-64 cross-build producers.
  Execution-only target jobs, and the CT, fuzz, benchmark, and profile workflows, stay cold.
- For this public repository, RunsOn reads the catalog from the default branch.
  A new profile must be on the default branch before workflow jobs can resolve its name.

#### Local compiler-cache evidence, 2026-10-05

On the local Apple Silicon checkout,
cargo-rail 0.30.1 initially reported drifted compiler components, no workspace enrollment,
and zero recorded events.
`just rail-cache-setup --max-size 10GiB`, with remote-cache environment settings removed, repaired the installation and passed its uncached,
cold-miss, and verified warm-hit checks.
The resulting profile uses physical-root identity, with no remote authority configured.

Two production builds ran `cargo check --locked --lib --no-default-features --features blake3` with the same dedicated `--target-dir`,
deleting only that probe's Cargo outputs before each build.
Both passed.
The first added one miss; the second added one hit.
Neither added a bypass or failure.
The selected profile's hit/miss/event counters were `1/1/2` after setup, `1/2/3` after the first build,
and `2/2/4` after the second.
This verifies private local compiler reuse for that library configuration.
It does not establish shared R2 reuse, registry/native-library coverage, target execution, timing,
or CT evidence.
Workflow cache coverage is unchanged until the relevant shared profiles demonstrate hits.

The probe used the dirty effective source based on `d4045559`, host `aarch64-apple-darwin`, and `nightly-2026-09-30` (`rustc 1.101.0-nightly`, commit `5c543b0b8c73c7b72bc8284ced4fb22ead15734d`).
The SHA-256 of the sorted compact JSON mapping of `src/`, `.cargo/`, `Cargo.toml`, `Cargo.lock`,
and `rust-toolchain.toml` paths to their file SHA-256 values was `f2a6db50d362dd32ec7cfc3b6e40fab14d0045303a0f7a3feb5ebb27c9b9ff0c`.
Raw commands, source hashes, build logs, and cache reports remain locally in `target/workflow-cleanup-20261005/cache/`;
these ignored artifacts are not distributed.

### Updates

`just update` refreshes the tooling catalog, stable Rust, every Cargo manifest
(including the standalone and fuzz support workspaces),
the lockfiles, and the existing GitHub Action pins.
It runs on local macOS, and it has no dependency publish-age filter.
Inspect its changes before you commit.

`rust-toolchain.toml` pins the one canonical nightly for every host, target, and tool lane, including formatting, Miri,
fuzzing, coverage, and CT evidence.
`lib/toolchain.py` reads that pin.
Build, check, test, and benchmark entry points use it, not an ambient `RUSTUP_TOOLCHAIN`.
`just update` pins the newest nightly that is complete for every catalog host, component, and target.

The MSRV lane uses `rust-version` from `Cargo.toml`.
While that Rust release is unpublished,
a canonical nightly of the same release runs the MSRV lane itself.
When the canonical nightly is one release ahead, the MSRV is in beta,
and the lane runs on the exact beta in `MSRV_PREVIEW` (`lib/toolchain.py`).

### Installers

Run `scripts/tooling/<platform>.sh` on the native Ubuntu version pinned in [the catalog](../.config/tooling.toml).
The platforms are `aarch64-linux`, `x86_64-linux`, `riscv64-linux`, `s390x-linux`, and `powerpc64le-linux`.
The installers use `sudo` when needed.
On Windows, use `aarch64-win.ps1` or `x86_64-win.ps1` in an elevated PowerShell session.
Local macOS tools are managed locally.
`scripts/tooling/aarch64-macos.sh` can install the pinned prerequisites for `just check-macos`.

All full profiles install the prerequisites for `just ci-check`, `just test`, and Criterion `just bench`.
RISC-V, IBM Z, and POWER use native CMake and Clang pinned to the snapshot.
They do not install cross targets, Miri, browsers, or profiling tools.

Only x86-64 and ARM64 Linux development setup installs `perf`, Valgrind, Gungraun, and samply.
That installer enables perf events, and it needs `perf` for the running kernel.
Use `just bench-structural` for Gungraun and `just profile` for samply.
Criterion benchmarks are available on every native platform.
Provisioning checks the tools, but you must still verify native test, benchmark,
and profiling execution on each machine.

### CI provisioning

CI calls the same installers with `--ci` on Linux or `-Ci` on Windows.

- The catalog's `ci` section selects the Cargo tools that `just ci-check`, `just test --all --release`, and `just test --all --release --portable` need.
  Both test commands include doctests.
- Only Linux x86-64 adds the `ci-policy` tools and runs `just ci-policy`.
  Cargo Deny checks the full target graph in `deny.toml`, and Cargo Audit checks the lockfile.
- Every host keeps native and portable Clippy, independent-workspace linting, documentation,
  and runtime tests.
  RISC-V runs its compilation checks on the cross-build host, and runs the tests on native hardware.
- Linux CI omits OpenSSL development packages, pkgconf, and recommended APT packages.
  CMake, Clang/libclang, Perl, and the C/C++ build tools stay,
  because native test dependencies need them.
- This mode omits Cargo Rail, because `--all` bypasses affected-work planning.
  Use the full installer for ordinary `just test` and benchmark work.
- CI does not install optional profiling, mutation, or live-fuzzing tools,
  and it does not change shell startup files.
  These jobs validate CI provisioning, not the full optional development toolset.
- Cargo Binstall selects compatible binaries on x86-64, ARM64, and RISC-V,
  and falls back to source when no binary exists.
  IBM Z and POWER build Cargo tools from source.

After Linux installation, source `$HOME/.local/share/rscrypto-tooling/environment.sh` in each new CI step.
Windows CI runs installation and validation in one PowerShell step,
to keep the MSVC and SDK environment.
Windows x86-64 installs catalog-pinned NASM for native dependency assembly in both modes.

### APT snapshot

Linux CI uses the catalog's `linux-ci` Ubuntu release,
with packages from the same archive snapshot as development provisioning.

- Only that signed snapshot supplies package indexes and version choices.
- If its package endpoint fails, APT can fetch the exact package from Ubuntu's live archive.
  The snapshot's authenticated package checksum still applies.
  The live archive cannot supply indexes or change the selected versions.
- The installer keeps APT's indexes and downloaded packages in `/var/cache/rscrypto-apt`
  (`RSCRYPTO_APT_STATE` overrides it).
- CI restores that directory with [`.github/actions/apt-state`](../.github/actions/apt-state/action.yml),
  and calls [`apt-state/save`](../.github/actions/apt-state/save/action.yml) after the last package installer.
  Cross-build jobs save inside setup; native jobs save as their final step, including failure paths.
  It saves only when the installer's `ready` marker shows that this run verified the state
  and installed from it.

APT trusts any index that is already in its lists directory.
`tooling/apt_state.py` therefore proves a restored state first:
each suite's InRelease must match the catalog's `inrelease` SHA-256 pin and Ubuntu's archive signature,
and every index must carry its signed size and SHA-256.
A verified state installs without contacting the snapshot service.
APT still rejects any cached package that differs from those indexes.
Otherwise, the installer discards the indexes, fetches them again, and requires the same proof.
`just update` records the pins each time it selects a snapshot.

### Workflows

**`ci.yml`** also runs `--ci-package` provisioning and `just ci-package` on an independent runner.
It runs the examples, verifies the publishable Cargo archive, and runs external `std`, `core`,
and `alloc` consumers against the unpacked crate on the canonical and MSRV toolchains.
`core` and `alloc` also compile on the existing Thumb sentinel.
No package is published.

**`fuzz.yml`** uses `--ci-fuzz` for committed ASan corpus replay and bounded live fuzzing.

- Manual runs select x86-64, ARM64, or both, exact target names, and a duration for each target.
- Pull-request campaigns use 60 seconds for each target.
  Manual campaigns default to 120 seconds.
- Both have a planned live-fuzzing budget of 30 minutes for each architecture,
  for eight concurrent targets.
  The selection must fit its budget before replay starts.
- Manual fuzz jobs have a 90-minute limit, including installation, builds, corpus replay,
  and live fuzzing.
  Pull-request and Miri jobs keep their 60-minute limit.
- `--ci-miri` installs the pinned interpreter for an independent focused Miri row,
  including the RSA unsafe-boundary tests.
- All rows share fail-fast cancellation.

**`ct.yml`** always runs full CT evidence, only through manual dispatch or a reusable workflow call.
It does not run on pull requests or pushes.

- Manual runs select one, several, or all six native platforms.
  The default is all.
- The release workflow calls it for all platforms,
  and needs it to pass on the same candidate before it publishes.
- Linux uses `--ci-ct-full`, and Windows uses `-CiCt`.
- On a native GNU Linux x86-64 or ARM64 development host,
  `scripts/tooling/x86_64-linux.sh --ct-proof` or `scripts/tooling/aarch64-linux.sh --ct-proof` adds the same pinned proof tools to the native tool catalog.
  The `--ci-*` modes retain their separate operating-system checks.
- On GNU Linux x86-64 and ARM64, the Linux installer also installs the pinned BINSEC, Bitwuzla,
  and the decoder.
  Proof dependencies use a fixed opam repository revision from `.config/tooling.toml`.
- Targets without proof support keep their explicit `ct.toml` policies.
  No solver is installed there.
- CT architectures run at the same time, on fixed AWS instances or donated native runners.
  Each host completes builds and proofs before the serial timing cases.
- `just ct-full` uses the cases and budgets that the manifest requires, without filtering.
- RSA timing lives in this one harness, including entropy-backed signing.
  Its consolidated operation cases keep 2,000 observations per class and a threshold of 8.
- Proof failures stop timing.
  A failed required timing case stops the later cases.
- Local `just ct-dudect --smoke` is a diagnostic shortcut outside this workflow.
- Full CT evidence does not establish the complete secret-lifecycle claim by itself.

The fuzz and CT workflows keep the final evidence for seven days, and run without caches.
CT preparation archives are kept for two days.
The CT, benchmark, and profile selection jobs validate requests
and emit only the requested runner rows.
They do not install Rust, build code, or call Cargo Rail.

**`bench.yml`** runs only on manual request.

- It selects one, several, or all six native CI platforms,
  and catalog algorithms, groups, or benchmark targets, with optional case filters.
- A small planner starts only the selected runners.
- The existing benchmark runner owns measurement and evidence.
- `--ci-bench` (Linux) and `-CiBench` (Windows) install the native benchmark build prerequisites and Just, without test,
  profiling, or cross-target tools.

See [Benchmarking](../docs/benchmarking.md#bench-workflow).

**`profile.yml`** runs only on manual request.
It accepts one native Linux architecture and one curated primitive.

- `.config/benchmark-matrix.json` maps each primitive to one benchmark target and one exact production case.
  It also owns the diagnostic-feature policy.
- The workflow records that case for five seconds.
- Preparation and native capture each have their own 20-minute cap.
- A newer request for the same architecture cancels the older request.
- It reuses the benchmark cross-build and sealed transfer path.
  The native runner verifies and discovers the transferred executable, then runs `perf record` and `perf report --stdio --no-inline`.
- The job summary shows the report.
  The artifact keeps the raw capture.
  The native runner never rebuilds production code.
- Native setup prefers the runner's `perf`, then a matching Ubuntu package.
  The pinned RISC-V kernel has no matching Ubuntu tools package,
  so setup builds `perf` from the matching pinned upstream stable source.
  Other donated runners fall back to the pinned Ubuntu generic userspace tool.
- Live probes decide whether the result is usable.
  When runner policy denies the unprivileged capture, native setup enables perf events.
- Setup, missing-package, denied-permission, and collector failures are kept as evidence.
  Preparation and native result artifacts are kept, also when collection fails.

### CI compatibility

The compatibility matrix row starts with every native row, and uses the same fail-fast policy.

- `x86_64-linux.sh --ci-compat` installs only the catalog-selected compatibility tools,
  Rust versions, and cross-target libraries.
- `just ci-compat` uses bounded workers with separate build directories and a shared CPU budget.
  A failed command stops the running siblings and stops queued work from starting.
  Logs stay under `target/compat/`.

The compatibility checks cover:

- each Cargo feature alone, on the development compiler and on the declared minimum Rust version;
- broad native and portable feature sets;
- allocation-free and allocation-enabled Thumb sentinels;
- MSRV `portable-only` builds on POWER, IBM Z, RV64, and RV32, including core-only,
  allocation-enabled, and release code generation;
- a release library build for every supported bare-metal target.
  Bare-metal evidence is compile-only, not device execution.

Bare WASM and WASI both compile and run the existing runtime vector harness in Wasmtime,
with scalar and SIMD artifacts tested separately.

- The scalar module must load with SIMD disabled.
- Bare WASM calls an explicit export with no arguments.
  WASI uses its command entry point.
- These are Wasmtime results, not browser-engine results.
- The library also gets broad feature builds for both WASM targets.

The x86-64 and ARM64 Linux rows install the native musl build prerequisites and run `just test-musl`:
the complete native and portable test suites, plus doctests and separate internal evidence suites,
compiled and run for the matching musl target.
Apple ARM64 checks and tests run locally through `just check-macos` in a background job queued by each push.
Windows ARM64 execution is deferred.
No compatibility lane enables persistent caches.

## Internal evidence builds

`scripts/ct/internal.py` enables `--cfg rscrypto_internal` for repository evidence builds.

- CT artifact generation, BINSEC, DudeCT, their self-tests,
  and their independent-workspace lint checks use it automatically.
- Diagnostic benchmarks, profiles, and code inspection use the same flag resolver.
- The RSA assembly gates use it for their public-operation candidate tests.
- The resolver keeps the target compiler flags and passes encoded arguments to Cargo.
  Build provenance records the effective flags.
- Normal builds, docs, tests, and published Cargo feature combinations do not opt in.

`just test-evidence` runs the production library and evidence integration tests,
with production-auto and portable-only dispatch.
It keeps the forced-kernel, component,
and PBKDF2 verification regressions after their hooks leave the public API.
`just ct-test` and native qualification include this recipe.
Cross-test archives carry separate production-auto and portable-only internal suites,
and need all four suites at execution.
Run it with ordinary tests when you change evidence hooks.
Ordinary tests continue to check the application build without internal access.

- Use `just bench <selector> --diag` or `just profile <target> --diag` for diagnostic workloads.
- The `aead-diag` selector and `--bench aead_kernels` enable their required internal hooks automatically.
- Cross-prepared diagnostic benchmarks resolve the flags for the destination target
  before they record build provenance.
- Keep internal builds separate from public-surface checks.
  Use `just ct-binsec` for binary proofs and `just ct-dudect` for timing evidence.

## Release orchestration

`.github/workflows/release.yml` calls CI, CT, and fuzz qualification before its publication job.
`scripts/release/release.py` validates the candidate, reconciles registry checksums on retries,
and creates the source tag and the GitHub Release.
Preflight, packaging, and publication each require the newest `rscrypto/macos` commit status to pass.
Its description binds the Git tree and a SHA-256 digest of `rustc -vV`, excluding only the host line
so the Linux release runner can compare the same compiler distribution.
An older successful status cannot override a newer failure or compiler mismatch.
The workflow can read statuses but cannot write them.
Its failure and recovery tests run through `just test-scripts`.
[CONTRIBUTING.md](../CONTRIBUTING.md#release) has the maintainer setup, preparation, deployment, and retry instructions.

`scripts/check/macos.sh` owns `just check-macos`, which replaces hosted macOS checks and tests with local Apple Silicon validation.
Install `.githooks` with `just install-hooks` in each maintainer checkout. macOS is still a supported release target.
Timing qualification on physical Apple Silicon is a separate local requirement before submission.

`scripts/check/macos_async.py` queues a detached worktree of the pushed commit and returns immediately.
It runs the unchanged `check-macos` suite, reuses `ci-check` and inert-only ancestor passes through
`qualified.py`, and records a pass only if the source and compiler still match after testing.
Jobs share the repository's common Git directory at `rscrypto-macos/`:

- Each commit/compiler/destination job retains `state.json`, the runner from that commit, and an append-only `log`.
  `just macos-status` reports results, interrupted jobs, and log paths.
- An inherited file lock prevents duplicate jobs. A separate build lock serializes qualification.
  Jobs create their source worktrees after acquiring that lock, so source timestamps cannot make
  Cargo reuse an older checkout's executable. Cargo shares compatible artifacts in `rscrypto-macos/target/`;
  native, portable, internal, debug, and release configurations remain distinct.
- The worktree is removed after qualification. The log retains the RSA assembly gate's output.
  The tree/compiler pass remains in `rscrypto-macos-qualified/` for reuse.
- For a GitHub destination, `gh api` writes the `rscrypto/macos` status for the pushed SHA.
  It briefly retries a commit that is not visible yet because the push is still transferring it.
  Local destinations retain local evidence only. No evidence lane moves to CI.
- `just qualify-macos` retries the clean current commit against `origin`.
  Failed status publication retains the local pass, so retrying publication does not repeat testing.
  Missing authentication, failed checks, interrupted jobs, and failed publication leave release blocked.

The status uses the maintainer's existing repository write authority, not a separate signing or attestation service.
The [GitHub status API](https://docs.github.com/en/rest/commits/statuses) owns permission and response semantics.

### Exact-commit CI reuse decision, 2026-10-05

Keep the release's full call to `ci.yml`.
Reusing an earlier successful run by commit SHA alone does not preserve the current release
qualification contract.
`check/dependencies.sh` runs `cargo deny` and `cargo audit`; advisory and registry state can change
after the push run without changing the source commit.
Release CI currently checks that state again.

A separate run verifier would also have to bind the repository, workflow, source,
compiler and tooling to a complete successful attempt,
reject missing or skipped required matrix jobs and later failed attempts,
and retain the required execution evidence.
Source identity alone does not establish those facts.
The current workflow graph already requires CI, full CT,
and fuzz/Miri qualification before publication,
and the separate Mac gate binds its latest status to the source tree and compiler.

Retaining the existing graph is the smallest complete choice.
Reuse with a full qualification receipt
and a fresh dependency-policy check remains a possible future design;
this review does not introduce another implementation obligation or weaken any current gate.
