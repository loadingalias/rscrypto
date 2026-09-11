# Scripts

Repository scripts implement local development, testing, evidence, and
benchmark commands. User-facing entry points are the recipes reported by
`just --list`. Every script in this directory is locally executable.

## Check entry points

| Script | Caller |
| --- | --- |
| `check/check.sh` | `just check`, `just ci-check`, `just ci-check-target` |
| `check/compat.py` | `just ci-compat` |
| `test/test-musl.sh` | `just test-musl` |
| `check/dependencies.sh` | `just ci-policy`, dependency checks within `just check` |
| `check/lint-independent-workspaces.sh` | `check/check.sh` |

`check/check_runner_test.py` tests command selection, repair behavior, and
failure propagation with substitute executors. Run it with
`scripts/lib/python.sh scripts/check/check_runner_test.py`.

## Test entry points

| Script | Caller |
| --- | --- |
| `test/test.sh` | `just test` |
| `test/riscv.py` | `just test-riscv prepare ARCHIVE`, `just test-riscv run ARCHIVE` |
| `test/doctest_bundle.py` | RISC-V doctest compilation and target execution |
| `test/test-examples.sh` | `just test-examples` |
| `test/test-miri.sh` | `just test-miri` |
| `test/test-fuzz.sh` | `just test-fuzz` |
| `test/test-fuzz-asan.sh` | `just test-fuzz-asan` |
| `test/test-coverage.py` | `just test-coverage` |
| `test/test-rsa-asm.sh` | `just test-rsa-linux-asm`, `just test-rsa-macos-asm` |

`just test-scripts` runs argument forwarding, toolchain, test, check, and fuzz regressions with
substitute executors. `just test-harnesses` directly runs the CT harness and
exporter self-tests; `just ct-test` includes them.

`just test-transfer` checks source binding, artifact integrity, safe extraction,
and the pinned rustdoc compile/run contract, including deliberate failures.
It requires the repository-pinned nightly and runs a small Rust fixture.

RISC-V CI builds on Ubuntu x86-64 using `--ci-riscv-build` tooling. It runs the
same target-specific checks and builds all release tests in both dispatch modes,
including all doctest compilation checks. Nextest archives and persisted doctest programs
are transferred to the physical RISC-V runner, whose `--ci-riscv-run` tooling
only executes them. Preparation is not a runtime pass. The Rust release profile,
target compiler, feature sets, and test assertions remain unchanged.

The archive records the Git revision, effective source digest, compiler, Nextest,
release settings, and every file's digest and executable bit. Execution rejects
different sources, missing or changed files, inherited selection overrides, and
the wrong host architecture. The source checkout supplies fixtures and must
match the build checkout, including untracked source files. CI downloads only
the named artifact from the current workflow run; artifacts are not shared caches.
The build runner and artifact service remain trusted. Digests detect corruption
and mismatches, not a compromised producer that forges its own metadata.

Doctests use rustdoc's extraction inventory and compilation checks, preserving
`compile_fail`, error-code checks, `no_run`, and `should_panic`. Transfer preparation
disables merging because the pinned rustdoc's merged runner executes despite
global `--no-run`. Each runnable standalone program must subsequently execute on
RISC-V. Ordinary `just test` doctests retain rustdoc's default merging behavior.

Example names and feature requirements come from Cargo metadata. Use
`just test-miri --rsa` for the focused RSA scope and `just test-fuzz --targets A,B`
for an explicit fuzz target group. Package scope is selected before targets:
`--all` includes matching targets from full and scoped packages; `--full` and
`--scoped` restrict it. The default is full, including named targets.
`just test-fuzz --build --all` builds every fuzz package without starting fuzzing.

## Constant-time evidence

`just ct-dudect --smoke` uses each selected case's `smoke_samples` from
`ct.toml`. `--samples` overrides the environment sample setting, which overrides
manifest smoke budgets. Each smoke case retains its own measurements; the
latest report summarizes the selected cases and their requested budgets.

| Script | Caller |
| --- | --- |
| `ct/zig-cc.sh` | `ct/artifacts.sh`, `ct/binsec.py` |
| `ct/test.sh` | `just ct-test` |
| `ct/artifacts.sh` | `just ct-artifacts`, `ct/full.py` |
| `ct/dudect.sh` | `just ct-dudect`, `ct/full.py` (preparation only) |
| `ct/dudect_execute.py` | `ct/dudect.sh`, `ct/full.py` |

`ct/manifest.py` owns shared target and measurement selection.
`ct/provenance.py` owns shared file hashing and build identity.

RISC-V CT uses `just ct-full --target riscv64gc-unknown-linux-gnu --prepare-archive ARCHIVE`
on the x86-64 build host and the corresponding `--run-archive ARCHIVE` on physical
RISC-V. Preparation retains strict API/artifact validation, generated-code checks,
and the cleanup sentinel. It also compiles and disassembles the exact DudeCT
executable that will be timed. The consumer verifies the source and artifacts,
then runs the existing full manifest campaign with unchanged sampling, threshold,
and per-case timeouts. No target code is rebuilt during measurement. Reports
distinguish build and measurement hosts and retain the original preparation bundle.
This transfer mode is restricted to RISC-V; it cannot bypass native BINSEC on
targets that require it.

`ct/full.py`, `ct/binsec.py`, and `ct/validate.py` back `just ct-full`,
`just ct-binsec`, and `just ct-validate`. The remaining Python files under
`ct/` implement local artifact provenance, disassembly analysis, report
parsing, and their focused regression tests.

DudeCT prepares one binary, disassembly, symbol map, linker log, and provenance
snapshot per invocation under `target/ct/<target>/<profile>/dudect/runs/<run>/shared/`.
The bundle is read-only after preparation. Each `ct-full` case executes that binary
and writes its own CSV, stdout, and report under `cases/<manifest-name>-<unique>/`.
Reports reference the shared files and their hashes; they do not copy them.
Standalone selection evidence lives under the same run's `selection/` directory,
with a small latest report at `dudect/dudect-report.json`. Failed preparation or
execution cannot reuse a previous run's measurements. Historical runs remain on
disk until explicitly removed; full reports inventory only their current run.

## Benchmarks and updates

| Script | Caller |
| --- | --- |
| `bench/runner.py` | `just bench`, `just profile`, `just bench-export`, code inspection recipes |
| `bench/execution.py` | shared Cargo build, discovery, and provenance for measurement/profiling |
| `bench/measure.py` | runner: measurement and completion verification |
| `bench/profile.py` | runner: exact-case Samply capture |
| `bench/evidence.py` | shared build/runtime environment collector |
| `bench/settings.py` | measurement, profiling, and watchdog |
| `bench/bounded.py` | `just bench`, `just profile`: process-tree deadline |
| `update-all.sh` | `just update` |

`bench/benchmark_catalog.py` owns algorithm and target selection.
`.config/criterion.json` owns shared Criterion defaults and the maximum run
budget. `benches/common/criterion.rs` applies them to every Criterion harness;
`bench/settings.py` resolves invocation-wide overrides. `bench/bounded.py` stops
the whole benchmark or profile process tree within the budget.
`bench/evidence.py` owns the shared build/runtime environment collector.
`bench/runner.py` resolves filters to unique cases. `bench/measure.py` executes
one process per configuration and verifies statistical artifacts before the
runner marks a run complete. Export is a separate runner command.
`just bench --list` uses that same resolver to list actual cases with catalog work
classes, without starting a measurement run. `--diag` enables diagnostic cases.
`bench/benchmark_catalog_test.py` includes benchmark runner and profiling
regression tests. Run that focused suite with
`scripts/lib/python.sh scripts/bench/benchmark_catalog_test.py`. It exercises the
Python runner, Just argument forwarding, and export in temporary directories, with substitute
Cargo, benchmark, and profiler executors; it does not run cryptographic benchmarks.

Local and development-machine benchmarks share unique run directories under
`benchmark_results/criterion/<run-id>/`, with logs, plan, provenance, raw data,
and completion status. Explicit exports and checksums live in
`benchmark_results/.transfers/`. See [benchmarking](../docs/benchmarking.md)
for explicit baseline comparisons and remote collection.

Cargo Rail planning supplies affected scope for `just test`. Check, Miri, and
fuzz commands run independently of that plan.

## Shared libraries

| Script | Sourced or invoked by |
| --- | --- |
| `lib/rail-plan.sh` | `test/test.sh` |
| `lib/fuzz-packages.sh` | Fuzz scripts |
| `lib/python.sh` | Python-backed check, test, CT, and benchmark scripts |
| `lib/toolchain.py`, `lib/toolchain.sh` | Shared toolchain selection for installers, builds, checks, tests, and benchmarks |
| `lib/evidence_bundle.py` | Source binding, sealing, and transfer integrity for RISC-V tests and CT |
| `lib/riscv_build.py` | Pinned RISC-V cross-compiler environment for test and CT preparation |

Python tooling requires Python 3.11 or newer. The updater installs its catalog-pinned
Python libraries into a temporary virtual environment; checks and benchmarks use
the standard library.

## Native tooling

`just update` refreshes the tooling catalog, stable Rust, every Cargo manifest
(including standalone and fuzz support workspaces), lockfiles, and existing
GitHub Action pins. It runs on local macOS and has no dependency publish-age
filter. Inspect its changes before committing.

Run `scripts/tooling/<platform>.sh` on the native Ubuntu version pinned in
[the catalog](../.config/tooling.toml). Platforms are `aarch64-linux`,
`x86_64-linux`, `riscv64-linux`, `s390x-linux`, and `powerpc64le-linux`.
The installers use sudo when needed. Windows uses the corresponding
`aarch64-win.ps1` or `x86_64-win.ps1` in an elevated PowerShell session.
macOS tools remain locally managed.

CI calls these same installers with `--ci` on Linux or `-Ci` on Windows.
The catalog's `ci` section selects the Cargo tools needed by `just ci-check`,
`just test --all --release`, and `just test --all --release --portable`. Both
test commands include doctests. Only Linux x86-64 adds the `ci-policy` tools and runs `just ci-policy`:
Cargo Deny checks the full target graph in `deny.toml`, and Cargo Audit checks
the lockfile. Every host retains native and portable Clippy, independent-workspace
linting, documentation, and runtime tests; RISC-V performs its compilation checks
on the cross-build host and executes the resulting tests on native hardware. Linux CI omits OpenSSL development
packages, pkgconf, and recommended APT packages; CMake, Clang/libclang, Perl,
and the C/C++ build tools remain prerequisites for native test dependencies.
This mode omits Cargo Rail because `--all` bypasses affected-work
planning; use the full installer for ordinary `just test` and benchmark work.
Linux CI uses the catalog's `linux-ci` Ubuntu release and packages from the
same archive snapshot as development provisioning. It uses Cargo Binstall on
x86-64, ARM64, and RISC-V to select compatible binaries, falling back to source
when unavailable. IBM Z and POWER build Cargo tools from source.
CI does not install optional profiling, mutation, or live-fuzzing tools or alter
shell startup files. These jobs validate CI provisioning, not the full optional
development toolset.

After Linux installation, source
`$HOME/.local/share/rscrypto-tooling/environment.sh` in each new CI step.
Windows CI runs installation and validation in one PowerShell step to retain
the MSVC/SDK environment. Windows x86-64 installs catalog-pinned NASM for native
dependency assembly in both modes.

All full profiles install the prerequisites for `just ci-check`, `just test`, and
Criterion `just bench`. RISC-V, Z, and POWER use snapshot-pinned native
CMake/Clang. They do not install cross targets, Miri, browsers, or profiling
tools. The shared selector in
`lib/toolchain.py` uses `.config/toolchains.toml` to choose the pinned nightly
for POWER, IBM Z, and RISC-V; other hosts use `rust-toolchain.toml`.
Installers provision stable tooling plus the selected native toolchain.
Build, native check, test, and benchmark entry points use that selection rather
than an ambient `RUSTUP_TOOLCHAIN`; formatting uses the stable development pin.
Specialized Miri and fuzz checks retain their opt-in nightly recipes.

`ci.yml` also runs `--ci-package` provisioning and `just ci-package` on an
independent runner. This executes examples, verifies the publishable Cargo
archive, and runs external std/core/alloc consumers against the unpacked crate
on stable and MSRV. Core and alloc also compile on the existing Thumb sentinel.
No package is published.

`fuzz.yml` uses `--ci-fuzz` for committed ASan corpus replay and bounded live
fuzzing. Manual runs select x86-64, ARM64, or both, exact target names, and a
per-target duration. PR campaigns use 60 seconds per target with a 30-minute
live budget. Manual
qualification defaults to 1200 seconds per target with a five-hour live budget
and a six-hour job limit including installation/builds. Selection must fit its
budget before replay starts.
`--ci-miri` installs the pinned interpreter for an independent focused Miri row,
including RSA's unsafe-boundary tests. All rows share fail-fast cancellation.

`ct.yml` always runs full CT evidence, only through manual dispatch or a reusable
workflow call. It does not run on pull requests or pushes. Manual runs select
one, many, or all six native platforms, defaulting to all. A future release
workflow must call it for all platforms and require success on the same candidate
before publishing; no release workflow exists yet. Linux uses `--ci-ct-full` and
Windows uses `-CiCt`. The Linux installer additionally installs the pinned BINSEC,
Bitwuzla and decoder on GNU Linux x86-64/ARM64. Proof dependencies use a fixed
opam repository revision from `.config/tooling.toml`. Unsupported proof targets
retain their explicit `ct.toml` policies. No solver is installed there.

CT architectures run concurrently on fixed AWS instances or donated native
runners. Each host completes builds and proofs before serial timing cases.
`just ct-full` uses manifest-required cases and budgets without filtering. RSA
timing lives in this single harness, including entropy-backed signing; its
consolidated operation cases retain 2000 observations per class and a threshold
of 8. Proof failures stop timing; required timing failures stop later cases.
Local `just ct-dudect --smoke` remains a diagnostic shortcut outside this workflow.
Full CT evidence does not establish the complete secret-lifecycle claim by itself.

Both workflows retain evidence for seven days and run without caches. Manual
dispatch becomes available once they reach the default branch. CT and benchmark
selection jobs validate requests and emit only the requested runner rows; they
do not install Rust, build code, or invoke Cargo Rail.

`bench.yml` is manual-only. It selects one, many, or all six native CI platforms
and catalog algorithms, groups, or benchmark targets, with optional case filters.
A small planner starts only the selected runners. The existing benchmark runner
owns measurement and evidence. `--ci-bench` (Linux) and `-CiBench` (Windows)
install native benchmark build prerequisites and Just without test, profiling,
or cross-target tools. See [Benchmarking](../docs/benchmarking.md#run-a-manual-workflow).

Only x86-64 and ARM64 Linux install perf, Valgrind, Gungraun, and samply.
Their installer enables perf events and requires perf for the running kernel.
Use `just bench-structural` for Gungraun and `just profile` for samply;
Criterion benchmarks remain available on every native platform. Provisioning
checks tools, but native test, benchmark, and profiling execution must still
be verified on each machine.

### CI compatibility

The compatibility matrix row starts alongside every native row and participates
in the same fail-fast policy. `x86_64-linux.sh --ci-compat` installs only the
catalog-selected compatibility tools, Rust versions, and cross-target libraries.
`just ci-compat` uses bounded workers with separate build directories and a
shared CPU budget. A failed command terminates running siblings and prevents
queued work from starting. Logs remain under `target/compat/`.

Compatibility checks cover each standalone Cargo feature on the development
compiler and the declared minimum Rust version, broad native/portable feature
sets, and allocation-free and allocation-enabled Thumb sentinels. Every supported
bare-metal target also receives a release library build. Bare-metal evidence is
compile-only; it is not device execution.

Bare WASM and WASI both compile and execute the existing runtime vector harness
in Wasmtime, with scalar and SIMD artifacts tested separately. The scalar module
must load with SIMD disabled. Bare WASM calls an explicit argument-free export;
WASI uses its command entry point. These are Wasmtime results, not browser-engine
results. The library also receives broad feature builds for both WASM targets.

The x86-64 and ARM64 Linux rows install native musl build prerequisites and run
`just test-musl`: the complete native and portable test suites plus doctests,
compiled and executed for the matching musl target. Apple ARM64 and Windows
ARM64 execution remain deferred. No compatibility lane enables persistent caches.
