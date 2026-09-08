# Scripts

Repository scripts implement local development, testing, evidence, and
benchmark commands. User-facing entry points are the recipes reported by
`just --list`. Every script in this directory is locally executable.

## Check entry points

| Script | Caller |
| --- | --- |
| `check/check.sh` | `just check`, `just ci-check` |
| `check/lint-independent-workspaces.sh` | `check/check.sh` |

`check/check_runner_test.py` tests command selection, repair behavior, and
failure propagation with substitute executors. Run it with
`scripts/lib/python.sh scripts/check/check_runner_test.py`.

## Test entry points

| Script | Caller |
| --- | --- |
| `test/test.sh` | `just test` |
| `test/test-examples.sh` | `just test-examples` |
| `test/test-miri.sh` | `just test-miri` |
| `test/test-fuzz.sh` | `just test-fuzz` |
| `test/test-fuzz-asan.sh` | `just test-fuzz-asan` |
| `test/test-coverage.py` | `just test-coverage` |
| `test/test-rsa-leakage.sh` | `just test-rsa-leakage` |
| `test/test-rsa-asm.sh` | `just test-rsa-linux-asm`, `just test-rsa-macos-asm` |

`just test-scripts` runs argument forwarding, toolchain, test, check, and fuzz regressions with
substitute executors. `just test-harnesses` directly runs the CT harness and
exporter self-tests; `just ct-test` includes them.

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

All profiles install the prerequisites for `just ci-check`, `just test`, and
Criterion `just bench`. RISC-V, Z, and POWER install pinned Cargo tools from
source and use snapshot-pinned native CMake/Clang. They do not install cross
targets, Miri, browsers, or profiling tools. The shared selector in
`lib/toolchain.py` uses `.config/toolchains.toml` to choose the pinned nightly
for POWER, IBM Z, and RISC-V; other hosts use `rust-toolchain.toml`.
Installers provision stable tooling plus the selected native toolchain.
Build, native check, test, and benchmark entry points use that selection rather
than an ambient `RUSTUP_TOOLCHAIN`; formatting uses the stable development pin.
Specialized Miri and fuzz checks retain their opt-in nightly recipes.

Only x86-64 and ARM64 Linux install perf, Valgrind, Gungraun, and samply.
Their installer enables perf events and requires perf for the running kernel.
Use `just bench-structural` for Gungraun and `just profile` for samply;
Criterion benchmarks remain available on every native platform. Provisioning
checks tools, but native test, benchmark, and profiling execution must still
be verified on each machine.
