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
| `check/provider.py`                    | `just ci-provider`, `check/check.sh` |
| `test/test-musl.sh`                    | `just test-musl` |
| `check/macos.sh`                       | `just check-macos`, background Mac qualification |
| `check/macos_async.py`                 | local pre-push hook, `just qualify-macos`, `just macos-status` |
| `check/qualified.py`                   | local qualification reuse and release status identity |
| `check/staged.py`                      | local pre-commit and pre-merge-commit hooks |
| `check/dependencies.sh`                | `just ci-policy`, and the dependency checks in `just check` |
| `check/lint-independent-workspaces.sh` | `check/check.sh` |
| `asm/p384.py check`                    | `check/check.sh` |
| `asm/rsa_pair.py check`                | `check/check.sh` |
| `asm/provenance.py check`              | `check/check.sh` |

`check/check_runner_test.py` tests command selection, repair behavior, and failure propagation with substitute executors.
Run it with `scripts/lib/python.sh scripts/check/check_runner_test.py`.

`check/lint-independent-workspaces.sh` lints every independent Cargo workspace that Git tracks or would add.
Ignored manifests, such as local evidence crates, are not repository policy.

### Provider workspace checks

`check/check.sh` checks rscrypto across the full target catalog. It checks
`rscrypto-rustls` separately, with the provider's own native and portable features,
on every catalog target with `std`, including WASI. Bare-metal and browser WASM
remain rscrypto-only. Workspace documentation and host builds include both crates.

`just ci-provider` checks empty, default, each individual feature, all production
features, and all features on the host and minimum-Rust toolchain. CI runs it in
the compatibility row of the existing fail-fast matrix. Usable TLS and QUIC feature
bundles also execute the connection matrix in native and portable modes:
mutual authentication, stateful/stateless resumption where available, exporters,
key updates, and closure against independent rustls crypto backends.
Each feature row also runs `cargo tree -p rscrypto-rustls -e normal,build --target all`.
The gate rejects `ring`, `aws-lc-*`, `openssl-sys`, `boring-sys`, `cc`, and `cmake`.
The all-features row contains every additive dependency feature, so its graph
covers arbitrary feature combinations as well as the compiled rows.
The shared lockfile is not the gate: it includes rscrypto's dev-only crypto oracles.

Dependency policy checks duplicate versions without dev edges, matching the
existing exclusion of dev-only duplicates. This keeps oracle-only features on
shared dependencies from appearing as production duplicates. A second pass keeps
every other deny rule, including wildcard checks, on the complete workspace graph.

`test/test.sh` and cross-target Nextest archives include both workspace members.
`lib/workspace_features.py` qualifies each feature with its package name, including
when Cargo-Rail selects only the provider. A workspace plan explicitly selects
`--workspace`; Cargo's root-package default would omit the provider.
Coverage still measures core rscrypto
and its corpus replays. Package checks and publication explicitly select rscrypto.

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

## Generated paired RSA assembly

`asm/rsa_pair_aarch64.py` generates `mul_kernel` and `square_kernel` in `src/auth/rsa_aarch64_pair.rs`, the Apple
AArch64 engine that runs an RSA private key's two CRT exponentiations in the two vector lanes.
`asm/rsa_pair.py` has the same `check`, `write`, and `simulate [--cases N]` modes as `asm/p384.py`.
Its simulator runs both kernels on bounds-checked memory against digit-serial radix-2^28 Montgomery reduction
in Python integers, for every supported prime width and two other block counts.
`asm/rsa_pair_test.py` (part of `just test-scripts`) checks that template edits and single-instruction mutations
are detected.
`neon_crt_pair_matches_portable_exponentiation` in `src/auth/rsa.rs` is the native differential evidence.

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
| `test/test-examples.sh`  | `just test-examples` |
| `test/test-miri.sh`      | `just test-miri` |
| `test/test-fuzz.sh`      | `just test-fuzz` |
| `test/test-fuzz-asan.sh` | `just test-fuzz-asan` |
| `test/test-coverage.py`  | `just test-coverage` |
| `test/coverage_run.py`   | Coverage-only nextest target runner, invoked by `test-coverage.py` |
| `test/test-rsa-asm.sh`   | `just test-rsa-linux-asm`, `just test-rsa-macos-asm` |

- `just test-scripts` runs the argument-forwarding, toolchain, test, check, and fuzz regressions
  with substitute executors.
- `just test-fuzz-asan` rebuilds the standard library and replays the selected corpus tests
  with address instrumentation. On AArch64 Linux it disables outlined atomics: the current
  rebuilt Rust runtime otherwise reads sanitizer shadow memory inside `__asan_init`, before
  that memory exists. Inline LL/SC atomics preserve the sanitizer checks and corpus inventory.
- Check recipes run `test/fixtures.py` before compilation.
  It verifies Git-visible payloads under `testdata/`, `tests/vectors/`, and
  `rscrypto-rustls/tests/vectors/`.
  Each directory's `SHA256SUMS` must cover its payloads, including new untracked inputs.
  Missing, extra, duplicate, changed, or symlinked inputs fail.
  `README.md` and `NIST-NOTICE.txt` are provenance metadata, not hashed payloads.
  Git-ignored scratch files stay outside this inventory.
  `test/provider_vectors.py` reproduces the provider's TLS/QUIC vectors from
  checksum-pinned RFC Editor text; it performs no cryptographic computation.
  `test/provider_credentials.py` generates equivalent fresh TLS credentials and
  independent signatures with pinned OpenSSL 4.0.3 in a new output directory.
  `test/rsa_implicit_rejection_vectors.py --draft <text>` extracts the
  checksum-pinned RSA guidance draft's Appendix B, cross-checks every message
  with OpenSSL 4.0.3, and records seeded and Wycheproof decryptions. It requires
  that OpenSSL version on `PATH`; [fixture provenance](../testdata/rsa/implicit_rejection/README.md)
  records the command, seed, hashes, and unsupported 2049-bit import.
- `just test-provider-interop` builds the bounded loopback endpoint and runs the
  full suite/group/credential matrix in both roles against OpenSSL 4.0.3 and
  Go 1.27.3. Each row requires a full and resumed mutually authenticated
  connection. Go also supplies an independent exporter check. This POSIX process
  harness retains commands, source, versions, stdout/stderr, and results beneath
  `benchmark_results/`; `--quick` selects one diagnostic profile and is not full
  qualification. `--protocol tls12` runs only the opt-in TLS 1.2 suites, with the
  classical credentials and groups and the extended master secret required;
  `--protocol tls13` runs only the TLS 1.3 matrix. The ephemeral session files and exporter values protect only
  public test fixtures. The runner enforces timeouts and reaps peer processes.
- `just test-provider-quic` runs the pinned quinn-proto and quinn 0.12 revisions through
  this provider against quic-go 0.61.0 (`579ee19d`) on Go 1.27.3. The peer builds
  with CGO disabled. Cases cover full/resumed mutual authentication, the selected
  classical and ML-DSA profiles, all suites/groups, Retry, explicit 0-RTT, and
  exporter equality. Peer qlogs must show a packet-key phase change. The Go-server
  fixture uses bounded stateful tickets because its default stateless ML-DSA-87
  tickets exceed quic-go's 16 KiB CRYPTO limit on resumption. Provider servers
  exercise stateless tickets in ordinary/Retry cases and stateful tickets for
  0-RTT. Stateless-reset cases replace the server endpoint while retaining its
  reset key, then require the independent client to authenticate the reset.
- `just test-provider-quic-runner` builds the pinned Quinn interoperability
  application with this provider and runs the pinned QUIC Interop Runner in
  local Docker. The application uses h3-quinn 0.0.10 from its release tag with
  only its Quinn requirement moved to the provider's Quinn revision. Source revisions, container digests, Python dependencies,
  commands and packet traces are retained. `--prepared PATH` reuses the exact
  recorded images; `--tests` selects a bounded case batch. `--budget-seconds` can
  lower the default ten-minute cap. Each campaign removes its own containers and networks on exit.
  Failed, unsupported and incomplete cases remain nonzero. The independent Go image
  contains quic-go `99c7436` built with Go 1.27.0. This transport harness uses
  the upstream application's permissive chain verifier; it tests handshake
  signatures and transport behavior, not certificate trust. The separate
  loopback matrix owns mutually authenticated trust-chain evidence.
  A newly prepared ECN campaign wraps the pinned Go image to map the application's
  `ecn` case to its ordinary transfer path. The upstream ECT and ACK-ECN trace assertions
  stay unchanged. The wrapper's base digest and image identity are retained and checked;
  it does not add preferred-address migration support.
- `just test-provider-bogo` prepares pinned upstream rustls and runner checkouts,
  applies the retained provider/fixture patches, and runs the declared TLS 1.3
  profile. Scope exclusions come from protocol/version requirements and the
  explicit profile, not observed failures. Failures remain nonzero. The shim
  treats ML-DSA as an explicit `-verify-prefs` opt-in for the runner's default-policy
  tests; all signature operations still use this provider. `--test` narrows the
  campaign for diagnosis. It is not full-profile qualification.
- The `provider_quic_*` fuzz targets call production Retry, header, token, and
  handshake methods through hooks compiled only under `cfg(fuzzing)` or
  `cfg(rscrypto_internal_fuzzing)`. Ordinary builds omit them.
  `just test-fuzz-asan` sets `rscrypto_internal_fuzzing` and replays the four committed
  `provider_quic_*` corpora. Live campaigns use the normal
  `just test-fuzz --full --targets` front door with explicit duration and limits.
- The `tls-provider` benchmark selector and `just bench-provider-scorecard`
  use the independent [provider benchmark workspace](../tools/provider-bench/README.md).
  Its catalog target declares OpenSSL 4.0.3 as an external prerequisite,
  so default runs omit it and the Bench workflow rejects it; run it locally.
  Criterion measures complete TLS pairs and application records; the supplementary
  command retains individual latencies, process CPU, wire flights and process RSS.
- `just bench-provider-resources` uses the same production TLS fixture with a
  System-allocation observer on macOS/Linux. It shares signing keys and trust
  anchors, seeds separate ticket stores for concurrent resumed connections, and
  holds every pending pair live before releasing the workers. Reports retain
  configuration/preparation, individual-pair and concurrent live-heap peaks,
  allocation counts, source and binary identity, and cap failures. It does not
  measure latency or establish static stack or hostile-input bounds.
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

### SLH-DSA ACVP fixtures

`test/slhdsa_acvp.py derive` writes `testdata/slhdsa/acvp/sigGen.bin`, `sigVer.bin`, and the WASM
runtime subset `runtime.bin` from the pinned ACVP-Server SLH-DSA JSON. It fetches the files from the pinned commit, or reads them
with `--source DIR`, and refuses any file whose SHA-256 differs from the pin. The output is
deterministic; the fixture README documents the format. No recipe or check runs it: `just check`
verifies the committed bytes through `SHA256SUMS`.

### Cross-built tests

RISC-V, POWER, and IBM Z CI build on Ubuntu x86-64 with the `--ci-cross-build TARGET` tooling.

CI, CT, Bench, and Profile share two preparation actions:

- [cross-build/setup](../.github/actions/cross-build/setup/action.yml) runs the pinned cross-build installer and adds its Cargo tools to `PATH`.
  CI selects compiler-cache authority afterward.
  The other workflows keep compiler caching disabled.
- [cross-build/upload](../.github/actions/cross-build/upload/action.yml) requires both the prepared suite and its runner-tools archive,
  then uploads them together for two days,
  under the target's artifact name or the name that the caller passes.
  A Bench suite is a directory with one sealed archive per catalog target, and it must hold at least one.
  Diagnostic logs retain their separate failure handling and retention periods.

Each workflow still owns its preparation command, runner selection, timeout, native execution,
and fail-fast matrix.

RISC-V is the exception to the shared matrix.
GitHub cannot time out a queued job or cancel one job of a run,
so CI and CT dispatch RISC-V execution as a [RISC-V](../.github/workflows/riscv.yml) run.
That run checks out the caller's commit and downloads the caller's prepared archives.
`lib/dispatched_run.py` reports its result to the calling job:

- If no runner starts the run within 30 minutes, it cancels the run and passes with a skip notice.
- A started run must pass within its budget: 120 minutes for CI tests, 360 for CT.
- If another job of the calling run fails, it cancels the run and fails.
- `require_riscv`, which releases set, removes the 30-minute limit.
  The dispatching job still stops after six hours, including the queue wait.
- Fork pull requests skip RISC-V, because their read-only token cannot dispatch a run.

Initial reads of the dispatched run and its jobs may return 404 before GitHub exposes them.
The dispatcher retries those responses within its existing polling-error budget, only until a runner starts.
A missing parent run and authorization failures remain fatal.
Cancellation waits for a completed run; a 202 acknowledgment or 409 conflict is not proof that it stopped.
If normal cancellation does not finish within a minute, cleanup tries force-cancel.
If completion cannot be confirmed within the cancellation budget, cleanup fails and retains the run ID.

Only the dispatching job holds `actions: write`; it runs no build or test tooling.
`lib/dispatched_run_test.py` and `test/cross_workflow_test.py` check these outcomes and that wiring through `just test-scripts`.
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
These local results do not claim native qualification;
hosted CI has used the actions since `9d023e97`.

Test preparation and native execution retain these contracts:

- The build runs the same target-specific checks.
  It builds all release tests in both dispatch modes.
- Nextest archives move to the matching native runner.
  Its `--ci-cross-run TOOLS_ARCHIVE` tooling only runs them.
- The test bundle uses XZ compression; existing gzip bundles remain readable.
- Doctests compile and run on the native CI rows only.
- The suites run under the `cross` and `cross-internal` Nextest profiles in
  [`.config/nextest.toml`](../.config/nextest.toml). The SLH-DSA small-set independent-implementation
  differentials run on the native rows only, and the internal suites leave the SLH-DSA ACVP vectors to
  the ordinary suite with the same dispatch on the same target. Every other test runs in every suite.
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
  come from the same Ubuntu CI archive as native provisioning.
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

### Examples, Miri, and fuzzing

- Example names and feature requirements come from Cargo metadata.
- Use `just test-miri --rsa` for the focused RSA scope.
- Use `just test-fuzz --targets A,B` for an explicit group of fuzz targets.
- Package scope is selected before targets.
  `--all` includes matching targets from the full and scoped packages.
  `--full` and `--scoped` limit the scope.
  The default is full, including named targets.
- `just test-fuzz --build --all` builds every fuzz package without starting fuzzing.
- `just test-fuzz-ci plan` accepts only a manual dispatch.
  It selects the requested exact target names, the daily rotation, or every target,
  in every package that contains each name, and validates the live-fuzzing budget.
- `just test-fuzz-ci restore` merges the corpus checkpoint into the working corpora by SHA256.
  It rejects symlinks and invalid hashes and preserves committed seed files.
  `snapshot` prepares a new checkpoint after successful replay and fuzzing.
- `just test-fuzz-ci retain` minimizes the planned corpora and collects branch coverage with
  the pinned compiler's LLVM tools. It keeps committed regression seeds even when edge
  minimization drops them. The merge runs the built production fuzzer directly because
  `cargo-fuzz` 0.13.2 can return success after a failed `cmin` merge.
  A failed merge, replay, or coverage export cannot publish a successful campaign.
  `target/fuzz-logs/campaign.json` binds the compiler, effective source, selected targets,
  binary hashes, and corpus hashes to each report. `branch-gaps.json` lists unobserved
  outcomes in production code, including parser, decrypt, and verify paths; it does not
  prove those outcomes are feasible. Coverage is specific to the recorded corpus and build.

## Constant-time evidence

| Script                 | Caller |
| ---------------------- | ------ |
| `ct/zig-cc.sh`         | `ct/artifacts.sh`, `ct/binsec.py`, `stack/frames.py` |
| `ct/test.sh`           | `just ct-test` |
| `ct/artifacts.sh`      | `just ct-artifacts`, `ct/full.py` |
| `ct/dudect.sh`         | `just ct-dudect`, `ct/full.py` (preparation only) |
| `ct/dudect_execute.py` | `ct/dudect.sh`, `ct/full.py`, `ct/replay.py` |
| `ct/renew_waivers.py`  | `just ct-renew` |

- `ct/full.py`, `ct/binsec.py`, and `ct/validate.py` implement `just ct-full`, `just ct-binsec`,
  and `just ct-validate`.
- `ct/manifest.py` owns the shared target and measurement selection.
- `ct/provenance.py` owns the shared file hashing and build identity.
- `ct/linked_image.py` reads the linked ELF, PE, or Mach-O binary and its linker map, so that
  `ct/asm_heuristics.py` can describe data references independently of their placement.
- The other Python files under `ct/` implement local artifact provenance, disassembly analysis,
  report parsing, and their focused regression tests.

`just ct-dudect --smoke` uses each selected case's `smoke_samples` from `ct.toml`.
`--samples` overrides the environment sample setting,
and the environment setting overrides the manifest smoke budgets.
Each smoke case keeps its own measurements.
The latest report summarizes the selected cases and their requested budgets.

### Waiver renewal

A waiver binds the backward slice of the condition it reviewed (`slice_sha256`,
[docs/constant-time.md](../docs/constant-time.md#evidence-model)), so it goes stale only when the condition
changed or, for an open slice, when its function changed.
A stale waiver fails the verdict; the evidence and timing of the run are still complete.

`just ct-renew --target TARGET --reviewed-artifacts REVIEWED --artifacts NEW --out DIR` prepares the review.
`NEW` is any retained CT artifact directory of the run.

- It runs the gate's scanner on the new artifacts (`ct/asm_heuristics.py`, with slices from
  `ct/condition_slice.py`).
- For each stale waiver, it finds the reviewed branch in `REVIEWED` and lists the new branches of the same kind
  in the same function, with their slices and identities.
  A candidate whose slice is unchanged differs only because its open slice binds the function:
  confirm its open leaves.
- It lists every enforced `select_branch` finding that no waiver covers, with its slice and identity.
- It writes `renewal-record.json` and `renewal-record.md` under `--out` and changes nothing.
  To renew a waiver, set its `slice_sha256` to the reviewed candidate's identity and update `reviewed_at`.

A call defines every register that the target's calling convention lets the callee change.
On Windows x86-64, `rsi` and `rdi` keep their values across a call; every vector register is still clobbered,
because callees may change the upper bits of `xmm6`-`xmm15`.

### Constant-time verdict

`ct/full.py`, `ct/artifacts.sh`, `ct/validate.py`, and `ct/asm_heuristics.py` exit with status 3 when the
evidence is complete and its findings fail the verdict. Other nonzero statuses mean missing or inconsistent evidence.
The Constant-Time matrix jobs fail only for missing evidence, so a finding on one platform never cancels another
platform's timing. The `verdict` job runs `ct/ci.py verdict` on every retained report and decides the run.

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

- `--case` takes an exact manifest case, a comma-separated list of cases, `mldsa` for every required
  ML-DSA kernel case, or `mldsa-probe` for the ML-DSA diagnostic probes.
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
- Linux targets other than the host link through `ct/zig-cc.sh`, with the pinned Zig on the Mac.
  POWER links against glibc 2.32,
  the first release with the IEEE-128 `printf` entry points that Zig's ppc64le libunwind calls.
  The linked binaries are reviewed, never run.
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
   HKDF scenarios derive the HMAC prefix states with the independent `sha2` crate.
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

`just stack-residue --native-mldsa --output DIR` checks signing scratch on macOS
AArch64 and Linux AArch64/x86-64. It runs the production compact and prepared
signer, with deterministic and fixed-randomness hedged inputs, for all three
parameter sets and both dispatch configurations. Each call uses a private
painted 1 MiB stack between inaccessible guard ranges; the caller scans that
allocation after the call returns.
It records no timing measurements.

The runner captures coefficient masks, NTT masks, rejected responses and
commitments, and hint maps from an isolated source copy. It retains the exact
capture patch and matches every signature against the uninstrumented measured
build. High-entropy values use 16-byte windows. Hint maps use discriminating
256-byte windows or their complete 1,024-byte pattern when sparse; all-zero maps
already equal the cleanup value. Planted and cleared controls cover each window
class. Raw stack snapshots, captures, binaries, build logs, source hashes, and
`report.json` stay in the output directory. This checks the selected calls and
artifact, not arbitrary compiler-created fragments, registers, heap residue,
every rejection path, or other targets. `--cases N` changes the deterministic
message count per parameter set, key form, and mode (default eight); `--start N`
selects the first message index for reproducing a discovered rejection case.

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
| `bench/bounded.py`    | `just bench`, `just bench-structural`, `just profile`: process-tree deadline |
| `bench/mldsa_tail.py` | `just bench-mldsa-tail`: ML-DSA signing-tail baseline |
| `bench/compare.py`    | `just bench-compare`, Bench `Score comparisons` job: catalog comparison scorecard |
| `update-all.sh`       | `just update` |

- `bench/benchmark_catalog.py` owns the selector name space (targets, algorithms, groups, and `all`),
  curated profile-preset selection, and the validation of comparison declarations.
  A target's optional `architectures` limits where it runs: the runner skips it on other hosts and
  transfer targets, and each Bench platform job skips it as well.
  Optional `rustflags` build that target as its own configuration;
  the runner appends them to the inherited flags as `CARGO_ENCODED_RUSTFLAGS`.
  Profiling, code inspection, and cross builds reject targets with `rustflags`.
- `bench/compare.py` pairs each declared comparison's canonical `rscrypto` row with the fastest external
  row of the same task, on each platform, from retained runs.
  A comparison's optional `architectures` and required `availability` explanation record where an external
  implementation exists. Elsewhere, the measured rscrypto row remains in coverage and the report explains why
  it has no comparison. Unexpected external rows on an excluded architecture fail validation.
  It fails on an incomplete comparison point, on a missing competitor build for the platform,
  and on a measured group that has external rows but no declaration.
  `--cases` checks the declarations against `just bench --list` output without measuring.
  It rejects a listing without case lines and reports how many comparisons the listed binaries cover.
  The Bench workflow's `Score comparisons` job runs it on the run's extracted measurement archives.
- Bench's `rescore=RUN_ID/ATTEMPT` mode runs only a report job and ignores measurement inputs.
  `bench/ci.py rescore-source` verifies a completed main-branch Bench attempt and every native job's success.
  `bench/ci.py rescore` requires its exact archive set, revalidates measurements, and records the source run,
  attempt, measured revision, and report revision. It never permits partial comparisons or claims new measurements.
- `bench/ci.py evidence` packs one measurement job's `target/bench/` into the single raw archive that the job uploads,
  without the transferred binaries that the preparation artifact keeps.
- `.config/criterion.json` owns the shared Criterion defaults, the ordinary run budget,
  and the ceiling for catalog comparison exceptions.
- `benches/common/criterion.rs` applies them to every Criterion harness.
  `bench/settings.py` resolves overrides for one invocation.
- Catalog algorithms on split targets may declare `max_run_seconds` within that ceiling.
  `bench/runner.py` grants it only to one complete operation without a filter; quick runs
  and preparation keep the ordinary budget.
- `bench/bounded.py` stops the whole benchmark or profile process tree within the budget.
  For `--list`, and before a native measurement, it runs one runner invocation per selected target
  that builds and discovers it within the `max_comparison_seconds` ceiling,
  so a cold build cannot spend a measurement's budget.
- `bench/runner.py` resolves filters to unique cases and records the budget in `requests.json`.
- `bench/measure.py` runs one process for each configuration,
  passes the cases and budget to the compiled harness through `RSCRYPTO_BENCH_CASES`,
  and verifies the statistical artifacts before the runner marks a run complete.
- Export is a separate runner command.
- `just bench --list` uses the same resolver to list the actual cases with their catalog work classes,
  without starting a measurement run.
  `--diag` enables diagnostic cases.

`bench/benchmark_catalog_test.py` includes the benchmark runner and profiling regression tests.
`just test-scripts` runs it; run it alone with `scripts/lib/python.sh scripts/bench/benchmark_catalog_test.py`.
It tests the Python runner, Just argument forwarding, and export in temporary directories,
with substitute Cargo, benchmark, and profiler executors.
It does not run cryptographic benchmarks.

`just bench-mldsa-tail` builds [`tools/mldsa-tail`](../tools/mldsa-tail/src/main.rs) twice and
retains both runs under `benchmark_results/mldsa-tail-<time>/`. The production build times each
compact and prepared, deterministic and hedged signature of fixed inputs; the second build adds
`--cfg rscrypto_mldsa_attempts`, which counts rejection-sampling attempts and is not production
latency. Inputs are deterministic, so attempt counts match on every target. The recipe prints both
outputs, raw rows and summaries, and the provenance, so a `just ssh-just` log carries them home.

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
| `lib/python.sh`                        | Python-based check, test, CT, and benchmark scripts, through uv |
| `lib/dispatched_run.py`                | The CI and CT `riscv64-linux` jobs |
| `lib/toolchain.py`, `lib/toolchain.sh` | Shared toolchain selection for installers, builds, checks, tests, and benchmarks |
| `tooling/transfer.py`                  | Cross-build and verify the pinned native runner tools |
| `tooling/uv.sh`                        | Install the catalog-pinned uv and Python on Linux and the Apple Silicon Mac, including jobs that run no installer |
| `tooling/zig.sh`                       | Select the Zig that `ct/zig-cc.sh` runs: the catalog pin on the Apple Silicon Mac, otherwise the one on PATH |
| `tooling/fingerprint.py`               | Digest the inputs of one platform's installation, so a converged host or image can trust it |
| `lib/evidence_bundle.py`               | Source binding, sealing, and transfer integrity for cross-compiled tests, tools, and CT |
| `lib/cross_build.py`                   | Explicit target identities and the cross-compiler environment for test, tool, and CT preparation |

Every Python command runs through uv: `lib/python.sh` runs `uv run`,
and scripts start with `#!/usr/bin/env -S uv run python`.
uv provides the interpreter that [`.python-version`](../.python-version) names,
and [`uv.lock`](../uv.lock) pins the packages of [`pyproject.toml`](../pyproject.toml); a stale lock refuses.
`lib/python.sh` always selects the project of its own checkout.
The staged check and Mac qualification also remove the caller's `uv run` variables and virtual environment,
so each worktree runs its own lock.
Checks and benchmarks use only the standard library.
The `tooling` group holds the updater's libraries and those of the catalog and workflow tests;
`lib/python.sh --group tooling` selects it.
The installers install the catalog's pinned uv.
On the Apple Silicon Mac, `lib/python.sh` runs the catalog's `aarch64-macos` uv, which `tooling/uv.sh` installs under `~/.local/share/rscrypto-tooling/uv/` on first use;
a uv on PATH, such as Homebrew's, is neither used nor changed.

## Native tooling

### Runner profiles

[`.github/runs-on.yml`](../.github/runs-on.yml) defines the AWS runner profiles.
Change CPU, instance families, images, storage, and Spot policy there.
Workflows refer to profile names, and do not override their shapes.

- CI, CT, and benchmark cross-builds have separate profiles.
  So do native CI, fuzzing, and CT measurement.
- Release and every AWS profile used by its called CI and CT workflows use On-Demand.
  This includes planning, cross-build preparation, package verification, compatibility,
  and publication.
  The workflow regression tests resolve the complete runner graph,
  including the generated CT matrix, and require explicit `spot: false` without job-label overrides.
- Fuzz and Miri also use On-Demand.
  CT and benchmark measurement use separate fixed On-Demand profiles.
- Only benchmark cross-build preparation uses price-capacity-optimized Spot.
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
Local macOS tools are managed locally, except the uv and Zig that the catalog pins for `aarch64-macos`.
The scripts install those two on first use under `~/.local/share/rscrypto-tooling/` and keep them off PATH,
so a Homebrew uv or Zig stays as it is.
`scripts/tooling/aarch64-macos.sh` can install the pinned prerequisites for `just check-macos`.

All full profiles install the prerequisites for `just ci-check`, `just test`, and Criterion `just bench`.
RISC-V, IBM Z, and POWER use native CMake and Clang from the Ubuntu archive.
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
  RISC-V, POWER, and IBM Z run their compilation checks on the cross-build host,
  and run the tests on native hardware.
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
It also installs pinned Strawberry Perl and selects it only through `OPENSSL_SRC_PERL`:
the vendored OpenSSL that lints `tools/provider-bench` cannot configure under Git's MSYS Perl.

### Ubuntu packages and converged hosts

Linux CI uses the catalog's `linux-ci` Ubuntu release; development uses `linux`.
Both install Ubuntu packages from the guest's own archive, with retries.
A host that already holds every selected package skips APT,
and a baked image keeps the versions it installed.

Every installer can fingerprint its inputs on any host:
`scripts/tooling/<platform>.sh fingerprint [<operation>]`, or the `.ps1` front door on Windows.
An operation is `dev` or a flag without its dashes, such as `ci` or `ci-ct-full`;
the installers also accept it in that form (`-Operation` on Windows).
The fingerprint covers the files that the platform's installer reads for that operation,
the operation, the required OS release, and the platform.

- A complete run writes it to `fingerprint.<operation>` in the tooling prefix.
- While it matches, a later run republishes the environment, checks the pinned tool versions, and stops.
  A failed check converges instead.
- Otherwise each step checks before it acts:
  rustup and Cargo tools by their reported versions,
  and on Windows Visual Studio by vswhere and Git and jq by their versions.
- Cross-build and cross-run installations depend on a target or a transferred archive,
  so they always converge.
- Inside a container, development setup leaves perf and its kernel setting to the host.

[`.config/images.toml`](../.config/images.toml) lists the dev-machines images baked from these installers,
with the operations each RunsOn image runs.

### Workflows

**`ci.yml`** also runs `--ci-package` provisioning and `just ci-package` on an independent runner.
It runs the examples, verifies the publishable Cargo archive, and runs external `std`, `core`,
and `alloc` consumers against the unpacked crate on the canonical and MSRV toolchains.
`core` and `alloc` also compile on the existing Thumb sentinel.
No package is published.

**`fuzz.yml`** uses `--ci-fuzz` for ASan corpus replay and bounded live fuzzing.

- It runs only through an explicit manual dispatch of Fuzz.
  Pushes, pull requests, schedules, and Release do not start it.
- Manual runs select x86-64, ARM64, or both, exact target names, and a duration for each target.
  Blank target names select every target.
  Every run builds all fuzz targets and replays committed and restored inputs under ASan
  before live fuzzing.
- Smoke campaigns default to 120 seconds per selected target.
  All campaigns have a planned live-fuzzing budget of 30 minutes
  per architecture, with eight concurrent package/target pairs.
- Manual runs with `campaign=daily` rotate eight target names according to the UTC date.
  They divide the same 30-minute budget across the selected package/target batches,
  then minimize those corpora and retain source and branch coverage for each one.
  Repeated runs across dates cover the rotation; reports from earlier cohorts retain their original
  source and corpus identities and do not qualify a newer build.
  The daily budget determines its per-target duration.
- Successful `main` runs save a separate corpus checkpoint for each architecture.
  Runs on other branches can restore checkpoints but cannot save them.
  Checkpoints contain inputs only, never build outputs. GitHub caches can be evicted;
  every run also uploads corpora, logs, and crash reproducers as 30-day artifacts.
  Concurrent runs restore independent snapshots; the latest cache is a starting point,
  while each run's artifact preserves its discoveries.
- Fuzz jobs have a 90-minute limit, including installation, builds, corpus replay,
  live fuzzing, and daily coverage when selected. Miri jobs have a 60-minute limit.
- `--ci-miri` installs the pinned interpreter for an independent focused Miri row,
  including the RSA unsafe-boundary tests.
- All rows share fail-fast cancellation.
- After those rows pass, `Release qualification` passes only for both architectures,
  a smoke campaign, blank targets, and at least 120 seconds per target.
  Release preflight requires this job to pass in the current attempt of the latest manual Fuzz run
  on the exact source commit. A missing, failed, unfinished, or scoped run blocks release;
  preflight does not fall back to an older success or dispatch Fuzz itself.

**`ct.yml`** runs full CT evidence, only through manual dispatch or a reusable workflow call.
It does not run on pull requests or pushes.
A manual `diagnostic_case` input replaces the full campaign with scoped evidence;
see [Diagnostic CT runs](#diagnostic-ct-runs).

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

The CT workflow keeps final evidence for seven days and runs without caches.
Fuzzing caches only its corpora; it retains evidence for 30 days.
CT preparation archives are kept for two days.
The CT, benchmark, and profile selection jobs validate requests
and emit only the requested runner rows.
They do not install Rust, build code, or call Cargo Rail.

**`bench.yml`** runs only on manual request.

- Its form takes catalog names in `benchmarks`, an optional case `filter`,
  one checkbox per native CI platform, diagnostic and quick checkboxes, and optional sampling overrides.
  The planner reads the form as one `DISPATCH_INPUTS` JSON value and gives later jobs a one-line copy,
  because `Enter-VsDevShell` keeps only the first line of a multi-line variable.
- `bench/ci.py plan` groups the selection by the catalog's primitive `family` and starts one job per platform and family.
  Each job runs one bounded `just bench` per selected target, one after another.
  A target marked `"split": "algorithms"`, such as `slhdsa`, runs one per catalog algorithm,
  and the runner fails discovery unless each of its cases belongs to exactly one algorithm.
  The planner rejects targets with an `external` prerequisite before any runner starts.
- SLH-DSA selectors separate operations
  while keeping every implementation of each comparison together.
  Its existing parameter-set names expand to those selectors,
  and the default `auth` schedule runs SLH-DSA after the other targets.
  The 12 small-signature signing comparisons have a 30-minute catalog budget;
  all other invocations retain the ordinary ten-minute limit.
  The CI planner sums the selected measurement budgets
  while preparation keeps ten minutes per invocation.
  A failed invocation still ends its job.
- RSA key generation runs one key size per invocation.
  `rsa-4096-keygen` also has the 30-minute catalog budget.
- Job labels use platform names, never target triples:
  `Prepare PLATFORM / FAMILY` and `Measure RUNNER / FAMILY`.
- RunsOn starts a host for every queued job and ignores `max-parallel`, so the job count bounds the AWS hosts.
  Each RunsOn job has its own runner label, so no job takes a host started for another.
  `test/cross_workflow_test.py` checks the largest run against the account's vCPU quotas.
- The existing benchmark runner owns measurement and evidence.
- Quick mode keeps every selected row and its output check, records Criterion's quick sampling in the sealed
  settings, and reports coverage without performance rankings. It rejects sample-count and warmup overrides
  and cannot compare performance baselines. Quick mode keeps the ten-minute invocation budget.
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
  Rust versions, and cross-target libraries. Nextest runs `just ci-provider`'s runtime matrix.
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

`--cfg rscrypto_mldsa_attempts` is a separate, narrower flag. Only `just bench-mldsa-tail` sets it,
to count ML-DSA signing attempts; CT and other evidence builds never contain the counter.

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
- The `aead_diag` and `aead_kernels` targets enable their required internal hooks automatically.
- Cross-prepared diagnostic benchmarks resolve the flags for the destination target
  before they record build provenance.
- Keep internal builds separate from public-surface checks.
  Use `just ct-binsec` for binary proofs and `just ct-dudect` for timing evidence.

## Release orchestration

`.github/workflows/release.yml` installs and verifies the `ci-package` tooling profile.
The profile includes the catalog-pinned `cargo-semver-checks` required by the release API gate.
The preflight then checks prior Mac and manual Fuzz qualification before calling CI and CT.
The final release job installs and verifies the same profile on its own runner.
That job runs `cargo rail release` through `cargo-rail-action/release`, which commits, tags, pushes, publishes to crates.io,
and creates the GitHub Release.
`scripts/release/release.py preflight` first requires the newest `rscrypto/macos` commit status of the source commit to pass.
Its description binds the Git tree and a SHA-256 digest of `rustc -vV`,
excluding only the host line so the Linux release runner can compare the same compiler distribution.
An older successful status cannot override a newer failure or compiler mismatch.
The preflight also reads the latest manually dispatched Fuzz run for the exact source commit.
It requires a successful run and a successful `Release qualification` job in that run's current attempt.
Only a full smoke campaign on both architectures, with Miri and at least 120 seconds per target,
can pass that job.
The default Fuzz inputs satisfy this contract.
The preflight can read statuses and workflow runs but cannot write them or dispatch Fuzz.
Its refusal tests run through `just test-scripts`.
[CONTRIBUTING.md](../CONTRIBUTING.md#release) has the maintainer setup, check, deployment, and retry instructions.

`scripts/check/macos.sh` owns `just check-macos`, which replaces hosted macOS checks and tests with local Apple Silicon validation.
Install `.githooks` with `just install-hooks` in each maintainer checkout. macOS is still a supported release target.
Timing qualification on physical Apple Silicon is a separate local requirement before submission.

`scripts/check/staged.py` runs the pre-commit check on the staged tree in `rscrypto-ci-check/checkout`,
a persistent detached worktree in the common Git directory, with Cargo output in `rscrypto-ci-check/target/`.
Git rewrites only the files that changed since the previous check, so Cargo rebuilds only what changed.
No other checkout shares that target directory, because Cargo decides freshness by timestamp.
Hook-local Git variables are removed before any command runs in the worktree.
A pass is recorded only if the staged tree, the checked files, and the compiler are unchanged after the check.
To reclaim the space, delete `rscrypto-ci-check/` and run `git worktree prune`; the next commit rebuilds it.

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
