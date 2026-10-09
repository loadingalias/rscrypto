# Benchmarking

Benchmark the exact primitive, operation, input size, feature set,
and target that you plan to deploy.
A crate-wide aggregate is not a deployment result.

## Read published results

[`benchmark_results/OVERVIEW.md`](../benchmark_results/OVERVIEW.md) owns the published summary.
Keep raw results and run metadata for every published claim.
A local measurement without that evidence applies only to the machine that made it.

## Run records

Each new local or development-machine run writes its own directory:

```text
benchmark_results/criterion/<run-id>/
```

Each run records:

- the literal requests and the log;
- source-file hashes;
- the compiler and Cargo identity;
- the build and runtime environment;
- CPU and operating-system information;
- the raw `criterion/` data;
- `status.txt`.

Successful discovery adds the resolved case plan.

`status.txt` starts as `running`.
It becomes `complete` or `failed`, with the exit code of the execution.
`complete` needs, for every planned case, the case identity, the statistical samples, and the estimates.
It also needs the comparison estimates when a baseline was given for that case.
If the process succeeds but results are missing, the run fails.
A run that is interrupted abruptly can stay `running`.
If packaging fails, the run is kept and no checksum is published.
Source hashes identify a dirty worktree.
They do not replace keeping the source.

### Export and collect

Export a run only when you need a portable artifact:

```bash
just bench-export benchmark_results/criterion/<run-id>
```

This archives the run, including failed-run evidence, under `benchmark_results/.transfers/` with a SHA-256 checksum.
Export refuses to overwrite an existing archive.
`output_dir=<path>` changes the results root, and keeps the `criterion/` and `.transfers/` layout.

On development machines, use the default root.
Collection exports the selected run, then downloads it:

```bash
just ssh-collect-bench <target> <run-id> <new-local-directory>
```

Collect before you destroy the machine.
The destination must not exist yet.
Historical result directories, named by date, operating system, and architecture, do not change.

### Compare with a baseline

Each run starts with new Criterion output.
To compare with a completed run, select it explicitly:

```bash
just bench sha256 baseline=benchmark_results/criterion/<previous-run-id>
```

- Only validated Criterion `base` data for matching cases and configurations is copied into the new run.
  The previous run does not change.
- Compatibility includes the Cargo command, compiler, manifest profiles,
  Cargo configuration file hashes, CPU identity, build and runtime controls,
  and effective Criterion settings.
- The source revision and the watchdog budget are recorded, but they are not part of compatibility.
- A baseline needs a resolved plan and verified measurements.
- If no configuration or case matches, the command fails.
  Otherwise, cases without a match run without a comparison.
- Matching metadata does not prove the same thermal, power, or system-load conditions.
  Control those before you interpret a comparison.

### Speedup

```text
speedup = comparison_time / rscrypto_time
```

A value above `1.00x` favors `rscrypto`.
A value below `1.00x` favors the comparison.
Summary tables treat `0.95x` through `1.05x` as a tie.

[`.config/benchmark-matrix.json`](../.config/benchmark-matrix.json) owns the benchmark binaries, required features, aliases,
and filters.
The benchmark source owns each timed operation.
Inspect both before you claim equivalent work.

## Bench workflow

The [Bench workflow](../.github/workflows/bench.yml) runs only on manual request.
Select the revision with the GitHub branch selector, then set the inputs:

| Input           | Example                                       | Meaning |
| --------------- | --------------------------------------------- | ------- |
| `architectures` | `x86_64-linux`                                | One native platform. x86-64 runs both Intel and AMD. |
| `architectures` | `s390x-linux,powerpc64le-linux,riscv64-linux` | Any subset, separated by commas or spaces. |
| `architectures` | `all`                                         | Linux x86-64, Linux ARM64, Windows x86-64, IBM Z, IBM POWER, and RISC-V. |
| `selection`     | `sha256`                                      | One catalog algorithm. |
| `selection`     | `sha256,blake3`                               | More than one algorithm. |
| `selection`     | `hashes`, `checksums`, `auth`, `aead`         | A catalog group. You can combine groups. |
| `selection`     | `all`                                         | Every algorithm in the catalog's `all` selector. |
| `selection`     | `bench=sha2,auth`                             | Whole benchmark targets, including cases outside one algorithm. |
| `filter`        | `^sha256/rscrypto/64$`                        | Limit the selected scope to matching Criterion cases. |

The other platform names are `aarch64-linux` and `x86_64-win`.
Algorithm and group selectors and explicit `bench=` targets are different forms of `selection`.
The workflow rejects invalid architecture and catalog selections
before it starts measurement runners.
A case filter that matches nothing fails during discovery.
The catalog is the authority for the available selectors and targets.

Optional sampling fields override the shared Criterion settings.
Empty fields keep the repository defaults.
The diagnostic checkbox enables diagnostic features for the selected targets.
It does not select other targets.

### Runners

A small planning job validates the request and creates the exact runner matrix.

- AWS supplies fixed On-Demand instance types for Linux x86-64, Linux ARM64, and Windows x86-64.
  Both x86-64 operating systems run separate Intel and AMD jobs, with different artifact names.
- [`runs-on.yml`](../.github/runs-on.yml) defines the machine shapes.
  Benchmark preparation uses `bench-cross-build`.
  Measurement uses the `measure-*` profiles.
  They are sized separately from the CI and CT profiles.
- These profiles use the fixed processor families named in the catalog, with no AVX2-only baseline.
- IBM and RISE supply their existing native runners.
- macOS benchmarks run locally on the Apple Silicon Mac.

The selected architectures run at the same time.
On each machine, benchmark configurations run one after another.

- RISC-V, POWER, and IBM Z compile the selected configurations on x86-64 with `--ci-cross-build TARGET`.
  Their native jobs use `--ci-cross-run` with the verified tools archive,
  then discover and measure the transferred binaries.
- Other Linux jobs use `--ci-bench`.
  Windows uses `-CiBench`.

No caches and no speed-regression gates are enabled.
Donated hosts can be shared, and fixed AWS instance types do not remove host noise.
Equal vCPU counts do not mean equal physical core counts.
Use the recorded CPU topology to interpret parallel results.

### Artifacts and limits

Each job keeps `target/bench/` as a GitHub artifact.
It includes failed-run evidence, source and machine identity, the resolved case plan, logs,
and raw Criterion results.

Preparation and native measurement each have their own 90-minute budget.
`all` is a selection; it does not guarantee that every case fits in that budget.
Limit large runs by algorithm, group, target, or case filter.
The workflow allows extra provisioning time, especially on RISC-V.
Manual dispatch is available after the workflow reaches the default branch.

### Cross-build preparation

Cross-build preparation never runs target code.
It seals each unique catalog build configuration, the source identity,
the compiler and linker evidence, the exact binary hash, and the requested sampling settings.

Native consumption rejects changed sources, settings, configurations,
or ELF architectures before discovery.
It uses the same case filtering, measurement, and result verification as an ordinary run,
without compiling again.
The retained input manifest identifies the build host.
Result compatibility records the measurement host and its runtime settings.
Cross-built and native-built results have different build identities for baseline comparisons.

The two invocations use matching selections and settings:

```bash
just bench ... target=TARGET prepare_archive=ARCHIVE
just bench ... target=TARGET run_archive=ARCHIVE
```

The same selections work locally:

```bash
just bench sha256 blake3
just bench hashes
just bench all
just bench bench=sha2 'filter=^sha256/rscrypto/64$'
```

## Profile workflow

The [Profile workflow](../.github/workflows/profile.yml) runs only on manual request.
It takes one native Linux architecture and one curated primitive.

- The benchmark catalog maps each primitive to one benchmark target and one exact production case.
  It also owns the diagnostic-feature policy.
- The default selection is `aead/aes`.
  It profiles `aes-128-gcm/copy-and-encrypt/rscrypto/4096` for five seconds.
- The ML-DSA presets use ML-DSA-65: `auth/mldsa65-keygen`, `auth/mldsa65-prepared-sign`, and `auth/mldsa65-prepared-verify`.
  Signing is deterministic.
  Signing and verification do not include key preparation.

The GitHub UI shows the same two inputs as the CLI:

```bash
gh workflow run profile.yml --ref BRANCH \
  -f architecture=riscv64-linux -f primitive=hashes/sha256
```

How a capture runs:

1. The planning job validates the request before it reserves native hardware.
1. An x86-64 job cross-builds and seals the production benchmark.
1. The native job verifies the archive and discovers the case exactly once.
1. It records with `perf record`.
1. It writes a symbol hotspot report with `perf report --stdio --no-inline`.
   The report includes call paths when the runner supports them.
1. The GitHub job summary shows the report.
   The artifact keeps the exact binary, the raw `perf.data`, the report, and the machine identity.

Collector setup:

- Native setup prefers the runner's `perf`.
- The pinned RISC-V kernel has no matching Ubuntu tools package,
  so setup builds `perf` from the matching pinned upstream stable source.
- Other donated runners fall back to the pinned Ubuntu generic userspace tool.
- The selected collector version is recorded next to the kernel identity.
- A live capability probe decides whether the tool and the host can collect evidence.
- When runner policy denies access, native setup enables perf events.
  The benchmark stays unprivileged.

The workflow keeps the evidence for missing packages, denied permissions, unsupported events,
and partial captures.
Native reports and raw evidence stay downloadable for 30 days, also when capture fails.
Preparation and native capture each have a 20-minute cap.
Planning has a five-minute cap.
A newer request for the same architecture cancels an older request that is still running.
Inspect the uncertainty, and repeat matched measurements, before you make a performance claim.

## Timed workload boundaries

Choose the timed boundary from the question that the workload answers.
Write it next to the benchmark group in the source.
Include input restoration, allocation, key or state construction, output handling, and destruction.
Put material included or excluded work in the case identity.
Implementation names must identify the library or backend that is actually called.
A renamed boundary starts a new baseline.

- **Reusable-buffer operation:** allocate storage and prepare reusable state outside timing.
  Time the operation on that state.
  If the operation needs fresh state, describe any batched setup explicitly.
  Do not call its allocation part of the measured operation.
- **Copy plus operation:** inside timing, restore the input into preallocated storage,
  then run the operation on it.
  Use `copy-and-…` in the operation name.
  Use this boundary to measure the cost of keeping an immutable source message.
- **Complete application operation:** include the real lifecycle under study, and name its stages,
  for example `copy-and-construct-and-seal`.
  State which application costs are still excluded.
  Constructing a cipher does not mean that packet allocation, entropy, or transport is included.

Do not move setup out of timing only to get a smaller number.

- `iter` times the work and destruction inside its closure, and the destruction of its return value.
- `iter_batched` excludes the setup closure and defers destruction of the returned output.
  Consumed inputs can still be destroyed inside the timed closure.
- `iter_batched_ref` also defers destruction of the setup object.

The P-256 ECDH batches prepare new consumed keys outside timing.
RapidHash map-insertion batches allocate empty maps outside timing.
These boundaries are different from the timed buffer restoration in AEAD rows.

### AEAD rows

The AEAD `copy-and-encrypt`, `copy-and-decrypt`, `copy-and-seal`, and `copy-and-open` groups reuse preallocated buffers and cipher contexts.

- **Timed:** input restoration, cryptography, and per-call output handling and cleanup.
- **Not timed:** fixture generation, the first buffer allocation,
  and construction and destruction of the reusable context.
- Rows labeled `appended-tag` copy or produce the combined ciphertext-and-tag form.
  Other rows use detached tags.
- Throughput counts message bytes, not restoration traffic or tag bytes.
- AES-SIV `copy-and-construct-and-seal` also constructs and destroys a cipher in each iteration.

These are not cryptography-only measurements,
and they are not complete packet-processing measurements.
Construction-only and header-mask groups state their own boundaries next to their registrations.

### Other primitive rows

- Ed25519 `verify/rscrypto` reuses a public key constructed outside timing.
  `verify/rscrypto-import` imports the encoded public key and verifies one signature in each timed call.
  Both rows use the same message and signature;
  untimed checks compare the public key and signature with Dalek.
  The import row exposes setup costs that key reuse can amortize.
  The diagnostic `verify-phase/portable-double-scalar` and `verify-phase/aarch64-asm-double-scalar` rows include portable public-point decoding and result encoding;
  they do not measure cached-key verification alone.
- The ChaCha diagnostic group `chacha20-copy-and-xor` restores the message and applies the keystream in the timed closure,
  with a reused allocated buffer.
  Poly1305 reads immutable fixture bytes and returns a tag, without restoring a message buffer.
  Neither is a complete AEAD operation.
- BLAKE2 `short-oneshot` and `short-keyed` keep only the 16-byte and 128-byte inputs that the main size matrix does not have.
  `single-update` measures construction, one update, and finalization at the small sizes.
  It is different from the multi-chunk streaming workload.
  Plain parameter-group duplicates are removed.
  The main one-shot rows are the baselines for salt and personalization hashing.
  All of these are complete hash operations, not isolated host overhead.
- Ascon's `rscrypto/scalar-loop` rows compare repeated calls to the `rscrypto` scalar API with its batch API.
  They do not compare with an external library.

## ML-KEM and Argon2 comparison contracts

Compare only rows in the same operation group, with matching build and host identities.
These contracts replace the old ML-KEM IDs, and the Argon2 IDs without `salt16-raw32`.
Do not reuse those old measurements as baselines.
The effect of the old mismatches on the reported ratios has not been measured.

### ML-KEM

ML-KEM uses fixed 64-byte key-generation seeds,
and fixed 32-byte encapsulation randomness in the `derand` groups.
Fixture construction is not timed.
Decapsulation uses the same key material and ciphertext in every implementation.
It consumes no entropy.
Each parameter set has its own groups.

| Operation suffix | Timed input and preparation | Timed output |
| --- | --- | --- |
| `keygen/derand-encoded` | Seed from the caller; generation and export | Encoded public key and expanded secret key as fixed byte arrays |
| `keygen/internal-entropy-encoded` | AWS-LC generation, internal entropy, and export | The same key encodings as fixed byte arrays |
| `encapsulate/derand-reuse-matrix-prepared` | Reused `rscrypto` key with cached public matrix; randomness from the caller | Ciphertext array and 32-byte shared-secret array |
| `encapsulate/derand-reuse-decoded` | Reused RustCrypto decoded key and cached key hash; matrix sampling and caller randomness stay timed | The same ciphertext and secret arrays |
| `encapsulate/derand-reuse-encoded` | Reused `rscrypto`, libcrux, or fips203 encoded-key wrapper; caller randomness; per-call decoding stays timed | The same ciphertext and secret arrays |
| `encapsulate/derand-import-encoded` | Identical public-key bytes; each API's import, validation, and encapsulation with caller randomness | The same ciphertext and secret arrays |
| `encapsulate/internal-entropy-reuse-native` | Reused AWS-LC key object; internal entropy | Ciphertext and secret converted to fixed arrays |
| `encapsulate/internal-entropy-import-encoded` | The same public-key bytes; AWS-LC import and internal entropy | Ciphertext and secret converted to fixed arrays |
| `decapsulate/reuse-matrix-prepared` | Reused `rscrypto` key with cached public matrix, and typed ciphertext | 32-byte shared-secret array |
| `decapsulate/reuse-decoded` | Reused RustCrypto decoded key and typed ciphertext; re-encryption samples the public matrix inside timing | 32-byte shared-secret array |
| `decapsulate/reuse-encoded` | Reused `rscrypto`, libcrux, or fips203 encoded-key wrapper and typed ciphertext; per-call decoding stays timed | 32-byte shared-secret array |
| `decapsulate/reuse-native` | Reused AWS-LC key object and borrowed ciphertext bytes | Shared secret converted to a 32-byte array |
| `decapsulate/import-encoded` | Identical expanded secret-key and ciphertext bytes; each API's import, validation, and decapsulation | 32-byte shared-secret array |

- All ML-KEM rows include output conversion and the destruction of per-call objects.
  `Criterion::iter` includes the destruction of returned arrays.
- Reused keys are constructed and destroyed outside timing.
  Imported keys are constructed and destroyed inside timing.
- Internal allocations and their cleanup stay timed,
  including the ciphertext and shared-secret buffers and key objects that AWS-LC allocates.
- The harness does not supply reusable scratch storage,
  and it does not equalize the cleanup policies of the libraries.
  The rows measure the selected APIs on valid inputs.
  They do not show identical validation or zeroization guarantees.
- RustCrypto takes part in encapsulation and in reused decoded-key decapsulation.
  Its decoded secret key is built from the fixture seed outside timing.
- Expanded-key generation and import rows use implementations that have expanded-key APIs.

Correctness checks before timing:

- Each deterministic ML-KEM row runs its actual timed closure once outside timing,
  and checks the complete output against the shared fixture.
- Key-generation checks compare both encoded keys.
  Encapsulation checks compare the ciphertext and the secret.
  Decapsulation checks compare the secret.
- AWS-LC's randomized generation and encapsulation closures are checked
  through decapsulation by another implementation.
- A mismatch stops execution before that row is timed.
- Discovery lists identities only.
  It does not replace the correctness checks of the selected rows.

### Argon2 and scrypt

Argon2 comparison groups include `salt16-raw32` in their IDs.

- All rows use the same password, the full 16-byte salt, Argon2 version 0x13,
  the same memory, time, and lane parameters, and 32-byte raw output.
- They consume no entropy and do no PHC encoding.
- Parameter objects and caller-owned output buffers are prepared outside timing.
- Each call includes the selected API's scratch allocation, computation, and scratch cleanup.
- The output buffer is reused, and destroyed outside timing.
- Each library keeps its own cleanup policy.
- Untimed checks compare all 32 output bytes with RustCrypto and,
  where its parameter limits allow the row, with dryoc.
- Argon2 parallel-scaling rows use the same salt and output size, and change the lane count.
- scrypt and PHC fixtures are separate workloads.

The reused-memory groups (`argon2id-owasp/salt16-raw32-reused-memory` and `scrypt-owasp-reused-memory`) isolate work-memory reuse at the OWASP shapes.

- `rscrypto/fresh-allocation` calls `derive`,
  so its allocation, zero fill, computation, and cleanup are timed.
- `reused-memory` rows lend one buffer that is allocated and freed outside timing.
  `rscrypto` still clears every block it used, inside timing.
- `rustcrypto/reused-memory` calls `hash_password_into_with_memory`, built without RustCrypto's `zeroize` feature.
  It does not clear its buffer, so the rows do not do equal cleanup.
- Untimed checks compare all 32 output bytes with `derive`.

## Measure locally

### Settings

[`.config/criterion.json`](../.config/criterion.json) supplies one configuration for every Criterion harness,
also for direct Cargo invocations: 20 samples, 100 ms warmup, 400 ms requested measurement time,
10,000 bootstrap resamples, 95% confidence, 5% significance, and a 1% noise threshold.
Benchmark groups must not override these settings.

`warmup_ms=`, `measure_ms=`, and `sample_size=` override the shared defaults for every selected case in that invocation.
Their `BENCH_` environment variables have lower precedence than explicit arguments.
Boolean controls reject unknown values and empty strings.

These are bounded development defaults, not a promise of statistical precision.
Inspect the confidence intervals.
Repeat a focused selection when the uncertainty cannot support the claim you want to make.
For slow operations, Criterion can extend the requested measurement window to collect the requested
samples.
`argon2id` includes small, OWASP, and parallel workloads.
No opt-in for expensive workloads is needed.

### Time budget

`just bench` limits the whole pipeline to 10 minutes: build, discovery, measurement, analysis,
and result verification.
`just bench-structural` and `just profile` use the same limit, including their builds.
You can lower the limit, but it cannot be more than 600 seconds.

- Shutdown starts before the deadline.
  It keeps up to five seconds to save failed-run evidence, then stops the remaining child processes.
- A run that times out exits with status 124.
  Partial results are not a complete run.
- A plan whose requested sampling windows alone use up the budget is rejected before measurement.
  Build cost, analysis, and slow operations can still make a smaller plan reach the deadline.
- Direct Cargo invocation limits each Criterion harness separately.
  Use `just bench` to limit a selection that spans more than one harness, together with its builds.
  Use `just bench-structural` for the structural benchmark deadline.

A planned comparison must fit within this limit, including every repetition.
Do not split an over-budget comparison across repeated invocations to bypass it.
If required evidence cannot fit, record that requirement as unmet. A smaller future
experiment needs a prospectively stated question and gates; it does not complete
an older, larger qualification plan. Preserve historical frozen plans and failed
captures unchanged; their old budgets do not authorize rerunning them.

### Select cases

Use an algorithm or family selector, or select explicit benchmark targets with `bench=<target>`
(`bench=<csv>` for more than one).
`filter=<pattern>` limits the selected algorithms or targets.
For example, `sha256 filter=rscrypto` stays within SHA-256, but `bench=sha2 filter=rscrypto` searches the whole SHA-2 target.

- Repeat `filter=<pattern>` for more than one pattern.
- Each build configuration is listed once.
- A lightweight invocation of the same executable matches all patterns with Criterion's regex
  engine, without building benchmark fixtures.
- The matched cases run as one measurement process for each build configuration.
  The harness reads the resolved case set from a file and applies an anchored, escaped union filter.
- A pattern that matches no cases fails before measurement.
- Patterns are passed exactly as written.
  Commas are regex characters, not separators.
  Quote each argument for your shell.
- `BENCH_FILTER` supplies one literal pattern, in addition to any `filter=` arguments.
- Empty `filter=` values are rejected.
  For an unfiltered run, omit the argument.
- Positional selectors accept catalog names.
  Use `filter=` for raw regexes.
- An exact algorithm name selects only that algorithm.
  Use a family name, such as `crc64`, to select more than one.
- `blake2` includes all implementations and operations, including the dryoc one-shot and keyed cases.

### Discover cases

Discover the actual cases before you choose a measurement scope:

```bash
just bench crc64-nvme --list
just bench bench=sha2 --list
just bench blake3 --diag --list
just bench bench=aead_kernels --list
```

`--list` builds the selected configuration and lists its cases.
It does not measure, create a run, or copy baseline data.
It uses the same filters and case deduplication as measurement.
Each row shows its benchmark binary, the exact case name, and its work class:

- `ordinary`: public operation and comparison workloads.
- `expensive`: high-cost workloads that the catalog declares,
  including password hashing, PBKDF2, and RSA private signing.
- `diagnostic`: internal components, backend experiments, and overhead probes.

The classes come from `.config/benchmark-matrix.json`.
They describe the intent of a workload, not its measured duration,
and they give no timing guarantee.
A diagnostic case can also be costly.
Classes do not block execution.

`--diag` (or `diag=true`) enables `diag` and the internal compiler cfg for the selected benchmark builds.
Some target configurations already need it.
The cases depend on the compiled features and the host capabilities.
Dedicated diagnostic targets, such as `aead_kernels`, need explicit selection.
Generic runs include the catalog's required Criterion targets.

### Run

Run the narrowest useful case:

```bash
just bench bench=sha2
just bench bench=auth filter='^ecdsa-p256/'
just bench sha256 'filter=^sha256/rscrypto/\d+$' 'filter=^sha256/rscrypto/[0-9]{1,3}$'
just bench p256-ecdh
just bench p384-ecdh
just bench mlkem
```

Explicit targets, including unfiltered `bench=sha2`, run without a scope override.

- P-256 ECDH uses the `p256-ecdh` selector.
  Exact profiling uses the `auth` catalog target that owns it.
  Its rows compare caller-filled generation, public derivation, canonical SEC1 parsing, agreement,
  and a TLS-shaped two-party roundtrip.
- P-384 ECDH uses the `p384-ecdh` selector, with public-derivation, parsing, and agreement rows.

The raw target results and the overview are the only performance record.

### Run files

- `requests.json` keeps the resolved target and filter requests and the run budget.
- `plan.json` records, for each configuration: the selected cases, the Cargo command and artifact,
  the executable hash, the compatibility evidence, the effective settings, the baseline cases,
  the execution command, and the output location.
- Raw results are under `criterion/<binary>-<configuration-id>/`, in Criterion's directory layout.
- `output.txt` is the build, discovery, and measurement log.
- `source.json` and `source-state.json` identify the source files and the worktree.

The runner verifies all planned measurements once before it marks the run complete.

The shared environment collector separates build inputs from runtime controls,
including Rayon thread controls, CRC backend overrides, and `RSCRYPTO_FORCE_AVX512`.
For the `auth` benchmark on an AVX2-capable x86-64 host,
`RSCRYPTO_BENCH_DISABLE_IFMA=1` removes IFMA from the detected capabilities before initialization.
This measures the production AVX2 Curve25519 fallback on IFMA hosts.
It does not change the separate Linux assembly paths.
Known runtime controls that are not set are recorded as explicit JSON nulls.
Benchmark plans and profile metadata carry the same compatibility evidence.

Criterion measures elapsed time.
`just bench-structural` uses Gungraun and Valgrind to count instructions and cache events on supported Linux hosts.
Those counts do not prove wall-clock speed.

## Profile locally

After a benchmark shows a specific cost, inspect it:

```bash
just profile sha2 --list
just profile sha2 'sha256/rscrypto/64' 10
just profile blake3 --diag --list
just perf-codegen sha2 -- --asm <function>
just perf-llvm-lines sha2 -- --filter <pattern>
```

- Profiling needs one exact case name.
- `--list` builds the selected target and lists its cases without recording.
- Capture checks that the name occurs exactly once,
  then runs that executable with only the resolved case selected.
- Unrelated workload groups skip fixture construction.
- The requested duration applies to that case.
  Process startup and profiler overhead add to the total elapsed time.

Each local Samply capture gets its own directory under `target/profiles/`.
It contains `profile.json.gz`, `cases.json`, `metadata.json`, a log, and source evidence.

- The metadata records the exact case, the executable path and SHA-256,
  the Cargo artifact description, the build and capture commands, compiler and tool versions,
  the build and runtime environment, and the capture outcome.
- The source evidence records input hashes, the revision, and the worktree status.
- Keep the matching executable and its symbols when you investigate a saved profile.

Transferred CI captures instead keep the verified input bundle, raw `perf.data`, the native `perf-report.txt`,
host and capability facts, metadata, status, and an outer sealed manifest.
Only a complete sampled capture exits successfully.
A partial, unavailable, or failed capture stays downloadable and keeps the workflow from passing.

## Build configuration

Benchmarking and profiling share the Cargo command and the CPU-flag policy.

- Both use the `bench` profile.
  It inherits the release optimization settings and keeps debug symbols, without stripping.
- Explicit `RUSTFLAGS` or `CARGO_ENCODED_RUSTFLAGS` take precedence.
  Otherwise, local macOS runs use `-C target-cpu=native`.
  Match these flags, the target, and the Cargo features when you compare with a deployment build.
- Each target uses its explicit catalog features with Cargo defaults disabled.
  Algorithm selectors, raw filters, and multi-target selection do not change this.
- Each distinct target configuration is built once, before its selected cases run.
- Profiling and code inspection use the same catalog target configuration.
  `--diag` enables the same feature and internal compiler cfg in each command.
- Only the BLAKE3 and password-hashing targets enable `parallel`, where their workloads use it.
- Cargo ignores the panic setting for benchmarks,
  so the release profile's `panic = "abort"` stays a difference
  ([Cargo profiles](https://doc.rust-lang.org/cargo/reference/profiles.html)).
